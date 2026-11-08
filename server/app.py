"""Real model server. Same SSE events as web/mock-server/server.py.

Pipeline: prompt -> CLAP text embedding -> DDIM + CFG on 4x20x64 latents
-> VAE decoder -> 80x256 mel -> BigVGAN -> 22.05 kHz wav.

Needs these files next to it: unet_fsd_ema.pt, vae_gan2.pt, mel_stats.npy
Run locally:  uvicorn app:app --port 8000
"""

import base64
import io
import json
import os
import sys
import threading
import time
import wave

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import ClapModel, ClapProcessor

from model import SCALE, T, Autoencoder, UNet

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "BigVGAN"))
import bigvgan  # noqa: E402

SR = 22050
UNET_FILE = os.environ.get("UNET_FILE", "unet_fsd_ema.pt")
torch.set_num_threads(os.cpu_count() or 2)
torch.set_grad_enabled(False)

# ---------- load everything once ----------
betas = torch.linspace(0.0001, 0.02, T)
alpha_bar = torch.cumprod(1 - betas, dim = 0)

unet = UNet()
unet.load_state_dict(torch.load(os.path.join(HERE, UNET_FILE), map_location = "cpu"))
unet.eval()

vae = Autoencoder()
vae.load_state_dict(torch.load(os.path.join(HERE, "vae_gan2.pt"), map_location = "cpu"))
vae.eval()

MIN, MAX = np.load(os.path.join(HERE, "mel_stats.npy"))

clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel = False)
vocoder.remove_weight_norm()
vocoder.eval()

lock = threading.Lock()
waiting = 0

app = FastAPI(title = "text-to-sound")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["https://abrandonwang.github.io"],
    allow_origin_regex = r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods = ["GET"],
    allow_headers = ["*"],
)


# ---------- helpers ----------
def text_embedding(prompt):
    t_in = proc(text = [prompt], return_tensors = "pt", padding = True)
    out = clap.get_text_features(**t_in)
    c = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(c, dim = 1)


def latent_to_mel(z):
    """Scaled latent [1, 4, 20, 64] -> mel in [-1, 1], shape [1, 1, 80, 256]."""
    return vae.decoder(z / SCALE).clamp(-1, 1)


def mel_img(mel):
    m = ((mel[0, 0].numpy() + 1) / 2).clip(0, 1)
    return {"bins": 80, "frames": 256, "data": base64.b64encode((m * 255).round().astype(np.uint8).tobytes()).decode()}


def wav_b64(y):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def sse(ev):
    return f"data: {json.dumps(ev)}\n\n"


def ms(start):
    return round((time.perf_counter() - start) * 1000)


def run(prompt, guidance, steps, seed):
    """Sync generator of SSE strings. FastAPI runs it in a worker thread."""
    global waiting
    t0 = time.perf_counter()
    waiting += 1
    try:
        yield sse({"type": "queued", "position": waiting - 1})
        lock.acquire()  # one generation at a time
    finally:
        waiting -= 1
    try:
        try:
            with torch.inference_mode():
                # 1. CLAP
                yield sse({"type": "stage_start", "stage": "encode"})
                s = time.perf_counter()
                c = text_embedding(prompt)
                bars = c[0].view(64, 8).mean(1)
                bars = (bars / (bars.abs().max() + 1e-8)).round(decimals = 3).tolist()
                yield sse({"type": "embedding", "values": bars})
                yield sse({"type": "stage_end", "stage": "encode", "ms": ms(s), "info": "512-d CLAP embedding"})

                # 2. DDIM + CFG (cond and uncond in one batch of 2)
                yield sse({"type": "stage_start", "stage": "denoise"})
                s = time.perf_counter()
                g = torch.Generator().manual_seed(seed)
                x = torch.randn(1, 4, 20, 64, generator = g)
                cc = torch.cat([c, torch.zeros_like(c)])
                ts = np.linspace(T - 1, 0, steps).round().astype(int)
                for i, t in enumerate(ts):
                    st = time.perf_counter()
                    t_batch = torch.full((2,), int(t))
                    eps_label, eps_none = unet(torch.cat([x, x]), t_batch, cc).chunk(2)
                    eps = eps_none + guidance * (eps_label - eps_none)
                    ab = alpha_bar[t]
                    ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
                    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
                    x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
                    step = i + 1
                    ev = {"type": "denoise_step", "step": step, "total": steps, "timestep": int(t),
                          "noise": float(torch.sqrt(1 - ab)), "ms": ms(st)}
                    if step == 1 or step % 5 == 0 or step == steps:
                        ev["preview"] = mel_img(latent_to_mel(x0))
                    yield sse(ev)
                yield sse({"type": "stage_end", "stage": "denoise", "ms": ms(s),
                           "info": f"{steps} DDIM steps · cfg {guidance:.1f}"})

                # 3. VAE decode
                yield sse({"type": "stage_start", "stage": "decode"})
                s = time.perf_counter()
                mel = latent_to_mel(x)
                yield sse({"type": "stage_end", "stage": "decode", "ms": ms(s), "info": "80 × 256 log-mel"})

                # 4. BigVGAN
                yield sse({"type": "stage_start", "stage": "vocode"})
                s = time.perf_counter()
                m = (mel + 1) / 2 * (MAX - MIN) + MIN
                audio = vocoder(m.squeeze(1).float())[0, 0].numpy()
                secs = len(audio) / SR
                yield sse({"type": "stage_end", "stage": "vocode", "ms": ms(s), "info": f"{secs:.2f} s @ 22.05 kHz"})

            yield sse({"type": "result", "mel": mel_img(mel), "sampleRate": SR, "seconds": secs, "wav": wav_b64(audio)})
            yield sse({"type": "done", "totalMs": ms(t0)})
        except Exception as e:  # send the error to the page instead of a dead stream
            yield sse({"type": "error", "message": str(e)})
    finally:
        lock.release()


# ---------- routes ----------
@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "mock": False, "unet": UNET_FILE}


@app.get("/generate")
def generate(
    prompt: str = Query(..., min_length = 1, max_length = 200),
    guidance: float = Query(3.0, ge = 1, le = 10),
    steps: int = Query(50, ge = 1, le = 100),
    seed: int = 0,
):
    return StreamingResponse(run(prompt, guidance, steps, seed), media_type = "text/event-stream",
                             headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
