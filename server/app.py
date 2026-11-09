"""Real model server. Same SSE events as web/mock-server/server.py.

Pipeline: prompt -> CLAP text embedding -> DDIM + CFG on 4x20x64 latents
-> VAE decoder -> 80x256 mel -> BigVGAN -> 22.05 kHz wav.

Needs these files next to it: unet_fsd_ema.pt, vae_gan2.pt, mel_stats.npy
Run locally:  python app.py   (port 7860)
"""

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import wave

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import ClapModel, ClapProcessor

from model import T, Autoencoder, UNet, UNet4, make_betas

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("WEIGHTS_DIR", HERE)  # unet, vae, mel_stats live here
BIGVGAN_DIR = os.environ.get("BIGVGAN_DIR", os.path.join(HERE, "BigVGAN"))
if not os.path.isdir(BIGVGAN_DIR):
    subprocess.run(["git", "clone", "-q", "https://github.com/NVIDIA/BigVGAN", BIGVGAN_DIR], check = True)
# same huggingface_hub patch the notebooks apply with sed
_p = os.path.join(BIGVGAN_DIR, "bigvgan.py")
_src = open(_p).read()
_new = _src.replace("proxies: Optional[Dict],", "proxies: Optional[Dict] = None,").replace(
    "resume_download: bool,", "resume_download: bool = False,")
if _new != _src:
    open(_p, "w").write(_new)
sys.path.insert(0, BIGVGAN_DIR)
import bigvgan  # noqa: E402

SR = 22050
UNET_FILE = os.environ.get("UNET_FILE", "unet_fsd_ema.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 2)
torch.set_grad_enabled(False)

# ---------- load everything once ----------
# Which U-Net, latent scale and noise schedule. A <unet file stem>_config.json next to the weights
# (written by the training notebook) overrides these v3 defaults.
CFG = {"arch": "v3", "scale": 0.6863, "schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02}
_cfg_path = os.path.join(WEIGHTS, os.path.splitext(UNET_FILE)[0] + "_config.json")
if os.path.exists(_cfg_path):
    CFG.update(json.load(open(_cfg_path)))
SCALE = CFG["scale"]
betas = make_betas(CFG["schedule"], CFG["beta_start"], CFG["beta_end"]).to(DEVICE)
alpha_bar = torch.cumprod(1 - betas, dim = 0)

unet = UNet4() if CFG["arch"] == "v4" else UNet()
unet.load_state_dict(torch.load(os.path.join(WEIGHTS, UNET_FILE), map_location = "cpu"))
unet = unet.eval().to(DEVICE)

vae = Autoencoder()
vae.load_state_dict(torch.load(os.path.join(WEIGHTS, "vae_gan2.pt"), map_location = "cpu"))
vae = vae.eval().to(DEVICE)

MIN, MAX = np.load(os.path.join(WEIGHTS, "mel_stats.npy"))

clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().to(DEVICE)
proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel = False)
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(DEVICE)

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
    """str or list of str -> [n, 512] unit vectors."""
    t_in = proc(text = [prompt] if isinstance(prompt, str) else prompt, return_tensors = "pt", padding = True).to(DEVICE)
    out = clap.get_text_features(**t_in)
    c = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(c, dim = 1)


# Prompts known to sound good, each with the seed that sounded best. /suggest finds the
# ones closest in meaning (CLAP text space) to whatever the user typed.
PROMPTS_FILE = os.environ.get("PROMPTS_FILE", os.path.join(HERE, "good_prompts.json"))
GOOD = json.load(open(PROMPTS_FILE))["prompts"]
with torch.inference_mode():
    GOOD_EMB = torch.cat([text_embedding([g["prompt"] for g in GOOD[i:i + 64]]) for i in range(0, len(GOOD), 64)])


def audio_embedding(wavs):
    """List of 22.05 kHz numpy wavs -> [n, 512] CLAP unit vectors (CLAP wants 48 kHz)."""
    a48 = [librosa.resample(w, orig_sr = SR, target_sr = 48000) for w in wavs]
    try:
        inp = proc(audio = a48, sampling_rate = 48000, return_tensors = "pt")
    except TypeError:  # older transformers call it "audios"
        inp = proc(audios = a48, sampling_rate = 48000, return_tensors = "pt")
    out = clap.get_audio_features(**inp.to(DEVICE))
    return F.normalize(out if torch.is_tensor(out) else out.pooler_output, dim = 1)


def latent_to_mel(z):
    """Scaled latent [n, 4, 20, 64] -> mel in [-1, 1], shape [n, 1, 80, 256]."""
    return vae.decoder(z / SCALE).clamp(-1, 1)


def mel_img(mel):
    m = ((mel[0, 0].float().cpu().numpy() + 1) / 2).clip(0, 1)
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


def run(prompt, guidance, steps, seed, n):
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
                bars = c[0].view(64, 8).mean(1).cpu()
                bars = (bars / (bars.abs().max() + 1e-8)).round(decimals = 3).tolist()
                yield sse({"type": "embedding", "values": bars})
                yield sse({"type": "stage_end", "stage": "encode", "ms": ms(s), "info": "512-d CLAP embedding"})

                # 2. DDIM + CFG on n candidates at once (cond and uncond in one batch of 2n).
                # Candidate 0 gets the same noise a single-sample run with this seed would.
                yield sse({"type": "stage_start", "stage": "denoise"})
                s = time.perf_counter()
                g = torch.Generator().manual_seed(seed)
                x = torch.randn(n, 4, 20, 64, generator = g).to(DEVICE)
                cn = c.expand(n, -1)
                cc = torch.cat([cn, torch.zeros_like(cn)])
                ts = np.linspace(T - 1, 0, steps).round().astype(int)
                for i, t in enumerate(ts):
                    st = time.perf_counter()
                    t_batch = torch.full((2 * n,), int(t), device = DEVICE)
                    eps_label, eps_none = unet(torch.cat([x, x]), t_batch, cc).chunk(2)
                    eps = eps_none + guidance * (eps_label - eps_none)
                    ab = alpha_bar[t]
                    ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device = DEVICE)
                    x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
                    x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
                    step = i + 1
                    ev = {"type": "denoise_step", "step": step, "total": steps, "timestep": int(t),
                          "noise": float(torch.sqrt(1 - ab)), "ms": ms(st)}
                    if step == 1 or step % 5 == 0 or step == steps:
                        ev["preview"] = mel_img(latent_to_mel(x0[:1]))
                    yield sse(ev)
                yield sse({"type": "stage_end", "stage": "denoise", "ms": ms(s),
                           "info": f"{steps} DDIM steps · cfg {guidance:.1f}" + (f" · {n} takes" if n > 1 else "")})

                # 3. VAE decode
                yield sse({"type": "stage_start", "stage": "decode"})
                s = time.perf_counter()
                mels = latent_to_mel(x)
                yield sse({"type": "stage_end", "stage": "decode", "ms": ms(s), "info": "80 × 256 log-mel"})

                # 4. BigVGAN, then CLAP keeps the take that best matches the prompt
                yield sse({"type": "stage_start", "stage": "vocode"})
                s = time.perf_counter()
                m = (mels + 1) / 2 * (MAX - MIN) + MIN
                wavs = vocoder(m.squeeze(1).float())[:, 0].cpu().numpy()
                best, info = 0, ""
                if n > 1:
                    sims = (audio_embedding(list(wavs)) @ c.T)[:, 0]
                    best = int(sims.argmax())
                    info = f" · best of {n} (CLAP {float(sims[best]):.2f})"
                audio, mel = wavs[best], mels[best:best + 1]
                secs = len(audio) / SR
                yield sse({"type": "stage_end", "stage": "vocode", "ms": ms(s),
                           "info": f"{secs:.2f} s @ 22.05 kHz" + info})

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
    return {"ok": True, "mock": False, "unet": UNET_FILE, "arch": CFG["arch"], "device": DEVICE}


@app.get("/suggest")
def suggest(q: str = Query(..., min_length = 1, max_length = 200), k: int = Query(3, ge = 1, le = 8)):
    """Closest known-good prompts by meaning: "puppy" -> "a dog barking"."""
    qn = " ".join(q.lower().split())
    with torch.inference_mode():
        sims = (text_embedding(qn) @ GOOD_EMB.T)[0].cpu()
    order = sims.argsort(descending = True).tolist()
    exact = [i for i, g in enumerate(GOOD) if g["prompt"] == qn]
    picks = (exact + [i for i in order if i not in exact])[:k]
    return {"query": q, "exact": bool(exact),
            "suggestions": [{"prompt": GOOD[i]["prompt"], "seed": GOOD[i]["seed"],
                             "similarity": round(float(sims[i]), 3)} for i in picks]}


@app.get("/generate")
def generate(
    prompt: str = Query(..., min_length = 1, max_length = 200),
    guidance: float = Query(3.0, ge = 1, le = 10),
    steps: int = Query(50, ge = 1, le = 100),
    seed: int = 0,
    n: int = Query(4, ge = 1, le = 8),   # takes generated; CLAP picks the best
):
    return StreamingResponse(run(prompt, guidance, steps, seed, n), media_type = "text/event-stream",
                             headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":  # local run: python app.py
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = int(os.environ.get("PORT", 7860)))
