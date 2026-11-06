"""Mock model server. Streams the same SSE events the real one will.

Run:
    pip install -r requirements.txt
    uvicorn server:app --port 8000 --reload
Then point the frontend at it:
    echo "VITE_API_URL=http://localhost:8000" > ../frontend/.env.local

When Phase 6 works, copy this file to make the real server: keep the event
format and replace each `fake_*` block with the real model call.
"""

import asyncio
import base64
import io
import json
import time
import wave

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

BINS, FRAMES, SR, HOP, NFFT, FMAX = 80, 256, 22050, 256, 1024, 8000
T = 1000
BETAS = np.linspace(1e-4, 0.02, T)
ALPHA_BAR = np.cumprod(1 - BETAS)

app = FastAPI(title="text-to-sound mock")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


# ---------- fake model pieces ----------
def hz_to_bin(f):
    mel = 2595 * np.log10(1 + np.asarray(f) / 700)
    return mel / (2595 * np.log10(1 + FMAX / 700)) * (BINS - 1)


def fake_target_mel(prompt: str, rng: np.random.Generator) -> np.ndarray:
    """A plausible [80, 256] mel in [0, 1]. Much simpler than the in-browser mock."""
    p = prompt.lower()
    m = 0.12 * (1 - np.arange(BINS)[:, None] / BINS) + 0.04 * rng.random((BINS, FRAMES))
    env = np.interp(np.arange(FRAMES), np.linspace(0, FRAMES, 10), rng.random(10))
    if any(w in p for w in ("rain", "water", "stream", "wind", "fire")):
        lo, hi = hz_to_bin(500), hz_to_bin(8000)
        m[int(lo):int(hi)] = np.maximum(m[int(lo):int(hi)], 0.6 * (0.6 + 0.4 * env))
    if any(w in p for w in ("thunder", "engine", "rumble", "traffic")):
        m[: int(hz_to_bin(400))] = np.maximum(m[: int(hz_to_bin(400))], 0.75 * env)
    if any(w in p for w in ("bird", "bell", "siren", "whistle", "chirp")):
        for _ in range(10):
            start, length = rng.integers(0, FRAMES - 20), rng.integers(6, 20)
            f0 = rng.uniform(1500, 4000)
            for i in range(length):
                b = int(hz_to_bin(f0 * 2 ** (0.6 * i / length)))
                if b < BINS:
                    m[b, start + i] = 0.95
    if any(w in p for w in ("dog", "knock", "step", "door", "clap")):
        for f in rng.integers(0, FRAMES - 8, 10):
            m[:, f : f + 4] = np.maximum(m[:, f : f + 4], 0.8)
    if m.max() < 0.5:  # nothing matched
        m = np.maximum(m, 0.5 * rng.random((BINS, FRAMES)) * env)
    return np.clip(m, 0, 1)


def latent_noise(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((20, 64))
    return np.kron(z, np.ones((4, 4))) * 0.8 + rng.standard_normal((BINS, FRAMES)) * 0.35


def noisy(target, noise, t):
    ab = ALPHA_BAR[t]
    return np.clip((np.sqrt(ab) * (target * 2 - 1) + np.sqrt(1 - ab) * noise * 0.7 + 1) / 2, 0, 1)


def fake_vocoder(mel: np.ndarray, rng: np.random.Generator, iters: int = 16) -> np.ndarray:
    """Griffin-Lim standing in for BigVGAN."""
    k = np.arange(NFFT // 2 + 1)
    b = np.clip(hz_to_bin(np.minimum(k * SR / NFFT, FMAX)), 0, BINS - 1)
    lin = np.stack([np.interp(b, np.arange(BINS), mel[:, f]) for f in range(FRAMES)])  # [F, K]
    mag = 10 ** ((-70 + 70 * lin) / 20)
    win = np.hanning(NFFT)
    n = (FRAMES - 1) * HOP + NFFT
    wsum = np.zeros(n)
    for f in range(FRAMES):
        wsum[f * HOP : f * HOP + NFFT] += win**2
    phase = np.exp(2j * np.pi * rng.random(mag.shape))

    def istft(spec):
        y = np.zeros(n)
        frames = np.fft.irfft(spec, NFFT) * win
        for f in range(FRAMES):
            y[f * HOP : f * HOP + NFFT] += frames[f]
        return y / np.maximum(wsum, 1e-3)

    y = istft(mag * phase)
    for _ in range(iters):
        frames = np.stack([y[f * HOP : f * HOP + NFFT] * win for f in range(FRAMES)])
        spec = np.fft.rfft(frames, NFFT)
        y = istft(mag * np.exp(1j * np.angle(spec)))
    y = y[NFFT // 2 : NFFT // 2 + FRAMES * HOP]
    return 0.85 * y / (np.abs(y).max() + 1e-6)


# ---------- encoding ----------
def mel_img(m: np.ndarray) -> dict:
    return {"bins": BINS, "frames": FRAMES, "data": base64.b64encode((m * 255).round().astype(np.uint8).tobytes()).decode()}


def wav_b64(y: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True, "mock": True}


@app.get("/generate")
async def generate(
    prompt: str = Query(..., max_length=200),
    guidance: float = Query(3.0, ge=1, le=10),
    steps: int = Query(50, ge=1, le=200),
    seed: int = 0,
):
    async def events():
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed + sum(map(ord, prompt)))
        ms = lambda s: round((time.perf_counter() - s) * 1000)  # noqa: E731

        yield sse({"type": "queued", "position": 0})
        await asyncio.sleep(0.25)

        # 1. CLAP
        yield sse({"type": "stage_start", "stage": "encode"})
        s = time.perf_counter()
        await asyncio.sleep(0.14)
        target = fake_target_mel(prompt, rng)
        yield sse({"type": "embedding", "values": np.tanh(rng.standard_normal(64) * 0.8).round(3).tolist()})
        yield sse({"type": "stage_end", "stage": "encode", "ms": ms(s), "info": "512-d embedding"})

        # 2. DDIM + CFG
        yield sse({"type": "stage_start", "stage": "denoise"})
        s = time.perf_counter()
        noise = latent_noise(rng)
        timesteps = [round((steps - 1 - i) * T / steps) for i in range(steps)]
        for i, t in enumerate(timesteps):
            st = time.perf_counter()
            await asyncio.sleep(0.055)
            step = i + 1
            ev = {"type": "denoise_step", "step": step, "total": steps, "timestep": t,
                  "noise": float(np.sqrt(1 - ALPHA_BAR[t])), "ms": ms(st)}
            if step == 1 or step % 5 == 0 or step == steps:
                ev["preview"] = mel_img(noisy(target, noise, t))
            yield sse(ev)
        yield sse({"type": "stage_end", "stage": "denoise", "ms": ms(s), "info": f"{steps} DDIM steps · cfg {guidance:.1f}"})

        # 3. VAE decode
        yield sse({"type": "stage_start", "stage": "decode"})
        s = time.perf_counter()
        await asyncio.sleep(0.09)
        yield sse({"type": "stage_end", "stage": "decode", "ms": ms(s), "info": "80 × 256 log-mel"})

        # 4. Vocoder
        yield sse({"type": "stage_start", "stage": "vocode"})
        s = time.perf_counter()
        audio = await asyncio.to_thread(fake_vocoder, target, rng)
        secs = len(audio) / SR
        yield sse({"type": "stage_end", "stage": "vocode", "ms": ms(s), "info": f"{secs:.2f} s @ 22.05 kHz"})

        yield sse({"type": "result", "mel": mel_img(target), "sampleRate": SR, "seconds": secs, "wav": wav_b64(audio)})
        yield sse({"type": "done", "totalMs": ms(t0)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
