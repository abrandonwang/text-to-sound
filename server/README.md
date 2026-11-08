---
title: text-to-sound
emoji: 🔊
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# text-to-sound model server

The backend for https://abrandonwang.github.io/text-to-sound/.
`GET /generate?prompt=&guidance=&steps=&seed=` streams server-sent events
(same format as `web/mock-server/server.py`); `GET /health` wakes it up.

Pipeline: CLAP text embedding → 50 DDIM steps with classifier-free guidance on 4 × 20 × 64 latents
→ VAE decoder → 80 × 256 mel → BigVGAN → 3 s of 22.05 kHz audio.

## Deploying to a Hugging Face Space

1. Create a Space: SDK **Docker**, template **Blank**, hardware **CPU basic** (free).
2. Upload everything in this folder, plus these three files from Google Drive:
   - `checkpoints/unet_fsd_ema.pt`
   - `checkpoints/vae_gan2.pt`
   - `data/mel_stats.npy`
3. Wait for the build (about 5–10 min the first time). `https://<user>-<space>.hf.space/health` should return `{"ok": true}`.

## Running locally

Put the three files above in this folder, then:

```bash
pip install torch -r requirements.txt
git clone https://github.com/NVIDIA/BigVGAN
uvicorn app:app --port 8000
```
