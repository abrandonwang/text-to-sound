# text-to-sound model server

The backend for https://abrandonwang.github.io/text-to-sound/.
`GET /generate?prompt=&guidance=&steps=&seed=` streams server-sent events
(same format as `web/mock-server/server.py`); `GET /health` wakes it up.

Pipeline: CLAP text embedding → 50 DDIM steps with classifier-free guidance on 4 × 20 × 64 latents
→ VAE decoder → 80 × 256 mel → BigVGAN → 3 s of 22.05 kHz audio.

## Deploy on Modal (T4 GPU, scales to zero)

```bash
pip install modal && modal setup
modal volume create tts-weights
modal volume put tts-weights unet_fsd_ema.pt /     # from Drive checkpoints/
modal volume put tts-weights vae_gan2.pt /         # from Drive checkpoints/
modal volume put tts-weights mel_stats.npy /       # from Drive data/
cd server && modal deploy modal_app.py
```

The deploy prints the URL. Set it as the `API_URL` repository variable on GitHub and rerun the Pages workflow.

## Run locally (CPU)

Put the three weight files in this folder, then:

```bash
pip install -r requirements.txt
python app.py
```
