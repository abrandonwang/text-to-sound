# text-to-sound model server

The backend for https://abrandonwang.github.io/text-to-sound/.
`GET /generate?prompt=&guidance=&steps=&seed=` streams server-sent events
(same format as `web/mock-server/server.py`); `GET /health` wakes it up.

Pipeline: CLAP text embedding, pulled toward the nearest training-audio embeddings → 50 DDIM steps
with classifier-free guidance on 4 × 20 × 64 latents (4 takes at once) → VAE decoder → 80 × 256 mel
→ BigVGAN → CLAP keeps the take closest to the prompt → high-pass, loudness and fades
→ 3 s of 22.05 kHz audio.

## Deploy on Modal (T4 GPU, scales to zero)

```bash
pip install modal && modal setup
modal volume create tts-weights
modal volume put tts-weights unet_v4_ema.pt /            # from Drive checkpoints/
modal volume put tts-weights unet_v4_ema_config.json /   # from Drive checkpoints/ (scale + noise schedule)
modal volume put tts-weights vae_gan2.pt /               # from Drive checkpoints/
modal volume put tts-weights mel_stats.npy /             # from Drive data/
modal volume put tts-weights clap_audio_bank.npy /       # from make_bank.py (optional, see Scripts)
cd server && modal deploy modal_app.py
```

The deploy prints the URL. Set it as the `API_URL` repository variable on GitHub and rerun the Pages workflow.

The server loads `unet_v4_ema.pt` if it is on the volume and falls back to the older
`unet_fsd_ema.pt` if not; `/health` says which one is live. `UNET_FILE` overrides the choice.

## Run locally

Put the weight files in this folder (or point `WEIGHTS_DIR` at them), then:

```bash
pip install -r requirements.txt
python app.py              # CPU; DEVICE=mps python app.py uses a Mac GPU
```

## Scripts

`eval.py` and `score_prompts.py` load the same models as the server and take the same
`WEIGHTS_DIR` / `UNET_FILE` / `DEVICE`.

- `python make_bank.py <data folder>` writes `clap_audio_bank.npy`, the CLAP audio embeddings of the
  training clips. With it the server conditions on the prompt blended with its 8 nearest clips, which
  closes most of CLAP's text-audio gap (mean CLAP score 0.328 → 0.387 in a paired 20-prompt × 3-seed
  test, Clotho-only bank). The deployed bank is Clotho-only (22,848 clips); run the script on the Drive
  data folder to include FSD50K. Without the file the server uses the text embedding alone.
- `python eval.py` prints the mean CLAP text-audio score over a fixed prompt set: the number
  to compare two checkpoints or two sampler settings with. Current: `unet_v4_ema.pt` with the bank,
  cfg 3, 50 steps, 4 takes → mean 0.415, best 0.474.
- `python score_prompts.py` rescores `good_prompts.json`: the best seed and CLAP score for every
  prompt `/suggest` can offer. Rerun it after changing the U-Net.
