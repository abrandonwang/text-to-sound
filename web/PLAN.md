# Showcase site plan

A public page where anyone types a sound and watches the model make it, step by step.
Built against a mock backend so it can be finished before the model is.

## 1. What's built so far

| Part | Where | Status |
|---|---|---|
| Frontend (React + Vite + TS) | `frontend/` | Done against mocks |
| In-browser mock backend | `frontend/src/api/mockClient.ts`, `frontend/src/mock/synth.ts` | Done. Default when no server is set |
| Python mock server (FastAPI, SSE) | `mock-server/server.py` | Done. Same events, proves the SSE path |
| Real model server | — | After Phase 6 |
| Precomputed gallery | — | Dropped from the page in the redesign; see §6 |
| GitHub Pages deploy | — | Workflow below, not added yet |

Run it locally: see `README.md`.

**Design (v2):** one full-screen page, very dark. Background is a canvas "LED panel"
(`LedField.tsx`): every cell is one RGB pixel with bloom. Idle, it shows a travelling neon
wave. During generation it shows the actual spectrogram the model is producing, mirrored
around the centre line (low frequencies in the middle), starting as static and resolving as
noise is removed. While the audio plays, a white playhead sweeps across and loudness drives
brightness (Web Audio analyser). The only UI is a centred text input and a faded text column
of steps on the left (`Steps.tsx`) that light up as each step runs.

## 2. Architecture

```
 GitHub Pages (static)                      Model server (Python, CPU or GPU)
 ┌──────────────────────┐   GET /generate   ┌───────────────────────────────┐
 │ React app            │ ───────────────▶  │ FastAPI                       │
 │  centre input        │                   │  CLAP → DDIM+CFG → VAE → BigVGAN
 │  step text column    │ ◀─ SSE events ─── │  yields one event per step    │
 │  LED spectrogram bg  │                   └───────────────────────────────┘
 └──────────────────────┘
```

- **Frontend** is plain static files (`npm run build` → `dist/`). It never needs a server of its own.
- **Backend** is one endpoint, `GET /generate?prompt=&guidance=&steps=&seed=`, which returns
  `text/event-stream`. Every message is one JSON object. The format lives in
  `frontend/src/api/events.ts` and is the contract between the two halves.
- **Why SSE, not WebSocket:** the data only flows one way (server → browser), SSE works over
  plain HTTP through any proxy, the browser reconnects for you, and FastAPI can stream it from
  a generator in a few lines. WebSockets are only worth it if we later add "cancel" or "change
  guidance mid-run".
- **Switching backends** is one env var: `VITE_API_URL`. Unset = in-browser mock.

### Event stream

| Event | When | Payload |
|---|---|---|
| `queued` | request accepted | position in queue |
| `stage_start` / `stage_end` | each of the 4 stages | stage id, wall time, short info string |
| `embedding` | after CLAP | 64 values for the bar visual (downsampled from 512) |
| `denoise_step` | every DDIM step | step, total, timestep `t`, noise level √(1−ᾱₜ), ms, optional `preview` |
| `result` | end | final mel image + base64 WAV |
| `done` / `error` | end | total time / message |

`preview` is the mel you get by decoding the current latent estimate with the VAE. Sending it
every 5 steps is a good default: 10 extra VAE decodes per run is cheap, and each preview is
only ~27 KB (80 × 256 bytes, base64). Which estimate to show is a choice to make with the real
model:

- **x_t** (the noisy latent itself): looks like the mock, noise fading out. Dramatic.
- **predicted x₀** (`(x_t − √(1−ᾱ)·ε̂) / √ᾱ`): shows what the model "thinks" the final sound
  is at each step. Blurry early, sharp late. More informative, and what most demos show.

Try both once Phase 6 works.

## 3. The real server (after Phase 6)

Copy `mock-server/server.py` to `server/app.py` and replace the fake blocks:

1. Load CLAP, U-Net (EMA weights), VAE decoder and BigVGAN once at startup.
2. `encode`: CLAP text embedding (and the empty-prompt embedding for CFG, cached).
3. `denoise`: DDIM loop; batch cond + uncond into one forward pass; `yield` an event each step;
   every 5 steps also VAE-decode the estimate and send it as `preview`.
4. `decode`: VAE decode the final latent → 80 × 256 log-mel. Un-normalise it the same way the
   training data was normalised.
5. `vocode`: BigVGAN → waveform → 16-bit WAV.
6. Run the model work in a thread (`asyncio.to_thread`) or with `torch.inference_mode()` in a
   sync generator, so the event loop can keep flushing events.

Watch out: the mel settings (n_fft, hop 256, 80 bins, fmin/fmax) must match the BigVGAN
checkpoint exactly, or the audio will be garbage.

## 4. Open question: is CPU fast enough?

The models are small, so CPU might be fine. Rough guesses, **to be measured** once Phase 6 exists:

| Stage | Size | CPU guess (2 vCPU) |
|---|---|---|
| CLAP text encoder | ~125M params, 1 short sequence | 50–150 ms |
| U-Net × 50 DDIM steps (CFG batch of 2) | ~9M params, 4 × 20 × 64 input | 0.5–2 s total |
| VAE decoder (+ ~10 previews) | small | 0.1–0.5 s |
| BigVGAN, ~3 s of audio | ~14M (base) – 112M (v2) params | 1–5 s |

How to measure: a small script that loads all four models on CPU, runs 10 generations, and
prints the median time per stage. Run it on a laptop, then on the target host. Things that
might help if it's too slow: `torch.set_num_threads`, fewer DDIM steps (25), the smaller
BigVGAN, `torch.compile`, or ONNX export.

**Decision rule:** if a full run is under ~10 s on free CPU, use free CPU. Otherwise use GPU.

## 5. Hosting options

Frontend: **GitHub Pages**, free. Decided.

Backend options (prices change; check before choosing):

| Option | Cost | Cold start | Notes |
|---|---|---|---|
| **HF Spaces, CPU basic (Docker)** | Free (2 vCPU, 16 GB) | Sleeps after ~48 h idle; wake takes ~30 s–2 min | Easiest. Runs our FastAPI as-is. One request at a time is realistic. **Best first choice if CPU is fast enough.** |
| HF Spaces, ZeroGPU | Needs HF PRO (~$9/mo) to host | GPU attached per call, few s overhead | Only works with the Gradio SDK, so our custom SSE endpoint would need to sit behind Gradio's API. More friction. |
| HF Spaces, paid GPU (T4) | ~$0.40/h while awake | Same as CPU when asleep | Set a short sleep timeout to cap cost. |
| **Modal** | Pay per second; free monthly credit (~$30) | ~2–10 s for a small container | Python-native, streams fine, scales to zero. **Best choice if GPU is needed.** |
| Replicate | Pay per second | Can be 10 s–minutes for custom models | Built for request/response, not custom SSE streams. Least good fit. |

### Cold starts

- Show them honestly: the `queued` event already drives a "Waiting for a GPU…" line in the rail.
  Add a `warming` event later if the host tells us it's booting.
- The page can call `GET /health` on load to wake the server early, before the user finishes typing.
- The **gallery** (below) means a visitor hears something good even if the server is asleep.

### Queueing and rate limiting

A public demo will get abused. In the real server:

- **One generation at a time** per CPU worker (`asyncio.Semaphore(1)`); others wait and get
  `queued` events with their position.
- **Per-IP limit**, e.g. 5 requests/minute and 50/day (a small in-memory token bucket is enough
  for one worker).
- **Hard caps:** prompt ≤ 200 chars, steps ≤ 100, guidance 1–10 (the mock server already
  validates these), and a total timeout per request (e.g. 60 s).
- **Cache** results by `(prompt, seed, steps, guidance)`: repeats are instant and free.
- **CORS:** allow only `https://abrandonwang.github.io` (and `localhost` for dev).
- Optional later: Cloudflare Turnstile if bots show up.

## 6. Precomputed example gallery

Visitors should hear good results in zero seconds, before (or instead of) waiting for the server.

1. Pick ~6–12 prompts that the model does well on (cherry-pick seeds, that's normal for demos).
2. Run them offline in a notebook. Save `name.wav` and `name.json` (the mel image in the same
   `MelImage` format) into `frontend/public/gallery/`, plus a `manifest.json`.
3. Show them without breaking the one-input design, e.g. the typewriter placeholder prompts
   become clickable, and clicking one replays its stored events + audio instantly.
   (`web/.gitignore` already allows WAVs in that folder; the root `.gitignore` blocks them elsewhere.)
4. Bonus: also save the per-step events for each example, and replay them, so the gallery can
   show the full step-by-step animation with no server at all.

## 7. Deploying the frontend to GitHub Pages

Not done yet, because it needs a file outside `web/`. When ready, add
`.github/workflows/pages.yml`:

```yaml
name: pages
on:
  push:
    branches: [main]
    paths: [web/frontend/**]
  workflow_dispatch:
permissions: { contents: read, pages: write, id-token: write }
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: '${{ steps.d.outputs.page_url }}' }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22 }
      - run: npm ci && npm run build
        working-directory: web/frontend
        env:
          VITE_API_URL: ${{ vars.API_URL }} # leave unset to ship the in-browser mock
      - uses: actions/upload-pages-artifact@v3
        with: { path: web/frontend/dist }
      - id: d
        uses: actions/deploy-pages@v4
```

Then in GitHub: Settings → Pages → Source: "GitHub Actions". The site lands at
`https://abrandonwang.github.io/text-to-sound/` (Vite is set to `base: './'` so paths work there).
Run `npm install` once first so `package-lock.json` exists for `npm ci`.

## 8. Licensing

Check each of these at the source before launch; this is a summary, not legal advice.

- **Clotho** (training data): the audio clips come from Freesound and each keeps its own
  Creative Commons licence (CC0, CC-BY, and some CC-BY-NC). The dataset terms are on its Zenodo
  record. Practical upshot: a **non-commercial** research demo is fine; credit Clotho and
  Freesound on the page. If any training clips are CC-BY-NC, don't sell the model or its output.
- **CLAP** (LAION-CLAP): the code and the Hugging Face checkpoints are under permissive licences
  (check the exact one on the model card you use). Credit LAION.
- **BigVGAN** (NVIDIA): the BigVGAN v2 code and checkpoints are released under MIT. Keep the
  licence notice and credit NVIDIA.
- **This repo**: MIT (see root `LICENSE`).
- **Generated audio**: say on the page that it's AI-generated and for demo use.
- **Fonts**: Instrument Serif, Inter Tight, JetBrains Mono are all OFL (free to use).

Add a small "Credits" line in the footer listing Clotho/Freesound, LAION-CLAP and BigVGAN once
the real model is live.

## 9. Next steps

1. ☐ Phase 6 lands → write the CPU benchmark script, fill in §4.
2. ☐ Pick the host from §5 based on the numbers.
3. ☐ Build `server/app.py` from the mock; add rate limiting + cache.
4. ☐ Generate the gallery offline.
5. ☐ Add the Pages workflow, set `API_URL`, deploy.
6. ☐ Credits + "AI-generated" note in the footer.
