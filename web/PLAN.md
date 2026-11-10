# Showcase site plan

A public page where anyone types a sound and watches the model make it, step by step.
It was built against a mock backend first; it now runs on the real model.

Live: https://abrandonwang.github.io/text-to-sound/

## 1. What's built

| Part | Where | Status |
|---|---|---|
| Frontend (React + Vite + TS) | `frontend/` | Done, live on GitHub Pages |
| In-browser mock backend | `frontend/src/api/mockClient.ts`, `frontend/src/mock/synth.ts` | Done. Used when `VITE_API_URL` is unset |
| Python mock server (FastAPI, SSE) | `mock-server/server.py` | Done. Same events, for testing the SSE path offline |
| Real model server | `../server/app.py`, deployed with `../server/modal_app.py` | Done, live on Modal (T4) |
| GitHub Pages deploy | `../.github/workflows/pages.yml` | Done. Builds on pushes to `web/frontend/**` with `VITE_API_URL` from the `API_URL` repo variable |
| Precomputed gallery | — | Dropped in the redesign; see §6 |
| Credits / "AI-generated" note | — | Not done; see §8 |

Run it locally: see `README.md`.

**Design (v2):** one full-screen page, very dark. Background is a canvas "LED panel"
(`LedField.tsx`): every cell is one RGB pixel with bloom. Idle, it shows a rainbow
backlight glowing around the typed text; after Enter it goes dark and stays still. The only
UI is a centred text input. On Enter it dissolves into a full-screen loading screen
(`Sequence.tsx`) driven only by server events. Each run makes four sounds (four `/generate`
calls, one after another, each with its own seed; the server makes 4 takes per call and keeps the
best by CLAP), so the big label is "sound 1 of 4" and so on, with "waking up the model" or
"waiting in line" before it when that is really happening. One thin bar fills across the whole
run, and a small line under it says what the model is doing ("removing noise 23 / 50", then
"turning it into audio"). The finale (`Finale.tsx`) shows the prompt and four mini players, each
with play, a scrubber and its own download, plus "make another". Text stays readable because the
LED panel switches off its own LEDs around any `[data-quiet]` element.

**Prompt help (never blocking):** free text up to 200 chars is allowed, since the model is
conditioned on CLAP text embeddings and trained on full captions. At most one thing shows at
a time. While a sound name is half typed, the rest appears dimmed in the input. Otherwise,
500 ms after typing stops, one suggestion appears under it: a typo fix or different word from
the FSD50K names in `src/vocab/` (local, 1-2 word prompts only), or else the closest match by
meaning from the server's `GET /suggest` (up to 3 words; must be >= 0.66 CLAP cosine, because
junk input lands on a few "hub" prompts at ~0.6). Tab or a click fills it in, and nothing
generates until Enter, which always uses exactly what is in the box. When the prompt is one of
the server's known-good prompts, the first of the four sounds uses that prompt's best seed. The
page calls `/health` on load to wake the server. In mock mode only the local layer runs.

## 2. Architecture

```
 GitHub Pages (static)                      Modal, T4 GPU, scales to zero
 ┌──────────────────────┐   GET /generate   ┌──────────────────────────────────────┐
 │ React app            │ ───────────────▶  │ FastAPI (server/app.py)              │
 │  centre input        │   GET /suggest    │  CLAP → retrieval blend → DDIM + CFG │
 │  loading sequence    │   GET /health     │  → VAE → BigVGAN → best of 4 by CLAP │
 │  4 mini players      │ ◀─ SSE events ─── │  yields one event per step           │
 └──────────────────────┘                   └──────────────────────────────────────┘
```

- **Frontend** is plain static files (`npm run build` → `dist/`). It never needs a server of its own.
- **Backend** endpoints:
  - `GET /generate?prompt=&guidance=&steps=&seed=&n=` returns `text/event-stream`. Every message is
    one JSON object. The format lives in `frontend/src/api/events.ts` and is the contract between
    the two halves. The page sends guidance 3 and 50 steps; `n` (takes per call) defaults to 4.
  - `GET /suggest?q=&k=` returns the known-good prompts closest in meaning, each with its best seed.
  - `GET /health` wakes the server and reports which U-Net, device and retrieval bank are loaded.
- **Why SSE, not WebSocket:** the data only flows one way (server → browser), SSE works over
  plain HTTP through any proxy, the browser reconnects for you, and FastAPI can stream it from
  a generator in a few lines. WebSockets are only worth it if we later add "cancel" or "change
  guidance mid-run".
- **Switching backends** is one env var: `VITE_API_URL`. Unset = in-browser mock.

### Event stream

| Event | When | Payload |
|---|---|---|
| `queued` | request accepted | position in queue |
| `stage_start` / `stage_end` | each of the 4 stages (`encode`, `denoise`, `decode`, `vocode`) | stage id, wall time, short info string |
| `embedding` | after CLAP | 64 values for the bar visual (downsampled from 512) |
| `denoise_step` | every DDIM step | step, total, timestep `t`, noise level √(1−ᾱₜ), ms, optional `preview` |
| `result` | end | final mel image + base64 WAV |
| `done` / `error` | end | total time / message |

`preview` is the **predicted x₀** (`(x_t − √(1−ᾱ)·ε̂) / √ᾱ`) of the first take, decoded by the VAE:
what the model "thinks" the final sound is at that step. The server sends it on step 1, every 5th
step and the last step, ~27 KB each (80 × 256 bytes, base64).

## 3. The model server

Built from `mock-server/server.py`; details, scripts and deploy steps are in `../server/README.md`.

1. Loads CLAP, U-Net v4 (EMA weights), VAE decoder and BigVGAN once at startup, plus the CLAP
   audio bank used for retrieval conditioning and the known-good prompt list for `/suggest`.
2. `encode`: CLAP text embedding, then blended with the 8 nearest training-audio embeddings.
3. `denoise`: 50 DDIM steps on 4 takes, conditional and unconditional passes batched together.
4. `decode`: VAE decode → 80 × 256 log-mel, un-normalised with the training `mel_stats.npy`.
5. `vocode`: BigVGAN on all takes, CLAP picks the best (near-silent takes rejected), then high-pass,
   loudness normalisation and fades → 16-bit WAV.
6. The model work runs in a sync generator under `torch.inference_mode()`, which FastAPI runs in a
   worker thread, so the event loop keeps flushing events.

## 4. Speed

Measured locally (Apple M4, `DEVICE=mps`): U-Net v4 ~17 ms per sample-step, BigVGAN ~1.6 s per
clip (the bottleneck), so ~15 s for one `/generate` call with 4 takes. Timings on the Modal T4 haven't
been written down yet; the `stage_end` events report them per run.

CPU was never benchmarked: U-Net v4 is 67.5 M params, 50 steps × 4 takes × 2 (CFG), so the decision
went straight to a GPU. Cold starts take ~20 s, which is why the page pings `/health` on load.

## 5. Hosting

- **Frontend:** GitHub Pages. Done.
- **Backend:** Modal, T4 GPU (`../server/modal_app.py`):
  - scales to zero 5 minutes after the last request;
  - `max_containers = 1` caps the bill at one GPU;
  - up to 8 concurrent requests per container; they wait on the server's lock and get `queued` events;
  - 120 s timeout per request;
  - CLAP and BigVGAN weights are baked into the image; the U-Net, VAE, `mel_stats.npy` and the
    retrieval bank live on the `tts-weights` volume.

Other hosts were considered (HF Spaces CPU or GPU, Replicate). Modal won because it streams custom
SSE as-is, bills per second and scales to zero.

### Abuse limits

Done:

- **One generation at a time** (a lock in `app.py`); the rest get `queued` events with their position.
- **Hard caps:** prompt ≤ 200 chars, steps ≤ 100, guidance 1–10, takes ≤ 8.
- **CORS:** only `https://abrandonwang.github.io` and `localhost`.
- **Cost cap:** one container at most.

Not done:

- Per-IP rate limit (e.g. 5 requests/minute, 50/day).
- Result cache by `(prompt, seed, steps, guidance)`.
- Cloudflare Turnstile, only if bots show up.

## 6. Precomputed example gallery (dropped)

The v2 design has no room for a gallery, and `/suggest` plus the known-good seeds cover the
"hear something good fast" goal once the server is warm. If cold starts become a problem, the idea
still works: run known-good prompts offline, save their events and audio into
`frontend/public/gallery/` (`web/.gitignore` already allows WAVs there), and replay them instantly.

## 7. GitHub Pages deploy

Done: `../.github/workflows/pages.yml` builds `web/frontend` on pushes to `main` that touch
`web/frontend/**` (or the workflow), with `VITE_API_URL` set from the `API_URL` repository variable,
and deploys to `https://abrandonwang.github.io/text-to-sound/` (Vite uses `base: './'`). Changing the
server URL means updating `API_URL` and rerunning the workflow. Server-only changes don't redeploy
the site; they need `modal deploy`.

## 8. Licensing

Check each of these at the source; this is a summary, not legal advice.

- **Clotho** and **FSD50K** (training data): the audio clips come from Freesound and each keeps its
  own Creative Commons licence (CC0, CC-BY, and some CC-BY-NC). Practical upshot: a
  **non-commercial** research demo is fine; credit Clotho, FSD50K and Freesound on the page. Because
  some training clips are CC-BY-NC, don't sell the model or its output.
- **CLAP** (LAION-CLAP): the code and the Hugging Face checkpoints are under permissive licences
  (check the exact one on the model card). Credit LAION.
- **BigVGAN** (NVIDIA): the BigVGAN v2 code and checkpoints are released under MIT. Keep the
  licence notice and credit NVIDIA.
- **This repo**: MIT (see root `LICENSE`).
- **Generated audio**: say on the page that it's AI-generated and for demo use.
- **Fonts**: Instrument Serif, Inter Tight, JetBrains Mono are all OFL (free to use).

The page doesn't have the credits line or the AI-generated note yet.

## 9. Next steps

1. ☑ Phase 6 lands; measure speed (§4).
2. ☑ Pick the host (Modal T4).
3. ☑ Build `server/app.py` from the mock.
4. ☑ Add the Pages workflow, set `API_URL`, deploy.
5. ☐ Credits (Clotho, FSD50K, Freesound, LAION-CLAP, BigVGAN) and an "AI-generated" note in the footer.
6. ☐ Per-IP rate limit and a result cache on the server.
7. ☐ Stop the local typo-fix (`src/vocab/sounds.ts`) from offering names the model can't do yet
   (drums, speech), which `/suggest` already leaves out.
