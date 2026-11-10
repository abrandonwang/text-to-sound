# web — showcase site

Public demo page for the model, live at https://abrandonwang.github.io/text-to-sound/ against the
real model server in [`../server`](../server). Separate from the notebooks. Plan and hosting notes: [PLAN.md](PLAN.md).

## Run the site (in-browser mock, no server)

```bash
cd web/frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Run against the Python mock server (tests the real SSE path)

```bash
cd web/mock-server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn server:app --port 8000
```

Then, in another terminal:

```bash
cd web/frontend
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Delete `.env.local` to go back to the in-browser mock.

## Run against the real model

Start the model server (see [`../server/README.md`](../server/README.md)), which listens on port 7860:

```bash
cd server
DEVICE=mps python app.py
```

Then point the site at it:

```bash
cd web/frontend
echo "VITE_API_URL=http://localhost:7860" > .env.local
npm run dev
```

`/suggest` only works with a real server; with the mocks, only the local typo-fix runs. The
deployed site gets its server URL from the `API_URL` repository variable at build time.

## Layout

```
frontend/src/
  api/events.ts        event format (the contract with the backend)
  api/mockClient.ts    in-browser fake backend
  api/sseClient.ts     EventSource client for a real server
  api/suggest.ts       /suggest client (closest known-good prompts) and the /health wake-up
  mock/synth.ts        fake mel, fake noise schedule, Griffin-Lim "vocoder"
  vocab/               FSD50K sound names and the local typo-fix / completion matcher
  state.ts             reducer: events -> UI state
  components/LedField  full-screen RGB LED canvas: rainbow backlight when idle, dark otherwise
  components/Sequence  full-screen step-by-step animation after you press enter
  components/Finale    the reveal: prompt as title, four mini players, make another
mock-server/server.py  FastAPI version of the mock; the real server (../server/app.py) started from it
```
