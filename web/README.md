# web — showcase site

Public demo page for the model. Separate from the notebooks. Plan and hosting notes: [PLAN.md](PLAN.md).

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

## Layout

```
frontend/src/
  api/events.ts        event format (the contract with the backend)
  api/mockClient.ts    in-browser fake backend
  api/sseClient.ts     EventSource client for a real server
  mock/synth.ts        fake mel, fake noise schedule, Griffin-Lim "vocoder"
  state.ts             reducer: events -> UI state
  components/LedField  full-screen RGB LED canvas (wave -> spectrogram)
  components/Steps     faded step column on the left
mock-server/server.py  FastAPI version of the mock; template for the real server
```
