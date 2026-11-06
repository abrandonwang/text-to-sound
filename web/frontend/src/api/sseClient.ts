// Talks to a real (or mock Python) server: GET {base}/generate?... returns
// text/event-stream, one JSON GenEvent per `data:` line.

import type { GenEvent, GenRequest } from './events'

export function sseGenerator(baseUrl: string) {
  return async function* sseGenerate(req: GenRequest, signal: AbortSignal): AsyncGenerator<GenEvent> {
    const q = new URLSearchParams({
      prompt: req.prompt,
      guidance: String(req.guidance),
      steps: String(req.steps),
      seed: String(req.seed),
    })
    const es = new EventSource(`${baseUrl.replace(/\/$/, '')}/generate?${q}`)
    const queue: GenEvent[] = []
    let wake: (() => void) | null = null
    let finished = false
    const push = (ev: GenEvent) => {
      queue.push(ev)
      if (ev.type === 'done' || ev.type === 'error') {
        finished = true
        es.close()
      }
      wake?.()
    }
    es.onmessage = (m) => push(JSON.parse(m.data) as GenEvent)
    es.onerror = () => {
      if (!finished) push({ type: 'error', message: 'Lost connection to the model server.' })
    }
    signal.addEventListener('abort', () => {
      finished = true
      es.close()
      wake?.()
    })
    try {
      while (true) {
        if (queue.length) {
          const ev = queue.shift()!
          yield ev
          if (ev.type === 'done' || ev.type === 'error') return
        } else if (finished) {
          return
        } else {
          await new Promise<void>((r) => (wake = r))
          wake = null
        }
      }
    } finally {
      es.close()
    }
  }
}
