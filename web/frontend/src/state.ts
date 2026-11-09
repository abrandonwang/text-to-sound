import type { GenEvent, StageId } from './api'

export const TAKES = 4 // how many versions of the sound each run makes

export interface Take {
  url: string // blob: URL of the WAV
  seconds: number
}

// One run is several takes generated one after another. A stage only counts as
// done once every take has been through it.
export interface RunState {
  prompt: string
  count: number
  stages: Record<StageId, 'idle' | 'active' | 'done'>
  ended: Record<StageId, number> // how many takes have finished each stage
  steps: number[] // denoising steps finished, per take
  total: number // denoising steps per take
  takes: Take[]
  failed: number
  settled: number // takes that have finished, one way or the other
  error?: string
}

const idleStages = (): RunState['stages'] => ({ encode: 'idle', denoise: 'idle', decode: 'idle', vocode: 'idle' })
const noneEnded = (): RunState['ended'] => ({ encode: 0, denoise: 0, decode: 0, vocode: 0 })

export const initialState: RunState = {
  prompt: '',
  count: TAKES,
  stages: idleStages(),
  ended: noneEnded(),
  steps: [],
  total: 0,
  takes: [],
  failed: 0,
  settled: 0,
}

export type Action =
  | { type: 'start'; prompt: string; steps: number; count: number }
  | { type: 'event'; ev: GenEvent; take: number; url?: string }

export function reducer(s: RunState, a: Action): RunState {
  if (a.type === 'start')
    return { ...initialState, stages: idleStages(), ended: noneEnded(), prompt: a.prompt, total: a.steps, count: a.count, steps: Array(a.count).fill(0) }

  const { ev, take, url } = a
  switch (ev.type) {
    case 'stage_start':
      return s.stages[ev.stage] === 'done' ? s : { ...s, stages: { ...s.stages, [ev.stage]: 'active' } }
    case 'stage_end': {
      const n = s.ended[ev.stage] + 1
      return {
        ...s,
        ended: { ...s.ended, [ev.stage]: n },
        stages: { ...s.stages, [ev.stage]: n + s.failed >= s.count ? 'done' : 'active' },
      }
    }
    case 'denoise_step':
      return { ...s, total: ev.total, steps: s.steps.map((v, i) => (i === take ? ev.step : v)) }
    case 'result':
      return url ? { ...s, takes: [...s.takes, { url, seconds: ev.seconds }] } : s
    case 'done':
      return { ...s, settled: s.settled + 1 }
    case 'error': {
      // a failed take stops counting towards every stage it never finished
      const failed = s.failed + 1
      const stages = { ...s.stages }
      for (const k of Object.keys(stages) as StageId[]) if (s.ended[k] + failed >= s.count) stages[k] = 'done'
      return { ...s, failed, stages, settled: s.settled + 1, error: ev.message }
    }
    default:
      return s
  }
}

export function wavUrl(b64: string) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
}
