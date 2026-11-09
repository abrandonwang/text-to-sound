import type { GenEvent, StageId } from './api'

export const TAKES = 4 // how many versions of the sound each run makes

export interface Take {
  url: string // blob: URL of the WAV
  seconds: number
}

// One run is several takes generated one after another, so `settled` is also
// the index of the take being made right now. `phase` is what the server is
// doing for that take; 'wait' means it hasn't started on it yet.
export interface RunState {
  prompt: string
  count: number
  phase: 'wait' | StageId
  ahead: number // generations in front of this take, until the server starts on it
  step: number // denoising steps finished in this take
  total: number // denoising steps per take
  vocodeMs: number // how long the last take spent making audio
  takes: Take[]
  settled: number // takes that have finished, one way or the other
  error?: string
}

export const initialState: RunState = {
  prompt: '',
  count: TAKES,
  phase: 'wait',
  ahead: 0,
  step: 0,
  total: 0,
  vocodeMs: 1500,
  takes: [],
  settled: 0,
}

export type Action =
  | { type: 'start'; prompt: string; steps: number; count: number }
  | { type: 'event'; ev: GenEvent; url?: string }

export function reducer(s: RunState, a: Action): RunState {
  if (a.type === 'start') return { ...initialState, prompt: a.prompt, total: a.steps, count: a.count }

  const { ev, url } = a
  switch (ev.type) {
    case 'queued': // position counts others waiting, not the one being generated
      return { ...s, ahead: ev.position + 1 }
    case 'stage_start':
      return { ...s, phase: ev.stage, ahead: 0 }
    case 'stage_end':
      return ev.stage === 'vocode' ? { ...s, vocodeMs: ev.ms } : s
    case 'denoise_step':
      return { ...s, step: ev.step, total: ev.total }
    case 'result':
      return url ? { ...s, takes: [...s.takes, { url, seconds: ev.seconds }] } : s
    case 'done':
      return { ...s, phase: 'wait', step: 0, ahead: 0, settled: s.settled + 1 }
    case 'error':
      return { ...s, phase: 'wait', step: 0, ahead: 0, settled: s.settled + 1, error: ev.message }
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
