import type { GenEvent, MelImage, StageId } from './api'

export interface StageState {
  status: 'idle' | 'active' | 'done'
  ms?: number
  info?: string
}

export interface Result {
  mel: MelImage
  url: string // blob: URL of the WAV
  seconds: number
  prompt: string
}

export interface RunState {
  phase: 'idle' | 'queued' | 'running' | 'done' | 'error'
  prompt: string
  stages: Record<StageId, StageState>
  embedding?: number[]
  step: number
  total: number
  timestep?: number
  noise: number[] // noise level per step so far
  stepMs: number[]
  preview?: MelImage
  previewStep?: number
  result?: Result
  totalMs?: number
  error?: string
}

const idleStages = (): Record<StageId, StageState> => ({
  encode: { status: 'idle' },
  denoise: { status: 'idle' },
  decode: { status: 'idle' },
  vocode: { status: 'idle' },
})

export const initialState: RunState = { phase: 'idle', prompt: '', stages: idleStages(), step: 0, total: 0, noise: [], stepMs: [] }

export type Action =
  | { type: 'start'; prompt: string; steps: number }
  | { type: 'event'; ev: GenEvent }

function wavUrl(b64: string) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
}

export function reducer(s: RunState, a: Action): RunState {
  if (a.type === 'start')
    return { ...initialState, phase: 'queued', prompt: a.prompt, stages: idleStages(), total: a.steps, result: undefined }

  const ev = a.ev
  switch (ev.type) {
    case 'queued':
      return { ...s, phase: 'queued' }
    case 'stage_start':
      return { ...s, phase: 'running', stages: { ...s.stages, [ev.stage]: { status: 'active' } } }
    case 'stage_end':
      return { ...s, stages: { ...s.stages, [ev.stage]: { status: 'done', ms: ev.ms, info: ev.info } } }
    case 'embedding':
      return { ...s, embedding: ev.values }
    case 'denoise_step':
      return {
        ...s,
        step: ev.step,
        total: ev.total,
        timestep: ev.timestep,
        noise: [...s.noise, ev.noise],
        stepMs: [...s.stepMs, ev.ms],
        preview: ev.preview ?? s.preview,
        previewStep: ev.preview ? ev.step : s.previewStep,
      }
    case 'result':
      return { ...s, result: { mel: ev.mel, url: wavUrl(ev.wav), seconds: ev.seconds, prompt: s.prompt } }
    case 'done':
      return { ...s, phase: 'done', totalMs: ev.totalMs }
    case 'error':
      return { ...s, phase: 'error', error: ev.message }
  }
}

export { wavUrl }
