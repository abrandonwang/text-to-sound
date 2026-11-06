// The wire format between backend and frontend.
// Every SSE message is one JSON object with a `type` field.
// The mock (in-browser and Python) and the real model server must all emit exactly this.

export type StageId = 'encode' | 'denoise' | 'decode' | 'vocode'

/** A log-mel image: `bins` rows x `frames` cols, uint8 (0 = quiet, 255 = loud), row 0 = lowest freq, base64. */
export interface MelImage {
  bins: number
  frames: number
  data: string
}

export interface GenRequest {
  prompt: string
  guidance: number // classifier-free guidance scale
  steps: number // DDIM steps
  seed: number
}

export type GenEvent =
  | { type: 'queued'; position: number }
  | { type: 'stage_start'; stage: StageId }
  | { type: 'stage_end'; stage: StageId; ms: number; info?: string }
  /** CLAP output, downsampled for display (e.g. 64 values in [-1, 1]). */
  | { type: 'embedding'; values: number[] }
  | {
      type: 'denoise_step'
      step: number // 1-based
      total: number
      timestep: number // DDPM timestep t this step lands on (999 -> 0)
      noise: number // sqrt(1 - alpha_bar_t): 1 = pure noise, 0 = clean
      ms: number // wall time of this step
      preview?: MelImage // decoded estimate, sent every few steps
    }
  | { type: 'result'; mel: MelImage; sampleRate: number; seconds: number; wav: string /* base64 WAV */ }
  | { type: 'done'; totalMs: number }
  | { type: 'error'; message: string }

export type Generator = (req: GenRequest, signal: AbortSignal) => AsyncGenerator<GenEvent>
