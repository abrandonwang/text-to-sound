// In-browser fake backend. Emits the same events, with realistic timings,
// that the real model server will stream over SSE. Lets the site run on
// GitHub Pages with no server at all until the model is ready.

import type { GenEvent, GenRequest } from './events'
import {
  BINS,
  FRAMES,
  SR,
  bytesToBase64,
  ddimTimesteps,
  encodeWav,
  fakeEmbedding,
  latentNoise,
  melToAudio,
  melToBase64,
  noiseLevel,
  noisyEstimate,
  targetMel,
} from '../mock/synth'

const PREVIEW_EVERY = 5

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(id)
      reject(new DOMException('aborted', 'AbortError'))
    })
  })
}
const jitter = (ms: number) => ms * (0.8 + Math.random() * 0.4)

export async function* mockGenerate(req: GenRequest, signal: AbortSignal): AsyncGenerator<GenEvent> {
  const t0 = performance.now()
  const img = (m: Float32Array) => ({ bins: BINS, frames: FRAMES, data: melToBase64(m) })

  yield { type: 'queued', position: 0 }
  await sleep(jitter(250), signal)

  // 1. CLAP text encoder
  yield { type: 'stage_start', stage: 'encode' }
  const tEnc = performance.now()
  await sleep(jitter(140), signal)
  const target = targetMel(req.prompt, req.seed)
  yield { type: 'embedding', values: fakeEmbedding(req.prompt) }
  yield { type: 'stage_end', stage: 'encode', ms: Math.round(performance.now() - tEnc), info: '512-d embedding' }

  // 2. CFG + DDIM denoising
  yield { type: 'stage_start', stage: 'denoise' }
  const tDen = performance.now()
  const noise = latentNoise(req.seed)
  const ts = ddimTimesteps(req.steps)
  for (let i = 0; i < ts.length; i++) {
    const s0 = performance.now()
    await sleep(jitter(55), signal) // ~ one U-Net forward pass x2 (cond + uncond)
    const step = i + 1
    const sendPreview = step % PREVIEW_EVERY === 0 || step === ts.length || step === 1
    yield {
      type: 'denoise_step',
      step,
      total: ts.length,
      timestep: ts[i],
      noise: noiseLevel(ts[i]),
      ms: Math.round(performance.now() - s0),
      preview: sendPreview ? img(noisyEstimate(target, noise, ts[i])) : undefined,
    }
  }
  yield {
    type: 'stage_end',
    stage: 'denoise',
    ms: Math.round(performance.now() - tDen),
    info: `${req.steps} DDIM steps · cfg ${req.guidance.toFixed(1)}`,
  }

  // 3. VAE decoder
  yield { type: 'stage_start', stage: 'decode' }
  const tDec = performance.now()
  await sleep(jitter(90), signal)
  yield { type: 'stage_end', stage: 'decode', ms: Math.round(performance.now() - tDec), info: '80 × 256 log-mel' }

  // 4. Vocoder (Griffin-Lim stands in for BigVGAN)
  yield { type: 'stage_start', stage: 'vocode' }
  const tVoc = performance.now()
  await sleep(30, signal) // let the UI paint the active state before the heavy work
  const audio = melToAudio(target, req.seed)
  await sleep(jitter(120), signal)
  const seconds = audio.length / SR
  yield { type: 'stage_end', stage: 'vocode', ms: Math.round(performance.now() - tVoc), info: `${seconds.toFixed(2)} s @ 22.05 kHz` }

  yield { type: 'result', mel: img(target), sampleRate: SR, seconds, wav: bytesToBase64(encodeWav(audio)) }
  yield { type: 'done', totalMs: Math.round(performance.now() - t0) }
}
