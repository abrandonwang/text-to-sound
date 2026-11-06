// Fake "model": turns a prompt into a plausible-looking log-mel spectrogram,
// fakes the noisy intermediate estimates, and turns the mel into audio
// (Griffin-Lim, standing in for BigVGAN). Deterministic per prompt + seed.

export const BINS = 80
export const FRAMES = 256
export const SR = 22050
const HOP = 256
const NFFT = 1024
const FMAX = 8000

// ---------- seeded RNG ----------
export function hashString(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

export function rng(seed: number) {
  let a = seed >>> 0
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const gauss = () => {
    const u = Math.max(next(), 1e-9)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next())
  }
  return { next, gauss }
}

// ---------- mel scale ----------
const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700)
const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1)
const MEL_MAX = hzToMel(FMAX)
/** Fractional mel bin for a frequency in Hz. */
export const hzToBin = (f: number) => (hzToMel(f) / MEL_MAX) * (BINS - 1)
export const binToHz = (b: number) => melToHz((b / (BINS - 1)) * MEL_MAX)

// ---------- target spectrogram from prompt ----------
type Layer = (m: Float32Array, r: ReturnType<typeof rng>) => void

const at = (_m: Float32Array, b: number, f: number) => b * FRAMES + f
function add(m: Float32Array, b: number, f: number, v: number) {
  if (b < 0 || b >= BINS || f < 0 || f >= FRAMES) return
  m[at(m, b, f)] = Math.max(m[at(m, b, f)], v)
}
const smoothEnv = (r: ReturnType<typeof rng>, speed: number) => {
  // slow random envelope in [0,1]
  const pts = Array.from({ length: 12 }, () => r.next())
  return (f: number) => {
    const x = (f / FRAMES) * (pts.length - 1) * speed
    const i = Math.floor(x) % (pts.length - 1)
    const t = x - Math.floor(x)
    return pts[i] * (1 - t) + pts[i + 1] * t
  }
}

/** Broadband noise between two frequencies (rain, wind, rumble, crowd hiss). */
const noiseBed =
  (lo: number, hi: number, level: number, speed = 1, tilt = 0): Layer =>
  (m, r) => {
    const env = smoothEnv(r, speed)
    const b0 = hzToBin(lo)
    const b1 = hzToBin(hi)
    for (let f = 0; f < FRAMES; f++) {
      const e = 0.55 + 0.45 * env(f)
      for (let b = Math.floor(b0); b <= Math.ceil(b1); b++) {
        const pos = (b - b0) / Math.max(1, b1 - b0)
        const edge = Math.min(1, Math.min(b - b0 + 1, b1 - b + 1) / 4)
        add(m, b, f, level * e * edge * (1 - tilt * pos) * (0.82 + 0.18 * r.next()))
      }
    }
  }

/** Short broadband hits (knocks, footsteps, drops, barks). */
const transients =
  (count: number, lo: number, hi: number, level: number, decay: number, jitter = 1): Layer =>
  (m, r) => {
    const b0 = hzToBin(lo)
    const b1 = hzToBin(hi)
    for (let i = 0; i < count; i++) {
      const f0 = Math.floor(((i + 0.5 + (r.next() - 0.5) * jitter) / count) * FRAMES)
      const amp = level * (0.6 + 0.4 * r.next())
      for (let d = 0; d < decay * 4; d++) {
        const v = amp * Math.exp(-d / decay)
        for (let b = Math.floor(b0); b <= Math.ceil(b1); b++) add(m, b, f0 + d, v * (0.85 + 0.15 * r.next()))
      }
    }
  }

/** Harmonic tone with a pitch track (birds, sirens, bells, voices). */
const tones =
  (opts: {
    count: number
    f0: [number, number]
    harmonics: number
    len: [number, number]
    sweep: number // octaves of pitch movement per note
    level: number
    vibrato?: number
  }): Layer =>
  (m, r) => {
    for (let n = 0; n < opts.count; n++) {
      const len = Math.floor(opts.len[0] + r.next() * (opts.len[1] - opts.len[0]))
      const start = Math.floor(r.next() * Math.max(1, FRAMES - len))
      const f0 = opts.f0[0] * (opts.f0[1] / opts.f0[0]) ** r.next()
      const dir = r.next() < 0.5 ? -1 : 1
      for (let i = 0; i < len; i++) {
        const t = i / len
        const env = Math.sin(Math.PI * Math.min(1, t * 1.2)) ** 0.6
        const vib = opts.vibrato ? Math.sin(i * 0.5) * opts.vibrato : 0
        const hz = f0 * 2 ** (dir * opts.sweep * t + vib)
        for (let h = 1; h <= opts.harmonics; h++) {
          const b = hzToBin(hz * h)
          if (b >= BINS) break
          const v = (opts.level * env) / h ** 0.7
          const bi = Math.round(b)
          add(m, bi, start + i, v)
          add(m, bi - 1, start + i, v * 0.45)
          add(m, bi + 1, start + i, v * 0.45)
        }
      }
    }
  }

const RECIPES: [RegExp, Layer[]][] = [
  [/rain|drizzle|shower|patter/, [noiseBed(600, 8000, 0.72, 2, 0.25), transients(40, 2000, 8000, 0.85, 1.2, 3)]],
  [/water|stream|river|brook|pour|splash|wave|ocean|sea/, [noiseBed(150, 6000, 0.78, 1.5, 0.35)]],
  [/thunder|rumble|explosion|boom/, [transients(2, 30, 900, 0.95, 18, 0.6), noiseBed(30, 400, 0.6, 0.6)]],
  [/wind|breeze|storm|air/, [noiseBed(80, 2500, 0.62, 0.7, 0.5)]],
  [/engine|car|truck|motor|traffic|train|plane|bus/, [noiseBed(40, 1200, 0.72, 0.5, 0.6), tones({ count: 2, f0: [60, 120], harmonics: 14, len: [160, 250], sweep: 0.3, level: 0.75 })]],
  [/bird|chirp|tweet|song|crow/, [tones({ count: 14, f0: [2200, 4500], harmonics: 2, len: [5, 18], sweep: 0.7, level: 0.95 })]],
  [/siren|alarm|beep|whistle|horn/, [tones({ count: 3, f0: [600, 1100], harmonics: 4, len: [70, 110], sweep: 0.5, level: 0.9 })]],
  [/bell|chime|ding|gong/, [tones({ count: 4, f0: [400, 900], harmonics: 6, len: [90, 160], sweep: 0, level: 0.92 })]],
  [/dog|bark/, [tones({ count: 6, f0: [350, 600], harmonics: 10, len: [8, 14], sweep: -0.4, level: 0.9 }), transients(6, 200, 3000, 0.5, 2)]],
  [/speak|speech|talk|voice|people|crowd|man|woman|child|laugh|chatter/, [tones({ count: 18, f0: [110, 240], harmonics: 18, len: [8, 24], sweep: 0.15, level: 0.8, vibrato: 0.02 }), noiseBed(200, 4000, 0.3, 3)]],
  [/foot|step|walk|knock|door|clap|tap|hammer|typing|keyboard|click/, [transients(14, 150, 6000, 0.88, 2.5, 0.8)]],
  [/music|guitar|piano|strum|melody/, [tones({ count: 10, f0: [180, 520], harmonics: 9, len: [20, 50], sweep: 0, level: 0.85 })]],
  [/fire|crackle|fry|sizzle/, [noiseBed(300, 7000, 0.45, 3, 0.2), transients(45, 800, 7500, 0.75, 1, 4)]],
]

/** Target mel in [0, 1], shape [BINS * FRAMES], row-major (bin-major). */
export function targetMel(prompt: string, seed: number): Float32Array {
  const m = new Float32Array(BINS * FRAMES)
  const r = rng(hashString(prompt.toLowerCase()) ^ seed)
  const p = prompt.toLowerCase()
  let layers = RECIPES.filter(([re]) => re.test(p)).flatMap(([, ls]) => ls)
  if (layers.length === 0) {
    // unknown prompt: pick two recipes pseudo-randomly
    const a = RECIPES[Math.floor(r.next() * RECIPES.length)][1]
    const b = RECIPES[Math.floor(r.next() * RECIPES.length)][1]
    layers = [...a, ...b]
  }
  // quiet room-tone floor, slightly louder at low frequencies
  for (let b = 0; b < BINS; b++)
    for (let f = 0; f < FRAMES; f++) m[at(m, b, f)] = 0.12 * (1 - b / BINS) + 0.04 * r.next()
  for (const layer of layers) layer(m, r)
  // 3x3 blur to look like a real (smeared) spectrogram
  const out = new Float32Array(m.length)
  for (let b = 0; b < BINS; b++)
    for (let f = 0; f < FRAMES; f++) {
      let s = 0
      let n = 0
      for (let db = -1; db <= 1; db++)
        for (let df = -1; df <= 1; df++) {
          const bb = b + db
          const ff = f + df
          if (bb < 0 || bb >= BINS || ff < 0 || ff >= FRAMES) continue
          const w = db === 0 && df === 0 ? 4 : 1
          s += m[at(m, bb, ff)] * w
          n += w
        }
      out[at(out, b, f)] = Math.min(1, s / n)
    }
  return out
}

// ---------- diffusion look-alike ----------
const T = 1000
const ALPHA_BAR = (() => {
  // linear beta schedule 1e-4 .. 0.02 (same as the DDPM in the notebooks)
  const ab = new Float64Array(T)
  let prod = 1
  for (let t = 0; t < T; t++) {
    prod *= 1 - (1e-4 + (0.02 - 1e-4) * (t / (T - 1)))
    ab[t] = prod
  }
  return ab
})()

/** DDIM timesteps, descending, e.g. 980, 960, ... 0 for 50 steps. */
export function ddimTimesteps(steps: number): number[] {
  const stride = T / steps
  return Array.from({ length: steps }, (_, i) => Math.round((steps - 1 - i) * stride))
}

export const noiseLevel = (t: number) => Math.sqrt(1 - ALPHA_BAR[t])

/** Noise at latent resolution (20 x 64), upsampled 4x like the VAE decoder would. */
export function latentNoise(seed: number): Float32Array {
  const r = rng(seed * 7919 + 13)
  const LB = 20
  const LF = 64
  const z = Float32Array.from({ length: LB * LF }, () => r.gauss())
  const out = new Float32Array(BINS * FRAMES)
  for (let b = 0; b < BINS; b++)
    for (let f = 0; f < FRAMES; f++) {
      const y = Math.min(LB - 1.001, b / 4)
      const x = Math.min(LF - 1.001, f / 4)
      const y0 = Math.floor(y)
      const x0 = Math.floor(x)
      const ty = y - y0
      const tx = x - x0
      const v =
        z[y0 * LF + x0] * (1 - ty) * (1 - tx) +
        z[y0 * LF + x0 + 1] * (1 - ty) * tx +
        z[(y0 + 1) * LF + x0] * ty * (1 - tx) +
        z[(y0 + 1) * LF + x0 + 1] * ty * tx
      out[b * FRAMES + f] = v * 0.8 + r.gauss() * 0.35
    }
  return out
}

/** x_t = sqrt(ab) * x0 + sqrt(1 - ab) * eps, mapped back into [0, 1] for display. */
export function noisyEstimate(target: Float32Array, noise: Float32Array, t: number): Float32Array {
  const ab = ALPHA_BAR[t]
  const a = Math.sqrt(ab)
  const s = Math.sqrt(1 - ab)
  const out = new Float32Array(target.length)
  for (let i = 0; i < out.length; i++) {
    const x0 = target[i] * 2 - 1
    out[i] = Math.min(1, Math.max(0, ((a * x0 + s * noise[i] * 0.7) + 1) / 2))
  }
  return out
}

// ---------- encoding helpers ----------
export function melToBase64(m: Float32Array): string {
  const bytes = new Uint8Array(m.length)
  for (let i = 0; i < m.length; i++) bytes[i] = Math.round(Math.min(1, Math.max(0, m[i])) * 255)
  return bytesToBase64(bytes)
}

export function bytesToBase64(bytes: Uint8Array): string {
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

/** Fake 512-d CLAP embedding, shown as 64 values. */
export function fakeEmbedding(prompt: string): number[] {
  const r = rng(hashString('clap:' + prompt.toLowerCase()))
  return Array.from({ length: 64 }, () => Math.tanh(r.gauss() * 0.8))
}

// ---------- vocoder stand-in (Griffin-Lim) ----------
function fft(re: Float64Array, im: Float64Array, inverse: boolean) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr
        re[i + k + len / 2] = re[i + k] - ar
        im[i + k + len / 2] = im[i + k] - ai
        re[i + k] += ar
        im[i + k] += ai
        const t = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = t
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) (re[i] /= n), (im[i] /= n)
}

const HANN = Float64Array.from({ length: NFFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT))

/** mel [0,1] -> waveform, using Griffin-Lim with `iters` iterations. */
export function melToAudio(mel: Float32Array, seed: number, iters = 16): Float32Array {
  const K = NFFT / 2 + 1
  // linear magnitude per frame
  const mag = new Float64Array(FRAMES * K)
  for (let k = 0; k < K; k++) {
    const hz = (k * SR) / NFFT
    const b = Math.min(BINS - 1, hzToBin(Math.min(hz, FMAX)))
    const b0 = Math.floor(b)
    const b1 = Math.min(BINS - 1, b0 + 1)
    const tb = b - b0
    const rolloff = hz > FMAX ? Math.exp(-(hz - FMAX) / 800) : 1
    for (let f = 0; f < FRAMES; f++) {
      const v = mel[b0 * FRAMES + f] * (1 - tb) + mel[b1 * FRAMES + f] * tb
      const db = -70 + 70 * v
      mag[f * K + k] = 10 ** (db / 20) * rolloff
    }
  }
  const len = (FRAMES - 1) * HOP + NFFT
  const r = rng(seed + 99)
  const phRe = new Float64Array(FRAMES * K)
  const phIm = new Float64Array(FRAMES * K)
  for (let i = 0; i < phRe.length; i++) {
    const a = r.next() * 2 * Math.PI
    phRe[i] = Math.cos(a)
    phIm[i] = Math.sin(a)
  }
  const re = new Float64Array(NFFT)
  const im = new Float64Array(NFFT)
  const winSum = new Float64Array(len)
  for (let f = 0; f < FRAMES; f++) for (let i = 0; i < NFFT; i++) winSum[f * HOP + i] += HANN[i] ** 2
  let y = new Float64Array(len)

  const istft = () => {
    y = new Float64Array(len)
    for (let f = 0; f < FRAMES; f++) {
      for (let k = 0; k < K; k++) {
        const mg = mag[f * K + k]
        re[k] = mg * phRe[f * K + k]
        im[k] = mg * phIm[f * K + k]
        if (k > 0 && k < K - 1) {
          re[NFFT - k] = re[k]
          im[NFFT - k] = -im[k]
        }
      }
      fft(re, im, true)
      for (let i = 0; i < NFFT; i++) y[f * HOP + i] += re[i] * HANN[i]
    }
    for (let i = 0; i < len; i++) y[i] /= Math.max(winSum[i], 1e-3)
  }
  const stftPhase = () => {
    for (let f = 0; f < FRAMES; f++) {
      for (let i = 0; i < NFFT; i++) {
        re[i] = y[f * HOP + i] * HANN[i]
        im[i] = 0
      }
      fft(re, im, false)
      for (let k = 0; k < K; k++) {
        const n = Math.hypot(re[k], im[k]) || 1
        phRe[f * K + k] = re[k] / n
        phIm[f * K + k] = im[k] / n
      }
    }
  }
  istft()
  for (let it = 0; it < iters; it++) {
    stftPhase()
    istft()
  }
  // trim padding, normalise, fade
  const out = new Float32Array(FRAMES * HOP)
  let peak = 1e-6
  for (let i = 0; i < out.length; i++) peak = Math.max(peak, Math.abs(y[i + NFFT / 2] ?? 0))
  const fade = 512
  for (let i = 0; i < out.length; i++) {
    const g = Math.min(1, i / fade, (out.length - i) / fade)
    out[i] = ((y[i + NFFT / 2] ?? 0) / peak) * 0.85 * g
  }
  return out
}

export function encodeWav(samples: Float32Array, sr = SR): Uint8Array {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const v = new DataView(buf)
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  w(0, 'RIFF')
  v.setUint32(4, 36 + samples.length * 2, true)
  w(8, 'WAVE')
  w(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true)
  v.setUint16(22, 1, true)
  v.setUint32(24, sr, true)
  v.setUint32(28, sr * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  w(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true)
  return new Uint8Array(buf)
}
