import { useEffect, useRef } from 'react'

// Full-screen "LED panel" background. Every cell is one RGB pixel.
// Idle: a travelling neon sound wave. During generation it becomes the
// spectrogram itself (mirrored around the centre line, low frequencies in the
// middle), starting as static and resolving as the model removes noise.

export interface Scene {
  mel: Uint8Array | null // 80 x 256, bin-major, 0..255 (latest preview or final result)
  noise: number // 0..1, current diffusion noise level
  playhead: number | null // 0..1 while audio plays
  level: number // audio loudness 0..1 while playing
  pulse: number // set to 1 on stage changes, decays
}

export const newScene = (): Scene => ({ mel: null, noise: 0, playhead: null, level: 0, pulse: 0 })

const BINS = 80
const FRAMES = 256

// Saturated rainbow lookup: hue (0..359) -> rgb
const HUE = (() => {
  const lut = new Float32Array(360 * 3)
  for (let h = 0; h < 360; h++) {
    const k = (n: number) => (n + h / 60) % 6
    const f = (n: number) => 1 - Math.max(0, Math.min(k(n), 4 - k(n), 1))
    lut[h * 3] = f(5) * 255
    lut[h * 3 + 1] = f(3) * 255
    lut[h * 3 + 2] = f(1) * 255
  }
  return lut
})()

export function LedField({ scene }: { scene: React.MutableRefObject<Scene> }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const small = document.createElement('canvas')
    const sctx = small.getContext('2d')!
    const tiny = document.createElement('canvas')
    const tctx = tiny.getContext('2d')!
    const mask = document.createElement('canvas')
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches

    let cols = 0
    let rows = 0
    let cell = 0
    let dpr = 1
    let img: ImageData
    const shown = new Float32Array(BINS * FRAMES) // mel eased toward the latest one
    const energy = new Float32Array(FRAMES)
    let mix = 0 // 0 = idle wave, 1 = spectrogram
    let noise = 0
    let dim = new Float32Array(0) // per-cell dimming behind the input and the step text

    // Dim LEDs under the UI so text stays readable. Uses the real element boxes.
    const buildDim = () => {
      dim = new Float32Array(cols * rows).fill(1)
      const zones: [string, number][] = [
        ['.prompt', 0.82],
        ['.side.on', 0.5],
      ]
      for (const [sel, amount] of zones) {
        const el = document.querySelector(sel)
        if (!el) continue
        const r = el.getBoundingClientRect()
        const pad = 150
        for (let y = 0; y < rows; y++)
          for (let x = 0; x < cols; x++) {
            const px = (x + 0.5) * cell
            const py = (y + 0.5) * cell
            const dx = Math.max(r.left - px, 0, px - r.right) / pad
            const dy = Math.max(r.top - py, 0, py - r.bottom) / pad
            const k = Math.exp(-(dx * dx + dy * dy) * 3)
            dim[y * cols + x] = Math.min(dim[y * cols + x], 1 - amount * k)
          }
      }
    }

    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1)
      const W = window.innerWidth
      const H = window.innerHeight
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
      cell = W < 700 ? 8 : 11
      cols = Math.ceil(W / cell)
      rows = Math.ceil(H / cell)
      small.width = cols
      small.height = rows
      tiny.width = Math.ceil(cols / 4)
      tiny.height = Math.ceil(rows / 4)
      img = sctx.createImageData(cols, rows)
      // LED mask: black sheet with a rounded hole per cell
      mask.width = canvas.width
      mask.height = canvas.height
      const m = mask.getContext('2d')!
      m.fillStyle = '#000'
      m.fillRect(0, 0, mask.width, mask.height)
      m.globalCompositeOperation = 'destination-out'
      const c = cell * dpr
      const dot = c * 0.74
      const off = (c - dot) / 2
      for (let y = 0; y < rows; y++)
        for (let x = 0; x < cols; x++) {
          m.beginPath()
          m.roundRect(x * c + off, y * c + off, dot, dot, dot * 0.28)
          m.fill()
        }
    }
    const onResize = () => {
      resize()
      buildDim()
    }
    onResize()
    const dimTimer = setInterval(buildDim, 1000) // cheap; keeps up with layout changes
    window.addEventListener('resize', onResize)

    let raf = 0
    const t0 = performance.now()
    const frame = (now: number) => {
      const s = scene.current
      const t = ((now - t0) / 1000) * (reduced ? 0.25 : 1)

      // ease state
      if (s.mel) for (let i = 0; i < shown.length; i++) shown[i] += (s.mel[i] / 255 - shown[i]) * 0.12
      mix += ((s.mel ? 1 : 0) - mix) * 0.04
      noise += (s.noise - noise) * 0.15
      s.pulse *= 0.93

      let emax = 0.2
      for (let f = 0; f < FRAMES; f++) {
        let e = 0
        for (let b = 0; b < BINS; b++) e += shown[b * FRAMES + f]
        energy[f] = e / BINS
        emax = Math.max(emax, energy[f])
      }

      const data = img.data
      const mid = (rows - 1) / 2
      const hMaxM = rows * 0.36
      const lvl = 0.9 + s.level * 0.9
      for (let x = 0; x < cols; x++) {
        const u = x / (cols - 1)
        // idle travelling wave
        const w = Math.sin(u * 9 + t * 1.1) * Math.cos(u * 3.3 - t * 0.6) + 0.35 * Math.sin(u * 23 - t * 2.3)
        // per-column flicker makes it read as a waveform / equaliser, not a blob
        const spike = 0.5 + 0.5 * Math.sin(x * 1.7 + t * 4.1) * Math.sin(x * 0.91 - t * 2.7)
        const hI = rows * (0.04 + 0.22 * Math.abs(w) * (0.45 + 0.55 * spike)) * (1 + s.pulse * 0.5)
        // spectrogram envelope
        const f = Math.min(FRAMES - 1, Math.floor(u * FRAMES))
        const hM = hMaxM * (0.15 + 0.85 * (energy[f] / emax))
        const h = hI + (hM - hI) * mix
        const hueBase = u * 290 + t * 28
        const play = s.playhead === null ? 0 : Math.max(0, 1 - Math.abs(u - s.playhead) * cols * 0.5)

        for (let y = 0; y < rows; y++) {
          const d = Math.abs(y - mid)
          let b = 0
          if (d < h) {
            const bI = (1 - d / h) ** 0.6
            const bin = Math.min(BINS - 1, Math.floor((d / hMaxM) * BINS))
            const bM = Math.min(1, shown[bin * FRAMES + f] * 1.35)
            b = bI + (bM - bI) * mix
            b *= 0.82 + 0.18 * Math.sin(t * 5 + x * 0.7 + y * 1.3)
            b += play * 0.9
          }
          if (noise > 0.01) {
            // static stays inside the wave's band and fades out toward its edges
            const reach = Math.max(0, 1 - d / (hMaxM * 1.1)) ** 0.7
            b = b * (1 - noise * 0.6) + Math.random() ** 2 * noise * reach * 1.1
          }
          b = Math.min(1.4, b * lvl) * (dim[y * cols + x] ?? 1)

          const hue = (Math.floor(hueBase + d * 3.5) % 360) * 3
          const white = Math.max(0, b - 0.9) * 2 + play * 0.6
          const o = (y * cols + x) * 4
          data[o] = 9 + HUE[hue] * b + 255 * white
          data[o + 1] = 9 + HUE[hue + 1] * b + 255 * white
          data[o + 2] = 13 + HUE[hue + 2] * b + 255 * white
          data[o + 3] = 255
        }
      }
      sctx.putImageData(img, 0, 0)

      const W = cols * cell * dpr
      const H = rows * cell * dpr
      ctx.globalCompositeOperation = 'source-over'
      ctx.globalAlpha = 1
      ctx.fillStyle = '#020203'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      // crisp LEDs
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(small, 0, 0, W, H)
      ctx.drawImage(mask, 0, 0)
      // bloom: blurry upscales added on top so light spills over the gaps
      ctx.globalCompositeOperation = 'lighter'
      ctx.imageSmoothingEnabled = true
      ctx.globalAlpha = 0.16
      ctx.drawImage(small, 0, 0, W, H)
      tctx.imageSmoothingEnabled = true
      tctx.clearRect(0, 0, tiny.width, tiny.height)
      tctx.drawImage(small, 0, 0, tiny.width, tiny.height)
      ctx.globalAlpha = 0.38
      ctx.drawImage(tiny, 0, 0, W, H)

      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      clearInterval(dimTimer)
      window.removeEventListener('resize', onResize)
    }
  }, [scene])

  return <canvas ref={ref} className="led" aria-hidden />
}
