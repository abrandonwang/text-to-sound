import { useEffect, useRef } from 'react'

// Full-screen "LED panel" background. Every cell is one RGB pixel.
//
// Kept deliberately calm:
//   idle       rainbow backlight glowing out from behind the text box
//   otherwise  the glow pulls in and fades out; the panel stays dark and still
//
// Text stays readable without boxes: any element with [data-quiet] switches
// off the LEDs around it, with a dithered edge, following its live opacity.

export interface Scene {
  glow: boolean // backlight on (idle) or off (everything else)
  focus: boolean // input focused (backlight brightens)
}

export const newScene = (): Scene => ({ glow: true, focus: false })

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

const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x))

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
    let thresh = new Float32Array(0) // per-cell random threshold for the dithered switch-off
    let quiet = new Float32Array(0) // per-cell brightness multiplier from text zones
    let glow = 1 // 0..1, how much of the idle backlight is showing
    let focusAmt = 0

    const resize = () => {
      dpr = Math.min(window.innerWidth < 700 ? 1.5 : 2, window.devicePixelRatio || 1)
      const W = window.innerWidth
      const H = window.innerHeight
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
      cell = W < 700 ? 8 : 11
      cols = Math.max(1, Math.ceil(W / cell)) // never 0, or createImageData throws
      rows = Math.max(1, Math.ceil(H / cell))
      small.width = cols
      small.height = rows
      tiny.width = Math.ceil(cols / 4)
      tiny.height = Math.ceil(rows / 4)
      img = sctx.createImageData(cols, rows)
      thresh = Float32Array.from({ length: cols * rows }, () => Math.random())
      quiet = new Float32Array(cols * rows)
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
    resize()
    window.addEventListener('resize', resize)

    // Quiet zones: read every [data-quiet] element's live box and opacity.
    const buildQuiet = () => {
      quiet.fill(1)
      const els = document.querySelectorAll<HTMLElement>('[data-quiet]')
      const narrow = window.innerWidth < 700
      for (const el of els) {
        let op = 1
        for (let e: HTMLElement | null = el; e && e !== document.body; e = e.parentElement) op *= +getComputedStyle(e).opacity
        if (op < 0.02) continue
        op = Math.min(1, op * 3) // keep the zone dark while a label is fading out
        const r = el.getBoundingClientRect()
        if (r.width === 0) continue
        const F = el.dataset.quiet === 'small' ? 48 : narrow ? 80 : 140
        const fs = parseFloat(getComputedStyle(el).fontSize) || 16
        const L = r.left - fs * 0.4
        const R = r.right + fs * 0.4
        const T = r.top - fs * 0.25
        const B = r.bottom + fs * 0.25
        const x0 = Math.max(0, Math.floor((L - F) / cell))
        const x1 = Math.min(cols - 1, Math.ceil((R + F) / cell))
        const y0 = Math.max(0, Math.floor((T - F) / cell))
        const y1 = Math.min(rows - 1, Math.ceil((B + F) / cell))
        for (let y = y0; y <= y1; y++)
          for (let x = x0; x <= x1; x++) {
            const px = (x + 0.5) * cell
            const py = (y + 0.5) * cell
            const dx = Math.max(L - px, 0, px - R)
            const dy = Math.max(T - py, 0, py - B)
            const k = (1 - (0.04 + 0.96 * smooth(Math.hypot(dx, dy) / F))) * op // darkness 0..1
            const i = y * cols + x
            // dithered edge: cells switch off one by one instead of a soft gradient
            const v = k > thresh[i] * 0.9 + 0.05 ? 0.04 : 1 - k * 0.45
            if (v < quiet[i]) quiet[i] = v
          }
      }
    }

    let raf = 0
    const t0 = performance.now()
    const frame = (now: number) => {
      // a window with no size yet (e.g. opening in a hidden pane): wait for a resize
      if (!canvas.width || !canvas.height) {
        if (window.innerWidth && window.innerHeight) resize()
        raf = requestAnimationFrame(frame)
        return
      }
      const s = scene.current
      const t = ((now - t0) / 1000) * (reduced ? 0.25 : 1)

      glow += ((s.glow ? 1 : 0) - glow) * (s.glow ? 0.03 : 0.07)
      focusAmt += ((s.focus ? 1 : 0) - focusAmt) * 0.06

      buildQuiet()

      // idle backlight: light spreading out from behind the text box, rainbow
      // around it, slowly rotating. The text box's quiet zone blacks out the middle.
      const box = document.querySelector('.prompt')?.getBoundingClientRect()
      const gL = box ? box.left : 0
      const gR = box ? box.right : 0
      const gT = box ? box.top + box.height * 0.2 : 0
      const gB = box ? box.bottom - box.height * 0.2 : 0
      const gcx = (gL + gR) / 2
      const gcy = (gT + gB) / 2
      const reach = (150 + 70 * focusAmt) * (0.3 + 0.7 * glow) // pulls inward as it fades
      const breathe = reduced ? 0.85 : 0.74 + 0.26 * Math.sin(t * 0.9)
      const gGain = glow * breathe * (0.6 + 0.4 * focusAmt)

      const data = img.data
      for (let x = 0; x < cols; x++) {
        for (let y = 0; y < rows; y++) {
          let g = 0
          let gh = 0
          if (gGain > 0.005) {
            const px = (x + 0.5) * cell
            const py = (y + 0.5) * cell
            const dx = Math.max(gL - px, 0, px - gR)
            const dy = Math.max(gT - py, 0, py - gB)
            g = Math.exp(-Math.hypot(dx, dy * 1.15) / reach) * gGain
            if (!reduced) g *= 0.85 + 0.15 * Math.sin(t * 2.6 + thresh[y * cols + x] * 40)
            gh = (Math.floor((Math.atan2(py - gcy, (px - gcx) * 0.45) * 180) / Math.PI + 360 + t * 22) % 360) * 3
          }
          g *= quiet[y * cols + x]

          const o = (y * cols + x) * 4
          data[o] = 9 + HUE[gh] * g
          data[o + 1] = 9 + HUE[gh + 1] * g
          data[o + 2] = 13 + HUE[gh + 2] * g
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
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(small, 0, 0, W, H)
      ctx.drawImage(mask, 0, 0)
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
      window.removeEventListener('resize', resize)
    }
  }, [scene])

  return <canvas ref={ref} className="led" aria-hidden />
}
