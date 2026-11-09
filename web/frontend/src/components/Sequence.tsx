import { useEffect, useRef, useState } from 'react'
import type { RunState } from '../state'

// Full-screen loading screen, driven only by what the server reports. One big
// label says which of the four sounds is being made (flying up and away when
// the next one starts), one thin bar fills across the whole run, and a small
// line under it says what the model is doing right now.

const WAKING = 'waking up the model'
const LINE = 'waiting in line'

const START_MS = 220 // let the input finish leaving first
const COLD_MS = 1200 // no reply (or no start) by now means it's asleep (or busy)
const HOLD_MS = 900 // a label stays at least this long, so it can be read
const RESOLVE_MS = 260 // the bar reaches the end before the label leaves

// Share of one take spent removing noise, measured on the live server (2.6 s of 4.1 s).
const DENOISE_SHARE = 0.64

function labelOf(s: RunState, waited: number, inLine: number) {
  if (s.ahead > 0 && inLine >= COLD_MS) return LINE
  if (s.phase === 'wait' && s.settled === 0) return waited >= COLD_MS && s.ahead === 0 ? WAKING : null
  return `sound ${Math.min(s.settled + 1, s.count)} of ${s.count}`
}

function detailOf(s: RunState, label?: string) {
  if (s.ahead > 0 && label === LINE) return `${s.ahead} ahead of you`
  switch (s.phase) {
    case 'encode':
      return 'reading your prompt'
    case 'denoise':
      return `removing noise  ${s.step} / ${s.total}`
    case 'decode':
    case 'vocode':
      return 'turning it into audio'
    default:
      return label === WAKING ? 'takes about 20 seconds' : ''
  }
}

export function Sequence(props: { s: RunState; active: boolean; onDone: () => void }) {
  const { s, active, onDone } = props
  const [labels, setLabels] = useState<{ id: number; text: string }[]>([]) // the last one is on screen
  const [finished, setFinished] = useState(false)
  const fill = useRef<HTMLElement>(null)
  const latest = useRef(s)
  latest.current = s
  const cb = useRef(onDone)
  cb.current = onDone
  const detail = useRef('')

  useEffect(() => {
    if (!active) {
      setLabels([])
      setFinished(false)
      detail.current = ''
      return
    }
    const startedAt = performance.now()
    let current: string | null = null
    let shownAt = 0
    let id = 0
    let ending = 0
    let lineSince = 0
    const timer = setInterval(() => {
      const s = latest.current
      const now = performance.now()
      lineSince = s.ahead > 0 ? lineSince || now : 0
      if (ending || now - startedAt < START_MS) return
      if (current && now - shownAt < HOLD_MS) return
      if (current && s.settled >= s.count) {
        ending = window.setTimeout(() => {
          setFinished(true)
          cb.current()
        }, RESOLVE_MS)
        return
      }
      const want = labelOf(s, now - startedAt, lineSince ? now - lineSince : 0)
      if (want && want !== current) {
        current = want
        shownAt = now
        const next = { id: id++, text: want }
        setLabels((l) => [...l.slice(-1), next])
      }
    }, 40)

    // the bar: real steps while removing noise, an estimate from the last take
    // while making audio, and it never moves backwards
    let raf = 0
    let shown = 0
    let last = performance.now()
    let phase = latest.current.phase
    let phaseAt = last
    const tick = (now: number) => {
      const dt = now - last
      last = now
      const s = latest.current
      if (s.phase !== phase) {
        phase = s.phase
        phaseAt = now
      }
      let take = 0
      if (s.phase === 'denoise') take = DENOISE_SHARE * (s.step / Math.max(1, s.total))
      else if (s.phase === 'decode') take = DENOISE_SHARE
      else if (s.phase === 'vocode') take = DENOISE_SHARE + (1 - DENOISE_SHARE) * Math.min(0.95, (now - phaseAt) / s.vocodeMs)
      const target = Math.min(1, (s.settled + take) / s.count)
      shown = Math.max(shown, shown + (target - shown) * (1 - Math.exp(-dt / 130))) // same speed at any frame rate
      if (fill.current) fill.current.style.transform = `scaleX(${shown})`
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      clearInterval(timer)
      clearTimeout(ending)
      cancelAnimationFrame(raf)
      if (fill.current) fill.current.style.transform = 'scaleX(0)'
    }
  }, [active])

  // between takes nothing is happening for a moment: keep the last line up
  const line = detailOf(s, labels[labels.length - 1]?.text)
  if (line) detail.current = line

  return (
    <div className={`sequence ${active ? 'on' : ''} ${finished ? 'is-finished' : ''}`} aria-live="polite">
      <p className="readout" aria-hidden>
        {s.prompt}
      </p>
      {labels.map((l, i) => (
        <div key={l.id} className={`stage ${finished || i < labels.length - 1 ? 'is-gone' : 'is-current'}`} data-quiet="">
          <span className="stage-label">{l.text}</span>
        </div>
      ))}
      <div className="bar" aria-hidden>
        <i ref={fill} />
      </div>
      <p className="detail" aria-hidden>
        {detail.current}
      </p>
    </div>
  )
}
