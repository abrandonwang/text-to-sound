import { useEffect, useRef, useState } from 'react'
import type { StageId } from '../api'
import type { RunState } from '../state'

// Full-screen step sequence. One stage label at a time, centre stage, above one
// thin loading bar that fills across the whole run. When a stage is done its
// label flies up and disappears, and the next one rises in from below.
//
// The *visual* cursor is separate from the backend: it only moves on when the
// backend has finished that stage AND the label has been on screen long enough
// to read. So a 90 ms stage still gets its moment, and nothing is ever skipped.

const STAGES: { key: StageId; label: string }[] = [
  { key: 'encode', label: 'reading prompt' },
  { key: 'denoise', label: 'diffusion' },
  { key: 'decode', label: 'decoding' },
  { key: 'vocode', label: 'generating audio' },
]

// Where the bar sits at the start of each stage. Diffusion is most of the work.
const BOUNDS = [0, 0.12, 0.78, 0.88, 1]

const START_MS = 220 // let the input finish leaving first
const ENTER_MS = 640
const DWELL_MS = 700
const RESOLVE_MS = 220 // bar reaches the end of the stage before the label leaves

export function Sequence(props: { s: RunState; active: boolean; onDone: () => void }) {
  const { s, active, onDone } = props
  const [cursor, setCursor] = useState(-1)
  const [resolving, setResolving] = useState(false)
  const fill = useRef<HTMLElement>(null)
  const state = useRef({ s, cursor, resolving, enteredAt: 0, busy: false })
  state.current.s = s
  state.current.cursor = cursor
  state.current.resolving = resolving
  const cb = useRef(onDone)
  cb.current = onDone

  useEffect(() => {
    if (!active) {
      setCursor(-1)
      setResolving(false)
      return
    }
    const st = state.current
    st.busy = false
    const startedAt = performance.now()
    const id = setInterval(() => {
      const { s, cursor } = st
      const now = performance.now()
      if (st.busy) return
      if (cursor === -1) {
        if (now - startedAt >= START_MS) {
          st.enteredAt = now
          setCursor(0)
        }
        return
      }
      const done = s.stages[STAGES[cursor].key] === 'done'
      const behind = STAGES.slice(cursor + 1).filter((x) => s.stages[x.key] === 'done').length
      const dwell = behind >= 2 ? DWELL_MS * 0.6 : DWELL_MS
      const last = cursor === STAGES.length - 1
      if (!done || now - st.enteredAt < ENTER_MS + dwell) return
      if (last && s.settled < s.count) return
      st.busy = true
      setResolving(true)
      setTimeout(() => {
        setResolving(false)
        if (last) {
          setCursor(STAGES.length) // everything rises away
          cb.current()
        } else {
          st.enteredAt = performance.now()
          setCursor(cursor + 1)
          st.busy = false
        }
      }, RESOLVE_MS)
    }, 40)

    // the bar: real progress during diffusion, a slow ease towards the next
    // boundary everywhere else, and it never moves backwards
    let raf = 0
    let shown = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = now - last
      last = now
      const { s, cursor, resolving, enteredAt } = st
      let target = 0
      if (cursor >= STAGES.length) target = 1
      else if (cursor >= 0) {
        const lo = BOUNDS[cursor]
        const hi = BOUNDS[cursor + 1]
        const steps = s.steps.reduce((a, b) => a + b, 0)
        const inner = resolving
          ? 1
          : STAGES[cursor].key === 'denoise' && s.total
            ? steps / (s.total * s.count)
            : 1 - Math.exp(-(now - enteredAt) / 900)
        target = lo + (hi - lo) * inner
      }
      shown = Math.max(shown, shown + (target - shown) * (1 - Math.exp(-dt / 130))) // same speed at any frame rate
      if (fill.current) fill.current.style.transform = `scaleX(${shown})`
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      clearInterval(id)
      cancelAnimationFrame(raf)
      if (fill.current) fill.current.style.transform = 'scaleX(0)'
    }
  }, [active])

  const finished = cursor >= STAGES.length

  return (
    <div className={`sequence ${active ? 'on' : ''} ${finished ? 'is-finished' : ''}`} aria-live="polite">
      {STAGES.map((st, i) => {
        const off = i - cursor
        const pos = cursor === -1 ? 'below' : finished || off < 0 ? 'gone' : off === 0 ? 'current' : 'below'
        const current = pos === 'current'
        return (
          <div key={st.key} className={`stage is-${pos}`} data-quiet={pos === 'current' || pos === 'gone' ? '' : undefined}>
            {st.key === 'encode' && <ReadOut text={s.prompt} reading={current && !resolving} />}
            <span className="stage-label">{st.label}</span>
          </div>
        )
      })}
      <div className="bar" aria-hidden>
        <i ref={fill} />
      </div>
    </div>
  )
}

/** The prompt, read letter by letter: a highlight sweeps through it on a loop. */
function ReadOut({ text, reading }: { text: string; reading: boolean }) {
  const chars = [...text]
  const period = 1100 // ms, one full pass over the prompt
  return (
    <span className={`readout ${reading ? 'is-reading' : 'is-read'}`} aria-hidden>
      {chars.map((c, i) => (
        <i key={i} style={{ animationDelay: `${(i / Math.max(1, chars.length)) * period}ms` }}>
          {c}
        </i>
      ))}
    </span>
  )
}
