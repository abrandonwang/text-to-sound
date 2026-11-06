import { useEffect, useRef, useState } from 'react'
import type { StageId } from '../api'
import type { RunState } from '../state'

// Full-screen step sequence. One stage label at a time, centre stage, with a
// tiny LED loader. When a stage is done it flies up and disappears, and the
// next one rises in from below.
//
// The *visual* cursor is separate from the backend: it only moves on when the
// backend has finished that stage AND the label has been on screen long enough
// to read. So a 90 ms stage still gets its moment, and nothing is ever skipped.

export const STAGES: { key: StageId; label: string; accent: string }[] = [
  { key: 'encode', label: 'reading prompt', accent: '#ff2bd6' },
  { key: 'denoise', label: 'diffusion', accent: '#2be4ff' },
  { key: 'decode', label: 'decoding', accent: '#3dff8b' },
  { key: 'vocode', label: 'generating audio', accent: '#ffe14d' },
]

const ENTER_MS = 640
const DWELL_MS = 700
const RESOLVE_MS = 220 // loader completes before the stage leaves
const SQUARES = 12

export function Sequence(props: { s: RunState; active: boolean; onStage: (i: number) => void; onDone: () => void }) {
  const { s, active, onStage, onDone } = props
  const [cursor, setCursor] = useState(-1)
  const [resolving, setResolving] = useState(false)
  const state = useRef({ s, cursor, enteredAt: 0, busy: false })
  state.current.s = s
  state.current.cursor = cursor
  const cb = useRef({ onStage, onDone })
  cb.current = { onStage, onDone }

  useEffect(() => {
    if (!active) {
      setCursor(-1)
      setResolving(false)
      return
    }
    const st = state.current
    st.busy = false
    const id = setInterval(() => {
      const { s, cursor } = st
      const now = performance.now()
      if (st.busy) return
      if (cursor === -1) {
        if (s.stages.encode.status !== 'idle') {
          st.enteredAt = now
          setCursor(0)
          cb.current.onStage(0)
        }
        return
      }
      const done = s.stages[STAGES[cursor].key].status === 'done'
      const behind = STAGES.slice(cursor + 1).filter((x) => s.stages[x.key].status === 'done').length
      const dwell = behind >= 2 ? DWELL_MS * 0.6 : DWELL_MS
      const last = cursor === STAGES.length - 1
      if (!done || now - st.enteredAt < ENTER_MS + dwell) return
      if (last && !s.result) return
      st.busy = true
      setResolving(true)
      setTimeout(() => {
        setResolving(false)
        if (last) {
          setCursor(STAGES.length) // everything rises away
          cb.current.onDone()
        } else {
          st.enteredAt = performance.now()
          setCursor(cursor + 1)
          cb.current.onStage(cursor + 1)
          st.busy = false
        }
      }, RESOLVE_MS)
    }, 40)
    return () => clearInterval(id)
  }, [active])

  const progress = s.total ? s.step / s.total : 0

  return (
    <div className={`sequence ${active ? 'on' : ''}`} aria-live="polite">
      {STAGES.map((st, i) => {
        const off = i - cursor
        const finished = cursor >= STAGES.length
        const pos =
          cursor === -1 ? 'below' : finished || off < 0 ? 'gone' : off === 0 ? 'current' : 'below'
        const current = pos === 'current'
        // loader state
        let lit = 0
        if (st.key === 'denoise') lit = Math.round(progress * SQUARES)
        if (resolving && current) lit = SQUARES
        const sweeping = current && !resolving && st.key !== 'denoise'
        return (
          <div key={st.key} className={`stage is-${pos}`} data-quiet={pos === 'current' || pos === 'gone' ? '' : undefined}>
            <span className="stage-label">{st.label}</span>
            {st.key === 'encode' ? (
              <ReadOut text={s.prompt} reading={current && !resolving} />
            ) : (
            <span className={`loader ${sweeping ? 'is-sweeping' : ''}`} style={{ '--accent': st.accent } as React.CSSProperties} aria-hidden>
              {Array.from({ length: SQUARES }, (_, k) => (
                <i key={k} className={k < lit ? 'lit' : ''} style={{ animationDelay: `${k * 75}ms` }} />
              ))}
            </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** The prompt, read letter by letter: a highlight sweeps through it on a loop. */
function ReadOut({ text, reading }: { text: string; reading: boolean }) {
  const chars = [...text]
  const period = 1100 // ms, same as the LED scanner
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
