import type { RunState } from '../state'

// The pipeline as a rotating drum: the current step sits in front, the others
// tilt away above and below it. It turns one notch each time a stage starts.

const STEPS = [
  { key: 'encode', label: 'reading prompt' },
  { key: 'denoise', label: 'diffusion' },
  { key: 'decode', label: 'decoding' },
  { key: 'vocode', label: 'generating audio' },
] as const

const ANGLE = 34 // degrees between items on the drum
const RADIUS = 58 // px

export function Steps({ s }: { s: RunState }) {
  // front = latest stage that has started
  let front = 0
  STEPS.forEach((st, i) => {
    if (s.stages[st.key].status !== 'idle') front = i
  })
  const visible = s.phase !== 'idle'
  const progress = s.total ? s.step / s.total : 0

  return (
    <aside className={`side ${visible ? 'on' : ''}`} aria-live="polite">
      <div className="drum">
        {STEPS.map((st, i) => {
          const status = s.stages[st.key].status
          const a = ((i - front) * ANGLE * Math.PI) / 180
          const y = Math.sin(a) * RADIUS
          const z = (Math.cos(a) - 1) * RADIUS
          const isFront = i === front
          return (
            <div
              key={st.key}
              className={`drum-item is-${status} ${isFront ? 'is-front' : ''}`}
              style={{
                transform: `translate3d(0, ${y}px, ${z}px) rotateX(${-(i - front) * ANGLE}deg)`,
                opacity: Math.max(0, 1 - Math.abs(i - front) * 0.3),
              }}
            >
              <span className="dot" />
              {st.label}
              {st.key === 'denoise' && (
                <span className="track">
                  <span style={{ transform: `scaleX(${status === 'idle' ? 0 : status === 'done' ? 1 : progress})` }} />
                </span>
              )}
            </div>
          )
        })}
      </div>
    </aside>
  )
}
