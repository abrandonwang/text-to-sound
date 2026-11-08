import { Fragment, useState } from 'react'

// The reveal: the prompt as the title (top), controls (bottom), and the
// spectrogram owning the middle of the LED panel.

const Play = () => (
  <svg viewBox="0 0 10 12" aria-hidden>
    <path d="M0 0l10 6-10 6z" />
  </svg>
)
const Pause = () => (
  <svg viewBox="0 0 10 12" aria-hidden>
    <path d="M0 0h3.5v12H0zM6.5 0H10v12H6.5z" />
  </svg>
)
const Down = () => (
  <svg viewBox="0 0 10 12" aria-hidden>
    <path d="M4 0h2v6.6l2.3-2.3 1.4 1.4L5 10.4.3 5.7l1.4-1.4L4 6.6zM0 11h10v1H0z" />
  </svg>
)
const Check = () => (
  <svg viewBox="0 0 12 12" aria-hidden>
    <path d="M1 6.5l3.2 3.2L11 2.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
  </svg>
)
const Plus = () => (
  <svg viewBox="0 0 10 10" aria-hidden>
    <path d="M4.2 0h1.6v10H4.2zM0 4.2h10v1.6H0z" />
  </svg>
)

export function Finale(props: {
  on: boolean
  prompt: string
  url?: string
  playing: boolean
  onToggle: () => void
  onAnother: () => void
}) {
  const { on, prompt, url, playing, onToggle, onAnother } = props
  const [saved, setSaved] = useState(false)
  const file = `${prompt.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'sound'}.wav`

  // letters stagger in, words never break mid-word
  let n = 0
  const words = `“${prompt}”`.split(' ')

  return (
    <div className={`finale ${on ? 'on' : ''}`} aria-hidden={!on}>
      <h1 className={`title ${prompt.length > 120 ? 'is-longer' : prompt.length > 50 ? 'is-long' : ''}`} data-quiet="">
        {on &&
          words.map((w, wi) => (
            <Fragment key={wi}>
              <span className="word">
                {[...w].map((ch, ci) => (
                  <span className="ch" key={ci} style={{ animationDelay: `${Math.min(n++, 60) * 22}ms` }}>
                    {ch}
                  </span>
                ))}
              </span>{' '}
            </Fragment>
          ))}
      </h1>

      <div className="controls" data-quiet="small">
        <button onClick={onToggle} tabIndex={on ? 0 : -1}>
          {playing ? <Pause /> : <Play />}
          <span>{playing ? 'pause' : 'play'}</span>
        </button>
        <a
          href={url}
          download={file}
          tabIndex={on ? 0 : -1}
          className={saved ? 'is-saved' : ''}
          onClick={() => {
            setSaved(true)
            setTimeout(() => setSaved(false), 1600)
          }}
        >
          {saved ? <Check /> : <Down />}
          <span>{saved ? 'saved' : 'download'}</span>
        </a>
        <button onClick={onAnother} tabIndex={on ? 0 : -1}>
          <Plus />
          <span>make another</span>
        </button>
      </div>
    </div>
  )
}
