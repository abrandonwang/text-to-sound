import { Fragment, useEffect, useRef, useState } from 'react'
import type { Take } from '../state'

// The reveal: the prompt as the title and one mini player per take. Only one
// plays at a time; each has its own download.

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

const clock = (sec: number) => `0:${String(Math.round(sec)).padStart(2, '0')}`

export function Finale(props: { on: boolean; prompt: string; takes: Take[]; onAnother: (initial?: string) => void }) {
  const { on, prompt, takes, onAnother } = props
  const root = useRef<HTMLDivElement>(null)
  const audio = useRef<HTMLAudioElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const [current, setCurrent] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [pos, setPos] = useState(0) // 0..1 through the current take
  const [saved, setSaved] = useState<number | null>(null)
  const savedTimer = useRef(0)
  const slug = prompt.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 44) || 'sound'

  // hidden: stop the sound and take the players out of tab order
  useEffect(() => {
    root.current?.toggleAttribute('inert', !on)
    if (on) return
    audio.current?.pause()
    audio.current?.removeAttribute('src')
    setCurrent(0)
    setPos(0)
  }, [on])

  // smooth progress while something plays
  useEffect(() => {
    if (!playing) return
    let raf = 0
    const tick = () => {
      const a = audio.current
      if (a?.duration) setPos(a.currentTime / a.duration)
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [playing])

  const isLoaded = (i: number) => i === current && audio.current?.src === takes[i]?.url

  const start = (i: number, frac: number) => {
    const a = audio.current
    if (!a || !takes[i]) return
    setCurrent(i)
    setPos(frac)
    if (a.src !== takes[i].url) a.src = takes[i].url
    a.currentTime = frac * takes[i].seconds
    a.play().catch(() => {})
  }

  const toggle = (i: number) => {
    const a = audio.current
    if (!a) return
    if (!isLoaded(i)) start(i, 0)
    else if (a.paused) a.play().catch(() => {})
    else a.pause()
  }

  const seek = (i: number, frac: number) => {
    const a = audio.current
    if (!a) return
    if (!isLoaded(i)) return start(i, frac)
    a.currentTime = frac * takes[i].seconds
    setPos(frac)
  }

  // keyboard: space play/pause, 1-4 pick a take, enter replay, d download, esc or any other key = make another
  useEffect(() => {
    if (!on) return
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as HTMLElement).closest?.('button, a') && (e.key === 'Enter' || e.key === ' ')) return
      const digit = Number(e.key)
      if (e.key === ' ') {
        e.preventDefault()
        toggle(current)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        start(current, 0)
      } else if (e.key === 'Escape') onAnother()
      else if (e.key.toLowerCase() === 'd') list.current?.querySelectorAll('a')[current]?.click()
      else if (digit >= 1 && digit <= takes.length) toggle(digit - 1)
      else if (e.key.length === 1) {
        e.preventDefault()
        onAnother(e.key)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // letters stagger in, words never break mid-word
  let n = 0
  const words = `“${prompt}”`.split(' ')

  return (
    <div ref={root} className={`finale ${on ? 'on' : ''}`} aria-hidden={!on}>
      <h1 className={`title ${prompt.length > 120 ? 'is-longer' : prompt.length > 50 ? 'is-long' : ''}`}>
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

      <ul className="takes" ref={list}>
        {takes.map((t, i) => {
          const live = i === current
          const here = live ? pos : 0
          const going = live && playing
          return (
            <li key={t.url} className={`take ${going ? 'is-playing' : ''}`} style={{ '--i': i, '--p': `${here * 100}%` } as React.CSSProperties}>
              <button className="take-play" onClick={() => toggle(i)} aria-label={`${going ? 'Pause' : 'Play'} version ${i + 1}`}>
                {going ? <Pause /> : <Play />}
              </button>
              <input
                className="take-track"
                type="range"
                min={0}
                max={1}
                step={0.001}
                value={here}
                aria-label={`Position in version ${i + 1}`}
                onChange={(e) => seek(i, Number(e.target.value))}
              />
              <span className="take-time">{clock(live && here > 0 ? here * t.seconds : t.seconds)}</span>
              <a
                className={`take-save ${saved === i ? 'is-saved' : ''}`}
                href={t.url}
                download={`${slug}-${i + 1}.wav`}
                aria-label={`Download version ${i + 1}`}
                onClick={() => {
                  setSaved(i)
                  clearTimeout(savedTimer.current)
                  savedTimer.current = window.setTimeout(() => setSaved(null), 1600)
                }}
              >
                {saved === i ? <Check /> : <Down />}
              </a>
            </li>
          )
        })}
      </ul>

      <div className="controls">
        <button onClick={() => onAnother()}>
          <Plus />
          <span>make another</span>
        </button>
      </div>

      <audio
        ref={audio}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false)
          setPos(0)
        }}
      />
    </div>
  )
}
