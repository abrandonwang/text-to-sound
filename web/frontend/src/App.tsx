import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { generate, type GenEvent, type MelImage } from './api'
import { LedField, newScene } from './components/LedField'
import { Steps } from './components/Steps'
import { initialState, reducer } from './state'

function decodeMel(m: MelImage): Uint8Array {
  const bin = atob(m.data)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export default function App() {
  const [s, dispatch] = useReducer(reducer, initialState)
  const [prompt, setPrompt] = useState('')
  const [playing, setPlaying] = useState(false)
  const scene = useRef(newScene())
  const resultMel = useRef<Uint8Array | null>(null)
  const audio = useRef<HTMLAudioElement>(null)
  const analyser = useRef<AnalyserNode | null>(null)
  const abort = useRef<AbortController | null>(null)
  const busy = s.phase === 'queued' || s.phase === 'running'

  // Web Audio graph, created on the first click (browsers require a user gesture).
  const ensureAudioGraph = () => {
    if (analyser.current || !audio.current) return
    const ctx = new AudioContext()
    const src = ctx.createMediaElementSource(audio.current)
    const an = ctx.createAnalyser()
    an.fftSize = 512
    src.connect(an).connect(ctx.destination)
    analyser.current = an
  }

  const onEvent = (ev: GenEvent) => {
    const sc = scene.current
    if (ev.type === 'stage_start') sc.pulse = 1
    if (ev.type === 'stage_start' && ev.stage === 'denoise') sc.noise = 1
    if (ev.type === 'denoise_step') {
      sc.noise = ev.noise
      if (ev.preview) sc.mel = decodeMel(ev.preview)
    }
    if (ev.type === 'result') {
      sc.mel = resultMel.current = decodeMel(ev.mel)
      sc.noise = 0
    }
    if (ev.type === 'error') sc.noise = 0
    dispatch({ type: 'event', ev })
  }

  const run = useCallback(async () => {
    const text = prompt.trim()
    if (!text || busy) return
    ensureAudioGraph()
    audio.current?.pause()
    abort.current?.abort()
    const ctl = new AbortController()
    abort.current = ctl
    Object.assign(scene.current, { mel: null, noise: 0, playhead: null, pulse: 1 })
    dispatch({ type: 'start', prompt: text, steps: 50 })
    try {
      for await (const ev of generate({ prompt: text, guidance: 3, steps: 50, seed: 42 }, ctl.signal)) onEvent(ev)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') onEvent({ type: 'error', message: String(e) })
    }
  }, [prompt, busy])

  // autoplay the result
  useEffect(() => {
    const a = audio.current
    if (!a || !s.result) return
    a.src = s.result.url
    a.currentTime = 0
    a.play().catch(() => {})
  }, [s.result])

  // feed playback position + loudness into the LED field
  useEffect(() => {
    if (!playing) {
      scene.current.playhead = null
      scene.current.level = 0
      return
    }
    const buf = new Uint8Array(512)
    let raf = 0
    const tick = () => {
      const a = audio.current
      if (a && a.duration) scene.current.playhead = a.currentTime / a.duration
      const an = analyser.current
      if (an) {
        an.getByteTimeDomainData(buf)
        let sum = 0
        for (let i = 0; i < an.fftSize; i++) sum += ((buf[i] - 128) / 128) ** 2
        scene.current.level = Math.min(1, Math.sqrt(sum / an.fftSize) * 4)
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [playing])

  return (
    <>
      <LedField scene={scene} />
      <div className="vignette" aria-hidden />

      <Steps s={s} />

      <main className="center">
        <form
          className={`prompt ${busy ? 'is-busy' : ''}`}
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <input
            autoFocus
            value={prompt}
            maxLength={200}
            spellCheck={false}
            autoComplete="off"
            aria-label="Describe a sound"
            placeholder="type a sound"
            onChange={(e) => setPrompt(e.target.value)}
          />
        </form>
        <div className="under">
          {s.phase === 'error' ? (
            <span className="err">{s.error}</span>
          ) : (
            <>
            <button className={`${s.result && !busy ? 'on' : ''} ${playing ? 'is-playing' : ''}`} onClick={() => audio.current?.play()} tabIndex={s.result ? 0 : -1}>
              <svg viewBox="0 0 10 12" aria-hidden>
                <path d="M0 0l10 6-10 6z" />
              </svg>
              play again
            </button>
            <a
              className={s.result && !busy ? 'on' : ''}
              href={s.result?.url}
              download={`${(s.result?.prompt ?? 'sound').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40)}.wav`}
              tabIndex={s.result ? 0 : -1}
            >
              <svg viewBox="0 0 10 12" aria-hidden>
                <path d="M4 0h2v6.5l2.3-2.3 1.4 1.4L5 10.3.3 5.6l1.4-1.4L4 6.5zM0 11h10v1H0z" />
              </svg>
              download
            </a>
            </>
          )}
        </div>
      </main>

      <audio
        ref={audio}
        onPlay={() => {
          scene.current.mel = resultMel.current // show the sound while it plays
          setPlaying(true)
        }}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          scene.current.mel = null // ease back into the idle wave
          setPlaying(false)
        }}
      />
    </>
  )
}
