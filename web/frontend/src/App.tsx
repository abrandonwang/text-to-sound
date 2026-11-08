import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react'
import { generate, type GenEvent, type MelImage } from './api'
import { Finale } from './components/Finale'
import { LedField, newScene, setMode, type LedMode } from './components/LedField'
import { Sequence } from './components/Sequence'
import { initialState, reducer } from './state'

type View = 'idle' | 'run' | 'finale' | 'error'
const STAGE_MODES: LedMode[] = ['encode', 'denoise', 'decode', 'vocode']

// Prompt length cap: short prompts keep the input big, clean and on one line.
export const MAX_PROMPT = 15
const measureCtx = document.createElement('canvas').getContext('2d')!

function decodeMel(m: MelImage): Uint8Array {
  const bin = atob(m.data)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export default function App() {
  const [s, dispatch] = useReducer(reducer, initialState)
  const [view, setView] = useState<View>('idle')
  const [prompt, setPrompt] = useState('')
  const [playing, setPlaying] = useState(false)
  const [shake, setShake] = useState(false)
  const scene = useRef(newScene())
  const resultMel = useRef<Uint8Array | null>(null)
  const audio = useRef<HTMLAudioElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const form = useRef<HTMLFormElement>(null)
  const analyser = useRef<AnalyserNode | null>(null)
  const audioCtx = useRef<AudioContext | null>(null)
  const abort = useRef<AbortController | null>(null)
  const timers = useRef<number[]>([])

  const later = (ms: number, fn: () => void) => timers.current.push(window.setTimeout(fn, ms))
  const clearTimers = () => {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }

  // Web Audio graph, created inside a user gesture (browsers require it).
  const ensureAudio = () => {
    if (!audioCtx.current && audio.current) {
      const ctx = new AudioContext()
      const src = ctx.createMediaElementSource(audio.current)
      const an = ctx.createAnalyser()
      an.fftSize = 512
      src.connect(an).connect(ctx.destination)
      audioCtx.current = ctx
      analyser.current = an
    }
    audioCtx.current?.resume()
  }

  const onEvent = (ev: GenEvent) => {
    const sc = scene.current
    if (ev.type === 'denoise_step') {
      sc.noise = ev.noise
      if (ev.preview) sc.mel = decodeMel(ev.preview)
    }
    if (ev.type === 'result') sc.mel = resultMel.current = decodeMel(ev.mel)
    if (ev.type === 'error') {
      setView('error')
      setMode(sc, 'idle')
      sc.mel = null
    }
    dispatch({ type: 'event', ev })
  }

  // load the result into the player as soon as it arrives
  useEffect(() => {
    if (audio.current && s.result) audio.current.src = s.result.url
  }, [s.result])

  const run = useCallback(async () => {
    const text = prompt.trim()
    if (!text) {
      setShake(true)
      later(260, () => setShake(false))
      return
    }
    ensureAudio()
    input.current?.blur()
    abort.current?.abort()
    clearTimers()
    const ctl = new AbortController()
    abort.current = ctl
    Object.assign(scene.current, { mel: null, noise: 0, playhead: null })
    resultMel.current = null
    setMode(scene.current, 'inhale')
    setView('run')
    dispatch({ type: 'start', prompt: text, steps: 50 })
    try {
      const seed = Math.floor(Math.random() * 1e5)
      for await (const ev of generate({ prompt: text, guidance: 3, steps: 50, seed }, ctl.signal)) onEvent(ev)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') onEvent({ type: 'error', message: String(e) })
    }
  }, [prompt])

  // sequence finished: the last label rises away, a beat of blackout, then the slam
  const onSequenceDone = () => {
    later(420, () => setMode(scene.current, 'drop'))
    later(560, () => {
      scene.current.mel = resultMel.current
      setMode(scene.current, 'finale')
      setView('finale')
      audio.current?.play().catch(() => {})
    })
  }

  const backToIdle = (initial?: string) => {
    abort.current?.abort()
    clearTimers()
    audio.current?.pause()
    setMode(scene.current, 'idle')
    scene.current.mel = null
    setView('idle')
    if (initial !== undefined) setPrompt(initial)
    requestAnimationFrame(() => {
      const el = input.current
      if (!el) return
      el.focus()
      el.setSelectionRange(el.value.length, el.value.length) // cursor at the end, nothing selected
    })
  }

  const togglePlay = () => {
    const a = audio.current
    if (!a) return
    ensureAudio()
    if (a.paused) a.play().catch(() => {})
    else a.pause()
  }

  // keyboard: finale (space play/pause, enter replay, d download, esc or any key = make another)
  useEffect(() => {
    if (view !== 'finale' && view !== 'error') return
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as HTMLElement).closest?.('button, a') && (e.key === 'Enter' || e.key === ' ')) return
      if (view === 'error') {
        if (e.key === 'Enter') run()
        else if (e.key === 'Escape') backToIdle()
        return
      }
      if (e.key === ' ') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (audio.current) audio.current.currentTime = 0
        audio.current?.play().catch(() => {})
      } else if (e.key === 'Escape') backToIdle()
      else if (e.key.toLowerCase() === 'd') document.querySelector<HTMLAnchorElement>('.controls a')?.click()
      else if (e.key.length === 1) {
        e.preventDefault()
        backToIdle(e.key)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // playback position + loudness -> LED panel
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

  // shrink the prompt text to fit on one line inside the box as it gets longer
  const [, forceFit] = useState(0)
  useEffect(() => {
    const onResize = () => forceFit((n) => n + 1)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  useLayoutEffect(() => {
    const el = input.current
    if (!el) return
    const base = Math.min(46, Math.max(28, window.innerWidth * 0.034))
    measureCtx.font = `400 ${base}px Geist, system-ui, sans-serif`
    const w = measureCtx.measureText(prompt || el.placeholder).width * 1.02
    const fit = Math.max(0.5, Math.min(1, (el.clientWidth - 24) / Math.max(1, w)))
    el.style.fontSize = `${base * fit}px`
  })

  // the input can't be reached while it's hidden
  useEffect(() => {
    form.current?.toggleAttribute('inert', view !== 'idle')
  }, [view])

  return (
    <>
      <LedField scene={scene} />
      <div className="vignette" aria-hidden />

      <main className={`stage-root view-${view}`}>
        <form
          ref={form}
          className={`prompt ${shake ? 'shake' : ''}`}
          data-quiet="small"
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <input
            ref={input}
            autoFocus
            value={prompt}
            maxLength={MAX_PROMPT}
            spellCheck={false}
            autoComplete="off"
            aria-label="Describe a sound"
            placeholder="type a sound"
            onChange={(e) => {
              setPrompt(e.target.value)
            }}
            // caret is locked to the end: just type and delete
            onMouseDown={(e) => {
              e.preventDefault()
              input.current?.focus()
            }}
            onKeyDown={(e) => {
              if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) e.preventDefault()
            }}
            onSelect={(e) => {
              const el = e.currentTarget
              const end = el.value.length
              if (el.selectionStart !== end || el.selectionEnd !== end) el.setSelectionRange(end, end)
            }}
            onFocus={() => (scene.current.focus = true)}
            onBlur={() => (scene.current.focus = false)}
          />
        </form>

        <Sequence s={s} active={view === 'run'} onStage={(i) => setMode(scene.current, STAGE_MODES[i])} onDone={onSequenceDone} />

        <Finale
          on={view === 'finale'}
          prompt={s.prompt}
          url={s.result?.url}
          playing={playing}
          onToggle={togglePlay}
          onAnother={() => backToIdle()}
        />

        {view === 'error' && (
          <div className="error" role="alert">
            <p className="error-label" data-quiet="">
              that one didn’t work
            </p>
            <div className="controls" data-quiet="small">
              <button onClick={() => run()}>try again</button>
              <button onClick={() => backToIdle()}>edit prompt</button>
            </div>
          </div>
        )}
      </main>

      <a className={`sig view-${view}`} href="https://github.com/abrandonwang" target="_blank" rel="noopener" data-quiet="small">
        Brandon Wang
      </a>

      <audio ref={audio} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
    </>
  )
}
