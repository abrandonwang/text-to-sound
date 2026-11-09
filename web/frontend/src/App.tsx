import { useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } from 'react'
import { generate, suggest, wakeServer, type GenEvent, type SuggestResult } from './api'
import { Finale } from './components/Finale'
import { LedField, newScene } from './components/LedField'
import { Sequence } from './components/Sequence'
import { TAKES, initialState, reducer, wavUrl } from './state'
import { completion, localSuggest, normPrompt } from './vocab/match'

type View = 'idle' | 'run' | 'finale' | 'error'

// Free text is allowed (the model is conditioned on CLAP text embeddings and
// trained on full captions); the server accepts up to 200 characters.
export const MAX_PROMPT = 200
// How close by meaning (CLAP cosine) a server match must be to show. Junk input
// ("pizza", "asdf") still lands on a few "hub" prompts at 0.60-0.65.
const MIN_MATCH = 0.66
const PAUSE_MS = 500 // a suggestion only appears once typing stops
const MAX_WORDS = 3 // longer prompts are already descriptive: leave them alone
const measureCtx = document.createElement('canvas').getContext('2d')!

export default function App() {
  const [s, dispatch] = useReducer(reducer, initialState)
  const [view, setView] = useState<View>('idle')
  const [prompt, setPrompt] = useState('')
  const [shake, setShake] = useState(false)
  const scene = useRef(newScene())
  const urls = useRef<string[]>([]) // audio of the current run, freed when the next one starts
  const input = useRef<HTMLInputElement>(null)
  const form = useRef<HTMLFormElement>(null)
  const abort = useRef<AbortController | null>(null)
  const timers = useRef<number[]>([])

  const later = (ms: number, fn: () => void) => timers.current.push(window.setTimeout(fn, ms))
  const clearTimers = () => {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }

  const onEvent = (ev: GenEvent) => {
    let url: string | undefined
    if (ev.type === 'result') {
      url = wavUrl(ev.wav)
      urls.current.push(url)
    }
    dispatch({ type: 'event', ev, url })
  }

  // the model server sleeps when idle: start waking it as soon as the page opens
  useEffect(() => {
    wakeServer()
  }, [])

  // Help while typing, at most one thing at a time and never in the way. Enter
  // always generates exactly what was typed; Tab takes the offer.
  //  - half-typed sound name: the rest appears dimmed inside the input
  //  - otherwise, once typing stops: one suggestion under it (typo fix or a
  //    different word first, then the server's closest match by meaning)
  const q = normPrompt(prompt)
  const rest = useMemo(() => completion(prompt), [prompt])
  const local = useMemo(() => localSuggest(prompt), [prompt])
  const [remote, setRemote] = useState<{ q: string; r: SuggestResult } | null>(null)
  const [paused, setPaused] = useState(false)
  const seeds = useRef(new Map<string, number>()) // known-good prompt -> the seed that sounds best
  useEffect(() => {
    setPaused(false)
    const id = setTimeout(() => setPaused(true), PAUSE_MS)
    return () => clearTimeout(id)
  }, [q])
  useEffect(() => {
    const ask = suggest // null in mock mode: local matching only
    if (!ask || q.length < 3 || q.split(' ').length > MAX_WORDS || view !== 'idle') return
    const ctl = new AbortController()
    const id = setTimeout(() => {
      ask(q, ctl.signal)
        .then((r) => {
          r.suggestions.forEach((x) => seeds.current.set(x.prompt, x.seed))
          setRemote({ q, r })
        })
        .catch(() => {})
    }, 300)
    return () => {
      clearTimeout(id)
      ctl.abort()
    }
  }, [q, view])
  const hint = useMemo(() => {
    if (rest) return null
    if (local) return { text: local.prompt, label: local.kind === 'typo' ? 'did you mean' : 'try' }
    if (remote?.q !== q || remote.r.exact) return null
    const top = remote.r.suggestions[0]
    return top && top.similarity >= MIN_MATCH && top.prompt !== q ? { text: top.prompt, label: 'try' } : null
  }, [rest, local, remote, q])
  const hintOn = !!hint && paused && view === 'idle'
  // keep the last suggestion on screen while it fades out
  const shown = useRef({ text: '', label: '' })
  if (hint && hintOn) shown.current = hint

  // fills the box and leaves the choice to Enter; a run is never started from here
  const accept = (text: string) => {
    setPrompt(text)
    input.current?.focus()
  }

  const run = async () => {
    const text = normPrompt(prompt)
    if (!text) {
      setShake(true)
      later(260, () => setShake(false))
      return
    }
    input.current?.blur()
    abort.current?.abort()
    clearTimers()
    const ctl = new AbortController()
    abort.current = ctl
    urls.current.forEach(URL.revokeObjectURL)
    urls.current = []
    scene.current.glow = false
    setView('run')
    dispatch({ type: 'start', prompt: text, steps: 50, count: TAKES })

    // one take at a time, so one visitor doesn't fill the server's queue. The
    // first take of a known-good prompt uses its best seed; the rest are fresh.
    let made = 0
    for (let take = 0; take < TAKES; take++) {
      const seed = (take === 0 ? seeds.current.get(text) : undefined) ?? Math.floor(Math.random() * 1e5)
      try {
        for await (const ev of generate({ prompt: text, guidance: 3, steps: 50, seed }, ctl.signal)) {
          if (ctl.signal.aborted) return
          if (ev.type === 'result') made++
          onEvent(ev)
        }
      } catch (e) {
        if ((e as Error).name === 'AbortError') return
        onEvent({ type: 'error', message: String(e) })
      }
      if (ctl.signal.aborted) return
    }
    if (!made) {
      scene.current.glow = true
      setView('error')
    }
  }

  // sequence finished: the last label rises away, a beat of blackout, then the players
  const onSequenceDone = () => later(560, () => setView('finale'))

  const backToIdle = (initial?: string) => {
    abort.current?.abort()
    clearTimers()
    scene.current.glow = true
    setView('idle')
    if (initial !== undefined) setPrompt(initial)
    requestAnimationFrame(() => {
      const el = input.current
      if (!el) return
      el.focus()
      el.setSelectionRange(el.value.length, el.value.length) // cursor at the end, nothing selected
    })
  }

  // keyboard on the error screen: enter retries, esc goes back
  useEffect(() => {
    if (view !== 'error') return
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest?.('button') && e.key === 'Enter') return
      if (e.key === 'Enter') run()
      else if (e.key === 'Escape') backToIdle()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

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
    form.current?.style.setProperty('--fit', `${base * fit}px`)
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
              if ((e.key === 'Tab' || e.key === 'ArrowRight') && rest) {
                e.preventDefault()
                accept(prompt + rest)
              } else if (e.key === 'Tab' && hintOn && hint) {
                e.preventDefault()
                accept(hint.text)
              }
            }}
            onSelect={(e) => {
              const el = e.currentTarget
              const end = el.value.length
              if (el.selectionStart !== end || el.selectionEnd !== end) el.setSelectionRange(end, end)
            }}
            onFocus={() => (scene.current.focus = true)}
            onBlur={() => (scene.current.focus = false)}
          />
          {rest && (
            // the typed text is invisible here: it only holds the dimmed rest in the right place
            <div className="ghost" aria-hidden>
              <span>
                {prompt}
                <i onMouseDown={(e) => e.preventDefault()} onClick={() => accept(prompt + rest)}>
                  {rest}
                </i>
              </span>
            </div>
          )}
          <div className={`suggest ${hintOn ? 'on' : ''}`} data-quiet="small" aria-hidden={!hintOn}>
            <span className="suggest-label">{shown.current.label}</span>
            <button type="button" tabIndex={-1} onMouseDown={(e) => e.preventDefault()} onClick={() => accept(shown.current.text)}>
              {shown.current.text}
            </button>
            <kbd>tab</kbd>
          </div>
        </form>

        <Sequence s={s} active={view === 'run'} onDone={onSequenceDone} />

        <Finale on={view === 'finale'} prompt={s.prompt} takes={s.takes} onAnother={backToIdle} />

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

    </>
  )
}
