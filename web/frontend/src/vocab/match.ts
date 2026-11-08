import { SOUNDS } from './sounds'

// Autocorrect a prompt to the closest sound the model knows.
//   "rain"   -> ok
//   "woof"   -> suggest "bark"     (known alias)
//   "rian"   -> suggest "rain"     (typo, edit distance)
//   "thund"  -> suggest "thunder"  (unfinished word)
//   "ninja"  -> suggest "whoosh"   (closest thing we have)

export interface Check {
  ok: boolean
  suggestion: string | null
}

const norm = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9\- ]/g, ' ').replace(/\s+/g, ' ').trim()

const NAMES = new Set(SOUNDS.map((s) => s.name))
const KEYS: { key: string; name: string }[] = SOUNDS.flatMap((s) => [s.name, ...s.aliases].map((key) => ({ key, name: s.name })))
const EXACT = new Map(KEYS.map((k) => [k.key, k.name]))

/** Edit distance with swaps (so "rian" is 1 away from "rain"). */
function dist(a: string, b: string) {
  const d: number[][] = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)])
  for (let j = 1; j <= b.length; j++) d[0][j] = j
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++) {
      const c = a[i - 1] === b[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1)
    }
  return d[a.length][b.length]
}

/** "dogs" -> "dog", "barking" -> "bark", ... */
function exactish(q: string) {
  for (const v of [q, q.replace(/s$/, ''), q.replace(/es$/, ''), q.replace(/ing$/, ''), q.replace(/ing$/, 'e')]) {
    const hit = EXACT.get(v)
    if (hit) return hit
  }
  return null
}

export function checkSound(input: string): Check {
  const q = norm(input)
  if (!q) return { ok: false, suggestion: null }
  if (q.length < 2) return { ok: false, suggestion: null } // too short to guess
  if (NAMES.has(q)) return { ok: true, suggestion: null }

  const hit = exactish(q)
  if (hit) return { ok: false, suggestion: hit }

  // one known word inside a longer prompt ("loud rain") -> that sound
  for (const w of q.split(' ').sort((a, b) => b.length - a.length)) {
    const h = w.length >= 3 ? exactish(w) : null
    if (h) return { ok: false, suggestion: h }
  }

  let best = KEYS[0].name
  let bestScore = Infinity
  for (const { key, name } of KEYS) {
    let score: number
    if (q.length >= 2 && key.startsWith(q)) score = 0.1 * ((key.length - q.length) / key.length)
    else score = dist(q, key) / Math.max(q.length, key.length)
    if (score < bestScore) {
      bestScore = score
      best = name
    }
  }
  return { ok: false, suggestion: best }
}
