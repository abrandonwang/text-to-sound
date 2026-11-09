import { SOUNDS } from './sounds'

// Local half of the prompt help. Matching by meaning happens on the server
// (/suggest). Only short prompts (1-2 words) are checked here; longer free text
// is left alone, since full phrases work better than a single label.
//   completion("thund") -> "er"      (finishes the word, shown dimmed in the input)
//   localSuggest:
//   "rain"   -> nothing (already a known sound)
//   "dogs"   -> nothing (just a plural, fine as typed)
//   "woof"   -> "bark"  (different word for a sound we know)
//   "rian"   -> "rain"  (typo, edit distance)
//   "pizza"  -> nothing (no close spelling; the server matches by meaning)

export interface LocalFix {
  prompt: string
  kind: 'alias' | 'typo'
}

/** Lowercase, single spaces: how prompts are compared and sent. */
export const normPrompt = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim()
const norm = (s: string) => normPrompt(s.replace(/[^a-zA-Z0-9\- ]/g, ' '))

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

/** "dogs" -> "dog", "barking" -> "bark", "woof" -> "bark" */
function exactish(q: string) {
  for (const v of [q, q.replace(/s$/, ''), q.replace(/es$/, ''), q.replace(/ing$/, ''), q.replace(/ing$/, 'e')]) {
    const hit = EXACT.get(v)
    if (hit) return hit
  }
  return null
}

/** The rest of a half-typed sound name: "thund" -> "er", "" if there's nothing to finish. */
export function completion(input: string) {
  const q = input.toLowerCase().trimStart()
  if (q.length < 3 || EXACT.has(q.trim())) return ''
  let best = ''
  for (const { key } of KEYS) if (key.length > q.length && key.startsWith(q) && (!best || key.length < best.length)) best = key
  return best.slice(q.length)
}

export function localSuggest(input: string): LocalFix | null {
  const q = norm(input)
  if (q.length < 2 || q.split(' ').length > 2 || NAMES.has(q)) return null

  const hit = exactish(q)
  // "barking" or "dogs" are fine as typed: only offer a genuinely different word
  if (hit) return q.slice(0, 3) === hit.slice(0, 3) ? null : { prompt: hit, kind: 'alias' }

  // typo: at most 2 edits, and no more than a third of the word
  let best: string | null = null
  let bestD = Infinity
  for (const { key, name } of KEYS) {
    const d = dist(q, key)
    if (d < bestD && d <= 2 && d / Math.max(q.length, key.length) <= 0.34) {
      bestD = d
      best = name
    }
  }
  return best ? { prompt: best, kind: 'typo' } : null
}
