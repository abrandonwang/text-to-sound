// Closest known-good prompts by meaning, from the model server:
//   GET {base}/suggest?q=puppy&k=3 -> { exact, suggestions: [{ prompt, seed, similarity }] }
// similarity is CLAP cosine (~0.4 to 1.0), best first.

export interface Suggestion {
  prompt: string
  seed: number
  similarity: number
}
export interface SuggestResult {
  query: string
  exact: boolean
  suggestions: Suggestion[]
}

export function suggester(baseUrl: string) {
  const base = baseUrl.replace(/\/$/, '')
  const cache = new Map<string, SuggestResult>()
  return async (q: string, signal?: AbortSignal): Promise<SuggestResult> => {
    const hit = cache.get(q)
    if (hit) return hit
    const r = await fetch(`${base}/suggest?${new URLSearchParams({ q, k: '3' })}`, { signal })
    if (!r.ok) throw new Error(`suggest ${r.status}`)
    const res = (await r.json()) as SuggestResult
    cache.set(q, res)
    return res
  }
}

/** The server sleeps after ~5 min idle and takes ~20 s to wake: poke it on page load. */
export function wake(baseUrl: string) {
  fetch(`${baseUrl.replace(/\/$/, '')}/health`).catch(() => {})
}
