import type { Generator } from './events'
import { mockGenerate } from './mockClient'
import { sseGenerator } from './sseClient'
import { suggester, wake, type SuggestResult } from './suggest'

// Set VITE_API_URL (e.g. in web/frontend/.env.local) to use a server:
//   VITE_API_URL=http://localhost:8000          -> Python mock server
//   VITE_API_URL=https://<you>-tts.hf.space     -> real model (later)
// Leave it unset to use the in-browser mock.
const API_URL = import.meta.env.VITE_API_URL as string | undefined

export const backendName = API_URL ? new URL(API_URL).host : 'in-browser mock'
export const isMock = !API_URL || /localhost|127\.0\.0\.1/.test(API_URL)
export const generate: Generator = API_URL ? sseGenerator(API_URL) : mockGenerate
/** Matching by meaning needs the server; the in-browser mock only has the local typo pass. */
export const suggest: ((q: string, signal?: AbortSignal) => Promise<SuggestResult>) | null = API_URL ? suggester(API_URL) : null
export const wakeServer = () => API_URL && wake(API_URL)

export type * from './events'
export type { Suggestion, SuggestResult } from './suggest'
