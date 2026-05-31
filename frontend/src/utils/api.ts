/**
 * utils/api.ts — API fetch wrapper
 * ──────────────────────────────────
 * Centralises all HTTP calls to the FastAPI backend.
 * If the base URL changes (e.g. for deployment), only this file needs updating.
 */

import type { AnalysisResult, Provider } from '../types/api'

// In development, Vite proxies /analyse/* to localhost:8000
// In production, set VITE_API_URL in your .env to point to your deployed backend
const BASE_URL = import.meta.env.VITE_API_URL ?? ''

/**
 * Upload a file for analysis.
 *
 * @param file     - The file to analyse
 * @param provider - Which AI provider to use
 * @returns        Structured analysis result
 */
export async function analyseFile(file: File, provider: Provider): Promise<AnalysisResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('provider', provider)

  const response = await fetch(`${BASE_URL}/analyse/auto`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    // FastAPI returns { detail: "..." } for errors
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail ?? 'Analysis failed')
  }

  return response.json() as Promise<AnalysisResult>
}

/**
 * Check which providers are configured on the backend.
 */
export async function fetchHealth(): Promise<{ providers: Record<Provider, boolean> }> {
  const response = await fetch(`${BASE_URL}/health`)
  if (!response.ok) throw new Error('Backend unreachable')
  return response.json()
}
