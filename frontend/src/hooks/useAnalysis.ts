/**
 * hooks/useAnalysis.ts — Custom hook for file analysis
 * ───────────────────────────────────────────────────────
 * Encapsulates all state and side effects for the analysis workflow.
 * Components stay clean — they only deal with rendering.
 *
 * State machine:
 *   idle → uploading → analysing → done
 *                               ↘ error
 */

import { useState, useCallback } from 'react'
import { analyseFile } from '../utils/api'
import type { Provider, AnalysisResult } from '../types/api'

type Status = 'idle' | 'analysing' | 'done' | 'error'

interface UseAnalysisReturn {
  /** Current workflow status */
  status: Status
  /** Analysis result — populated when status === 'done' */
  result: AnalysisResult | null
  /** Error message — populated when status === 'error' */
  errorMessage: string | null
  /** Currently selected file */
  file: File | null
  /** Set the file to analyse */
  setFile: (file: File | null) => void
  /** Run the analysis */
  analyse: (provider: Provider) => Promise<void>
  /** Reset back to idle state */
  reset: () => void
}

export function useAnalysis(): UseAnalysisReturn {
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [file, setFileState] = useState<File | null>(null)

  const setFile = useCallback((f: File | null) => {
    setFileState(f)
    // Clear previous results when a new file is selected
    setResult(null)
    setErrorMessage(null)
    setStatus('idle')
  }, [])

  const analyse = useCallback(async (provider: Provider) => {
    if (!file) return

    setStatus('analysing')
    setResult(null)
    setErrorMessage(null)

    try {
      const data = await analyseFile(file, provider)
      setResult(data)
      setStatus('done')
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Something went wrong')
      setStatus('error')
    }
  }, [file])

  const reset = useCallback(() => {
    setStatus('idle')
    setResult(null)
    setErrorMessage(null)
    setFileState(null)
  }, [])

  return { status, result, errorMessage, file, setFile, analyse, reset }
}
