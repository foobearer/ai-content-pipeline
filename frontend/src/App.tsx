/**
 * App.tsx — Root Component
 * ──────────────────────────
 * Composes all top-level pieces:
 *   - Header
 *   - ProviderSelector
 *   - DropZone
 *   - AnalyseButton
 *   - ResultsPanel
 *
 * State is managed entirely by the useAnalysis() hook.
 * This component only handles layout — no business logic lives here.
 */

import { useState } from 'react'
import type { Provider } from './types/api'
import { useAnalysis } from './hooks/useAnalysis'
import { ProviderSelector } from './components/ProviderSelector'
import { DropZone } from './components/DropZone'
import { AnalyseButton } from './components/AnalyseButton'
import { ResultsPanel } from './components/ResultsPanel'

export default function App() {
  const [provider, setProvider] = useState<Provider>('huggingface')
  const { status, result, errorMessage, file, setFile, analyse, reset } = useAnalysis()

  const isLoading = status === 'analysing'

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">

      {/* Header */}
      <header className="border-b border-border pb-8 mb-8">
        <h1 className="font-sans text-4xl font-extrabold tracking-tight mb-2">
          AI Content <span className="text-accent">Analysis</span> Pipeline
        </h1>
        <p className="font-mono text-xs text-muted">
          Multi-modal · Image · Video · Text · Documents · Provider Switcher
        </p>
        <p className="font-mono text-xs text-muted mt-1">
          Built by{' '}
          <a href="https://joycee.dev" className="text-accent2 hover:underline" target="_blank" rel="noopener noreferrer">
            Joycee Catamora Paragas
          </a>
          {' · '}
          <a href="/docs" className="text-muted hover:text-white transition-colors">
            API Docs →
          </a>
        </p>
      </header>

      {/* Provider selection */}
      <ProviderSelector selected={provider} onChange={setProvider} />

      {/* File upload */}
      <DropZone file={file} onFileSelect={setFile} />

      {/* Submit + error */}
      <AnalyseButton
        provider={provider}
        disabled={!file}
        isLoading={isLoading}
        onClick={() => analyse(provider)}
        errorMessage={errorMessage}
      />

      {/* Results */}
      {result && (
        <>
          <ResultsPanel result={result} provider={provider} />

          {/* Reset button */}
          <div className="mt-8 text-center">
            <button
              onClick={reset}
              className="font-mono text-xs text-muted hover:text-white border border-border
                         hover:border-muted px-6 py-2 transition-colors duration-200"
            >
              ↩ Analyse another file
            </button>
          </div>
        </>
      )}
    </div>
  )
}
