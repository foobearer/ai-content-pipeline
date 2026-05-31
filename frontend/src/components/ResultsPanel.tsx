/**
 * components/ResultsPanel.tsx
 * ─────────────────────────────
 * Routes the analysis result to the correct display component
 * based on content_type. Also renders the MetaCard at the top.
 */

import type { AnalysisResult, Provider } from '../types/api'
import { MetaCard } from './MetaCard'
import { ImageResults } from './ImageResults'
import { VideoResults } from './VideoResults'
import { TextResults } from './TextResults'

interface Props {
  result: AnalysisResult
  provider: Provider
}

export function ResultsPanel({ result, provider }: Props) {
  return (
    <div>
      <span className="section-label">03 // Analysis Results</span>

      {/* Job metadata always shown at top */}
      <MetaCard result={result} />

      {/* Route to the right result component */}
      {result.image_analysis && (
        <ImageResults analysis={result.image_analysis} provider={provider} />
      )}
      {result.video_analysis && (
        <VideoResults analysis={result.video_analysis} provider={provider} />
      )}
      {result.text_analysis && (
        <TextResults analysis={result.text_analysis} provider={provider} />
      )}

      {/* Catch-all error state */}
      {result.error && (
        <div className="card border-red-400/30 bg-red-400/5">
          <p className="font-mono text-xs text-red-400">⚠ Partial error: {result.error}</p>
        </div>
      )}
    </div>
  )
}
