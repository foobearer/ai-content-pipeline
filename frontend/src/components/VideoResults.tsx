/**
 * components/VideoResults.tsx
 * ────────────────────────────
 * Renders video analysis: summary, full transcript, timestamped segments, tags.
 */

import type { VideoAnalysis, Provider } from '../types/api'
import { getProviderAccent } from './ProviderSelector'

interface Props {
  analysis: VideoAnalysis
  provider: Provider
}

export function VideoResults({ analysis, provider }: Props) {
  const accentClass = getProviderAccent(provider)

  return (
    <div className="space-y-4">
      {/* Summary */}
      {analysis.summary && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Summary</p>
          <p className="font-mono text-sm text-muted leading-relaxed">{analysis.summary}</p>
        </div>
      )}

      {/* Metadata row */}
      <div className="card">
        <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Details</p>
        <div className="space-y-2">
          {analysis.detected_language && (
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted">Language</span>
              <span className={accentClass}>{analysis.detected_language}</span>
            </div>
          )}
          {analysis.duration_seconds && (
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted">Duration</span>
              <span className={accentClass}>{analysis.duration_seconds.toFixed(1)}s</span>
            </div>
          )}
          <div className="flex justify-between font-mono text-xs">
            <span className="text-muted">Segments</span>
            <span className={accentClass}>{analysis.transcript_segments.length}</span>
          </div>
        </div>
      </div>

      {/* Full transcript */}
      {analysis.transcript && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Full Transcript
          </p>
          <pre className="font-mono text-xs text-muted leading-relaxed whitespace-pre-wrap
                          bg-surface2 p-4 max-h-48 overflow-y-auto">
            {analysis.transcript}
          </pre>
        </div>
      )}

      {/* Timestamped segments — great for demonstrating searchable indexing */}
      {analysis.transcript_segments.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Timestamped Segments
          </p>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {analysis.transcript_segments.map((seg, i) => (
              <div key={i} className="flex gap-4 font-mono text-xs">
                <span className={`${accentClass} min-w-[120px] flex-shrink-0`}>
                  {seg.start_seconds.toFixed(1)}s → {seg.end_seconds.toFixed(1)}s
                </span>
                <span className="text-muted leading-relaxed">{seg.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tags */}
      {analysis.tags.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Search Tags</p>
          <div className="flex flex-wrap gap-2">
            {analysis.tags.map((tag) => (
              <span key={tag} className={`tag ${accentClass}`}>{tag}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
