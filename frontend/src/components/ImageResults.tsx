/**
 * components/ImageResults.tsx
 * ────────────────────────────
 * Renders all analysis output for image files:
 * description, labels, OCR text, dominant colours, tags.
 */

import type { ImageAnalysis, Provider } from '../types/api'
import { getProviderAccent } from './ProviderSelector'

interface Props {
  analysis: ImageAnalysis
  provider: Provider
}

export function ImageResults({ analysis, provider }: Props) {
  const accentClass = getProviderAccent(provider)

  return (
    <div className="space-y-4">
      {/* Natural language description */}
      {analysis.description && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Description</p>
          <p className="font-mono text-sm text-muted leading-relaxed">{analysis.description}</p>
        </div>
      )}

      {/* Labels with confidence bars */}
      {analysis.labels.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Labels & Confidence
          </p>
          <div className="space-y-2">
            {analysis.labels.map((label) => (
              <div key={label.name}>
                <div className="flex justify-between font-mono text-xs mb-1">
                  <span className="text-white">{label.name}</span>
                  <span className={accentClass}>{(label.confidence * 100).toFixed(1)}%</span>
                </div>
                {/* Confidence bar */}
                <div className="h-0.5 bg-surface2">
                  <div
                    className={`h-0.5 ${accentClass.replace('text-', 'bg-')}`}
                    style={{ width: `${label.confidence * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* OCR extracted text */}
      {analysis.extracted_text && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Extracted Text (OCR)
          </p>
          <pre className="font-mono text-xs text-muted leading-relaxed whitespace-pre-wrap
                          bg-surface2 p-4 max-h-48 overflow-y-auto">
            {analysis.extracted_text}
          </pre>
        </div>
      )}

      {/* Dominant colours */}
      {analysis.dominant_colours.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Dominant Colours
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            {analysis.dominant_colours.map((hex) => (
              <div key={hex} className="flex items-center gap-2">
                <div
                  className="w-6 h-6 border border-border flex-shrink-0"
                  style={{ backgroundColor: hex }}
                  title={hex}
                />
                <span className="font-mono text-xs text-muted">{hex}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Content safety */}
      {analysis.is_safe !== null && (
        <div className="card">
          <div className="flex justify-between font-mono text-xs">
            <span className="text-muted">Content Safety</span>
            <span className={analysis.is_safe ? 'text-accent' : 'text-red-400'}>
              {analysis.is_safe ? '✓ Safe' : '⚠ Flagged'}
            </span>
          </div>
        </div>
      )}

      {/* Searchable tags */}
      {analysis.tags.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Search Tags
          </p>
          <div className="flex flex-wrap gap-2">
            {analysis.tags.map((tag) => (
              <span key={tag} className={`tag border-border ${accentClass}`}>{tag}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
