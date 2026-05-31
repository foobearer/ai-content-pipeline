/**
 * components/TextResults.tsx
 * ───────────────────────────
 * Renders text/document analysis: summary, sentiment, entities, keywords, topics.
 */

import type { TextAnalysis, Provider, Sentiment } from '../types/api'
import { getProviderAccent } from './ProviderSelector'

interface Props {
  analysis: TextAnalysis
  provider: Provider
}

const SENTIMENT_STYLE: Record<Sentiment, string> = {
  positive: 'text-accent border-accent',
  negative: 'text-red-400 border-red-400',
  neutral:  'text-muted border-border',
  mixed:    'text-accent3 border-accent3',
}

export function TextResults({ analysis, provider }: Props) {
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

      {/* Sentiment */}
      {analysis.sentiment && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Sentiment</p>
          <div className="flex items-center gap-4">
            <span className={`font-mono text-xs px-3 py-1 border ${SENTIMENT_STYLE[analysis.sentiment]}`}>
              {analysis.sentiment}
            </span>
            {analysis.sentiment_score !== null && (
              <span className="font-mono text-xs text-muted">
                Score: {analysis.sentiment_score > 0 ? '+' : ''}
                {analysis.sentiment_score.toFixed(2)}
                {' '}(−1.0 very negative · +1.0 very positive)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Named entities */}
      {analysis.entities.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
            Named Entities
          </p>
          <div className="flex flex-wrap gap-2">
            {analysis.entities.map((e, i) => (
              <span key={i} className={`tag ${accentClass}`}>
                <span className="opacity-50 mr-1">[{e.type}]</span>
                {e.text}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Keywords */}
      {analysis.keywords.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Keywords</p>
          <div className="flex flex-wrap gap-2">
            {analysis.keywords.map((kw) => (
              <span key={kw} className="tag">{kw}</span>
            ))}
          </div>
        </div>
      )}

      {/* Topics */}
      {analysis.topics.length > 0 && (
        <div className="card">
          <p className="font-mono text-xs text-accent uppercase tracking-widest mb-3">Topics</p>
          <div className="flex flex-wrap gap-2">
            {analysis.topics.map((t) => (
              <span key={t} className={`tag ${accentClass}`}>{t}</span>
            ))}
          </div>
        </div>
      )}

      {/* Stats row */}
      <div className="card">
        <div className="space-y-2">
          {analysis.word_count && (
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted">Word Count</span>
              <span className={accentClass}>{analysis.word_count.toLocaleString()}</span>
            </div>
          )}
          {analysis.detected_language && (
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted">Language</span>
              <span className={accentClass}>{analysis.detected_language}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
