/**
 * components/MetaCard.tsx
 * ────────────────────────
 * Displays job metadata: provider, model, processing time, etc.
 */

import type { AnalysisResult } from '../types/api'
import { getProviderAccent } from './ProviderSelector'

interface Props { result: AnalysisResult }

export function MetaCard({ result }: Props) {
  const accentClass = getProviderAccent(result.provider.name)
  const rows = [
    ['Job ID',         result.job_id],
    ['File',           result.filename],
    ['Content Type',   result.content_type],
    ['Provider',       result.provider.name],
    ['Model',          result.provider.model_used],
    ['Processing Time',`${result.provider.processing_time_ms} ms`],
  ] as [string, string | null | undefined][]

  return (
    <div className="card mb-5">
      <p className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
        Job Metadata
      </p>
      <div className="space-y-2">
        {rows.filter(([, v]) => v).map(([label, value]) => (
          <div key={label} className="flex justify-between font-mono text-xs">
            <span className="text-muted">{label}</span>
            <span className={accentClass}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
