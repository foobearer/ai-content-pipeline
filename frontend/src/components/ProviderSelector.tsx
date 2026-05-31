/**
 * components/ProviderSelector.tsx
 * ─────────────────────────────────
 * Displays the three provider cards and lets the user choose one.
 * Each card shows the provider name, models used, and whether an API key is needed.
 */

import type { Provider } from '../types/api'

interface ProviderConfig {
  id: Provider
  name: string
  models: string
  note: string
  accentClass: string       // Tailwind text colour class
  borderClass: string       // Tailwind border colour class
}

// Static config — keeps all provider metadata in one place
const PROVIDERS: ProviderConfig[] = [
  {
    id: 'huggingface',
    name: 'HuggingFace',
    models: 'BLIP · Whisper-base · BART · RoBERTa · BERT-NER',
    note: 'Local · No API key needed · Free · Private',
    accentClass: 'text-accent',
    borderClass: 'border-accent',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    models: 'GPT-4o Vision · Whisper-1 · GPT-4o-mini',
    note: 'Cloud · API key required · $5 free credit on signup',
    accentClass: 'text-accent2',
    borderClass: 'border-accent2',
  },
  {
    id: 'google',
    name: 'Google Cloud',
    models: 'Vision API · Speech-to-Text · Natural Language API',
    note: 'Cloud · Service account credentials required',
    accentClass: 'text-accent3',
    borderClass: 'border-accent3',
  },
]

interface Props {
  selected: Provider
  onChange: (provider: Provider) => void
}

export function ProviderSelector({ selected, onChange }: Props) {
  return (
    <div>
      <span className="section-label">01 // Select AI Provider</span>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
        {PROVIDERS.map((p) => {
          const isSelected = selected === p.id
          return (
            <button
              key={p.id}
              onClick={() => onChange(p.id)}
              className={[
                'text-left p-4 bg-surface border transition-colors duration-200',
                isSelected ? `${p.borderClass} bg-surface2` : 'border-border hover:border-muted',
              ].join(' ')}
              aria-pressed={isSelected}
            >
              {/* Provider name */}
              <p className={`font-mono font-bold text-sm mb-1 ${p.accentClass}`}>
                {p.name}
              </p>

              {/* Models in use */}
              <p className="font-mono text-xs text-muted mb-2 leading-relaxed">
                {p.models}
              </p>

              {/* Key / cost note */}
              <p className="font-mono text-xs text-muted opacity-60">
                {p.note}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/**
 * Helper: given a Provider id, return its accent colour class.
 * Used by other components to colour-match the selected provider.
 */
export function getProviderAccent(provider: Provider): string {
  return PROVIDERS.find((p) => p.id === provider)?.accentClass ?? 'text-accent'
}

export function getProviderBorder(provider: Provider): string {
  return PROVIDERS.find((p) => p.id === provider)?.borderClass ?? 'border-accent'
}
