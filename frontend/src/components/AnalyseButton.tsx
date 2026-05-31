/**
 * components/AnalyseButton.tsx
 * ──────────────────────────────
 * Submit button with loading, disabled, and error states.
 * Colour-coded to match the selected provider.
 */

import type { Provider } from '../types/api'

const PROVIDER_COLORS: Record<Provider, string> = {
  huggingface: 'bg-accent text-bg hover:bg-accent/90',
  openai:      'bg-accent2 text-white hover:bg-accent2/90',
  google:      'bg-accent3 text-bg hover:bg-accent3/90',
}

interface Props {
  provider: Provider
  disabled: boolean
  isLoading: boolean
  onClick: () => void
  errorMessage: string | null
}

export function AnalyseButton({ provider, disabled, isLoading, onClick, errorMessage }: Props) {
  return (
    <div className="mb-8">
      {/* Error message */}
      {errorMessage && (
        <div className="font-mono text-xs text-red-400 bg-red-400/10 border border-red-400/30 px-4 py-3 mb-4">
          ⚠ {errorMessage}
        </div>
      )}

      <button
        onClick={onClick}
        disabled={disabled || isLoading}
        className={[
          'w-full font-mono text-sm font-bold uppercase tracking-widest py-3 transition-all duration-200',
          disabled || isLoading
            ? 'bg-surface2 text-muted cursor-not-allowed border border-border'
            : PROVIDER_COLORS[provider],
        ].join(' ')}
      >
        {isLoading
          ? '⚙ Analysing...'
          : disabled
            ? 'Select a file to continue'
            : `Analyse with ${provider.charAt(0).toUpperCase() + provider.slice(1)}`}
      </button>

      {/* Hint for slow HuggingFace first-run */}
      {isLoading && provider === 'huggingface' && (
        <p className="font-mono text-xs text-muted text-center mt-2">
          Loading local models — first run may take 30–60 seconds while models download.
        </p>
      )}
    </div>
  )
}
