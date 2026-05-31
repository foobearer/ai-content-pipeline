import type { Config } from 'tailwindcss'

const config: Config = {
  // Tell Tailwind which files contain class names so it can tree-shake unused styles
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      // Custom design tokens matching the portfolio dark theme
      colors: {
        bg:       '#0a0a0f',
        surface:  '#111118',
        surface2: '#1a1a24',
        border:   '#2a2a3a',
        accent:   '#00ff9d',
        accent2:  '#7c6af7',
        accent3:  '#ff9d00',
        muted:    '#7a7a9a',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Syne', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

export default config
