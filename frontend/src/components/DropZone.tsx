/**
 * components/DropZone.tsx
 * ─────────────────────────
 * Drag-and-drop file upload zone.
 * Accepts images, videos, PDFs, Word docs, and text files.
 *
 * Accessibility: keyboard-navigable via Enter/Space, labelled for screen readers.
 */

import { useCallback, useRef, useState } from 'react'

const ACCEPTED = [
  'image/jpeg', 'image/png', 'image/gif', 'image/webp',
  'video/mp4', 'video/quicktime', 'video/webm', 'video/x-msvideo',
  'text/plain', 'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
].join(',')

interface Props {
  file: File | null
  onFileSelect: (file: File) => void
}

export function DropZone({ file, onFileSelect }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) onFileSelect(dropped)
  }, [onFileSelect])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) onFileSelect(selected)
    // Reset input so the same file can be re-selected after a reset
    e.target.value = ''
  }, [onFileSelect])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
  }, [])

  return (
    <div>
      <span className="section-label">02 // Upload File</span>

      {/* Drop target */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload file for analysis"
        onClick={() => inputRef.current?.click()}
        onKeyDown={handleKeyDown}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={[
          'border-2 border-dashed p-12 text-center cursor-pointer transition-colors duration-200 mb-6',
          isDragging ? 'border-accent bg-accent/5' : 'border-border hover:border-muted',
        ].join(' ')}
      >
        {/* Icon */}
        <div className="text-4xl mb-4" aria-hidden>
          {file ? '📄' : '⬆'}
        </div>

        {/* File name or prompt */}
        {file ? (
          <div>
            <p className="font-mono text-sm text-white mb-1">{file.name}</p>
            <p className="font-mono text-xs text-muted">
              {(file.size / 1024 / 1024).toFixed(2)} MB · Click or drop to replace
            </p>
          </div>
        ) : (
          <div>
            <p className="font-mono text-sm text-muted mb-2">
              Drop a file here, or click to browse
            </p>
            <p className="font-mono text-xs text-muted opacity-60">
              Images · Videos · PDFs · Word docs · Text files · Max 50 MB
            </p>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={handleChange}
        className="hidden"
        aria-hidden
      />
    </div>
  )
}
