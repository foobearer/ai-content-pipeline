/**
 * types/api.ts — TypeScript types for the backend API
 * ──────────────────────────────────────────────────────
 * These mirror the Pydantic schemas in backend/src/models/schemas.py.
 * If you update a schema on the backend, update the matching type here.
 *
 * Keeping them in sync manually is fine for a project this size.
 * For larger projects, consider using openapi-typescript to auto-generate these.
 */

export type Provider = 'openai' | 'google' | 'huggingface'
export type ContentType = 'image' | 'video' | 'text' | 'document'
export type Sentiment = 'positive' | 'negative' | 'neutral' | 'mixed'

// ── Sub-types ────────────────────────────────────────────────────────────────

export interface Label {
  name: string
  confidence: number  // 0.0 – 1.0
}

export interface Entity {
  text: string
  type: 'PERSON' | 'LOCATION' | 'ORGANIZATION' | 'DATE' | 'OTHER'
  confidence: number
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface DetectedObject {
  name: string
  confidence: number
  bounding_box: BoundingBox | null
}

export interface TranscriptSegment {
  start_seconds: number
  end_seconds: number
  text: string
  confidence: number | null
}

export interface ProviderInfo {
  name: Provider
  model_used: string | null
  processing_time_ms: number
}

// ── Analysis results ─────────────────────────────────────────────────────────

export interface ImageAnalysis {
  labels: Label[]
  objects: DetectedObject[]
  extracted_text: string | null
  dominant_colours: string[]   // hex strings e.g. "#3a3a3a"
  description: string | null
  tags: string[]
  is_safe: boolean | null
}

export interface VideoAnalysis {
  transcript: string | null
  transcript_segments: TranscriptSegment[]
  duration_seconds: number | null
  detected_language: string | null
  scene_labels: Label[]
  tags: string[]
  summary: string | null
}

export interface TextAnalysis {
  summary: string | null
  sentiment: Sentiment | null
  sentiment_score: number | null   // -1.0 to 1.0
  entities: Entity[]
  keywords: string[]
  detected_language: string | null
  word_count: number | null
  topics: string[]
}

// ── Top-level response ────────────────────────────────────────────────────────

export interface AnalysisResult {
  job_id: string
  content_type: ContentType
  filename: string | null
  provider: ProviderInfo
  image_analysis: ImageAnalysis | null
  video_analysis: VideoAnalysis | null
  text_analysis: TextAnalysis | null
  error: string | null
}

// ── Provider metadata (from GET /providers) ───────────────────────────────────

export interface ProviderMeta {
  id: Provider
  name: string
  configured: boolean
  models: string
  requires_key: boolean
  capabilities: ContentType[]
  key_url?: string
  note?: string
}
