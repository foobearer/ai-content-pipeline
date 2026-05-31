"""
schemas.py — Data Models for the AI Content Analysis Pipeline
─────────────────────────────────────────────────────────────
All request and response shapes are defined here using Pydantic.

Why Pydantic?
  - Automatic validation: bad data is rejected before it reaches our logic
  - Auto-generated API docs (FastAPI uses these for Swagger UI)
  - Type safety: Python type hints that are actually enforced at runtime

Usage:
  from src.models.schemas import AnalysisResult, ImageAnalysis
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums — fixed sets of valid values
# ─────────────────────────────────────────────────────────────────────────────

class Provider(str, Enum):
    """The three AI providers supported by this pipeline."""
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class ContentType(str, Enum):
    """Types of content the pipeline can analyse."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    DOCUMENT = "document"


class Sentiment(str, Enum):
    """Sentiment classification output."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models — building blocks used inside the main result models
# ─────────────────────────────────────────────────────────────────────────────

class Label(BaseModel):
    """A single detected label with a confidence score."""
    name: str = Field(..., description="Human-readable label e.g. 'dog', 'outdoor scene'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")


class Entity(BaseModel):
    """A named entity extracted from text (person, place, organisation, etc.)"""
    text: str = Field(..., description="The entity text as it appeared")
    type: str = Field(..., description="Entity type: PERSON, LOCATION, ORGANIZATION, etc.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """
    Pixel coordinates of a detected object within an image.
    Origin (0,0) is top-left corner.
    """
    x: int = Field(..., description="Left edge pixel")
    y: int = Field(..., description="Top edge pixel")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


class DetectedObject(BaseModel):
    """An object detected within an image, with its location."""
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None


class TranscriptSegment(BaseModel):
    """A timestamped chunk of transcribed speech from a video."""
    start_seconds: float
    end_seconds: float
    text: str
    confidence: Optional[float] = None


class ProviderInfo(BaseModel):
    """Metadata about which provider processed this request."""
    name: Provider
    model_used: Optional[str] = Field(None, description="Specific model version e.g. gpt-4o")
    processing_time_ms: int = Field(..., description="How long the analysis took in milliseconds")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis result models — one per content type
# ─────────────────────────────────────────────────────────────────────────────

class ImageAnalysis(BaseModel):
    """
    Full analysis result for an image.

    Example response:
    {
        "labels": [{"name": "cat", "confidence": 0.98}],
        "objects": [{"name": "cat", "confidence": 0.95, "bounding_box": {...}}],
        "extracted_text": "SALE 50% OFF",
        "dominant_colours": ["#3a3a3a", "#ffffff"],
        "description": "A fluffy orange cat sitting on a windowsill",
        "tags": ["animal", "indoor", "pet", "cat"]
    }
    """
    labels: list[Label] = Field(default_factory=list, description="Scene/content labels")
    objects: list[DetectedObject] = Field(default_factory=list, description="Detected objects with locations")
    extracted_text: Optional[str] = Field(None, description="Any text visible in the image (OCR)")
    dominant_colours: list[str] = Field(default_factory=list, description="Hex colour codes of dominant colours")
    description: Optional[str] = Field(None, description="Natural language description of the image")
    tags: list[str] = Field(default_factory=list, description="Searchable tags for indexing")
    is_safe: Optional[bool] = Field(None, description="Whether the image passes safe-content checks")


class VideoAnalysis(BaseModel):
    """
    Full analysis result for a video.

    Includes both visual analysis (from key frames) and
    audio analysis (from the extracted soundtrack).
    """
    transcript: Optional[str] = Field(None, description="Full transcript of all speech in the video")
    transcript_segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Timestamped transcript — useful for building searchable indexes"
    )
    duration_seconds: Optional[float] = None
    detected_language: Optional[str] = Field(None, description="ISO language code e.g. 'en', 'de'")
    scene_labels: list[Label] = Field(default_factory=list, description="Visual content labels from key frames")
    tags: list[str] = Field(default_factory=list, description="Searchable tags derived from audio + visual")
    summary: Optional[str] = Field(None, description="Brief summary of what the video is about")


class TextAnalysis(BaseModel):
    """
    Full analysis result for text or document content.

    Works on raw text strings, extracted PDF text, or Word document content.
    """
    summary: Optional[str] = Field(None, description="Condensed version of the content")
    sentiment: Optional[Sentiment] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="-1 = very negative, +1 = very positive")
    entities: list[Entity] = Field(default_factory=list, description="Named entities: people, places, organisations")
    keywords: list[str] = Field(default_factory=list, description="Most important words/phrases")
    detected_language: Optional[str] = Field(None, description="ISO language code")
    word_count: Optional[int] = None
    topics: list[str] = Field(default_factory=list, description="High-level topic categories")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level request and response models
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Query parameters for an analysis request."""
    provider: Provider = Field(Provider.HUGGINGFACE, description="Which AI provider to use")
    content_type: Optional[ContentType] = Field(
        None,
        description="Force a content type. Leave empty for auto-detection."
    )


class AnalysisResult(BaseModel):
    """
    The top-level response returned by every /analyse endpoint.

    Always includes provider metadata and the appropriate analysis block.
    Exactly one of image_analysis, video_analysis, or text_analysis will be set.
    """
    job_id: str = Field(..., description="Unique ID — use with GET /results/{job_id}")
    content_type: ContentType
    filename: Optional[str] = None
    provider: ProviderInfo

    # Exactly one of these will be populated depending on content_type
    image_analysis: Optional[ImageAnalysis] = None
    video_analysis: Optional[VideoAnalysis] = None
    text_analysis: Optional[TextAnalysis] = None

    error: Optional[str] = Field(None, description="Set if analysis partially or fully failed")


class HealthResponse(BaseModel):
    """Response from GET /health — shows which providers are ready."""
    status: str = "ok"
    providers: dict[str, bool] = Field(
        ...,
        description="Map of provider name → whether it's configured and ready",
        example={"openai": False, "google": False, "huggingface": True}
    )
    default_provider: Provider
