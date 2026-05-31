"""
models/schemas.py — Pydantic Data Models
─────────────────────────────────────────
Single source of truth for all request/response shapes.
FastAPI uses these for automatic validation and Swagger docs generation.
The frontend TypeScript types in frontend/src/types/api.ts mirror these exactly.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class Provider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class ContentType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    DOCUMENT = "document"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# ── Sub-models ─────────────────────────────────────────────────────────────────

class Label(BaseModel):
    """A detected label with a confidence score between 0 and 1."""
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Entity(BaseModel):
    """A named entity found in text (person, place, organisation, etc.)."""
    text: str
    type: str = Field(..., description="PERSON | LOCATION | ORGANIZATION | DATE | OTHER")
    confidence: float = Field(..., ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """Pixel location of a detected object. Origin is top-left."""
    x: int
    y: int
    width: int
    height: int


class DetectedObject(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None


class TranscriptSegment(BaseModel):
    """A timestamped chunk of speech from a video or audio file."""
    start_seconds: float
    end_seconds: float
    text: str
    confidence: Optional[float] = None


class ProviderInfo(BaseModel):
    name: Provider
    model_used: Optional[str] = None
    processing_time_ms: int


# ── Analysis result models ─────────────────────────────────────────────────────

class ImageAnalysis(BaseModel):
    labels: list[Label] = Field(default_factory=list)
    objects: list[DetectedObject] = Field(default_factory=list)
    extracted_text: Optional[str] = None
    dominant_colours: list[str] = Field(default_factory=list, description="Hex colour codes")
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    is_safe: Optional[bool] = None


class VideoAnalysis(BaseModel):
    transcript: Optional[str] = None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    detected_language: Optional[str] = None
    scene_labels: list[Label] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    summary: Optional[str] = None


class TextAnalysis(BaseModel):
    summary: Optional[str] = None
    sentiment: Optional[Sentiment] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    detected_language: Optional[str] = None
    word_count: Optional[int] = None
    topics: list[str] = Field(default_factory=list)


# ── Top-level request / response ───────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """
    Unified response for every /analyse/* endpoint.
    Exactly one of image_analysis, video_analysis, text_analysis will be set.
    """
    job_id: str
    content_type: ContentType
    filename: Optional[str] = None
    provider: ProviderInfo
    image_analysis: Optional[ImageAnalysis] = None
    video_analysis: Optional[VideoAnalysis] = None
    text_analysis: Optional[TextAnalysis] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    providers: dict[str, bool]
    default_provider: Provider
