"""
providers/openai_provider.py — OpenAI GPT-4o + Whisper
────────────────────────────────────────────────────────
Models:
  gpt-4o       → image description + structured JSON extraction
  gpt-4o-mini  → text analysis (cheaper, fast)
  whisper-1    → audio/video speech-to-text

Get a key at: https://platform.openai.com/api-keys
New accounts receive $5 free credit.
"""

import base64
import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, DetectedObject, Entity, Sentiment, TranscriptSegment,
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class OpenAIProvider(BaseProvider):

    def __init__(self):
        if not settings.openai_available:
            raise ProviderError("openai", "No API key found. Set OPENAI_API_KEY in backend/.env")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_info(self) -> str:
        return f"{settings.openai_vision_model} + {settings.openai_whisper_model}"

    # ── Image ──────────────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        start = time.monotonic()
        log.info("openai.image.start", file=image_path.name)
        try:
            image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp"}.get(image_path.suffix.lower(), "image/jpeg")

            resp = await self._client.chat.completions.create(
                model=settings.openai_vision_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a content analysis system. Analyse the image and return ONLY "
                            "valid JSON with keys: labels (array of {name,confidence}), "
                            "objects (array of {name,confidence}), extracted_text (string or null), "
                            "description (one sentence), tags (array of strings), "
                            "is_safe (boolean). Confidence values 0.0–1.0."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                            {"type": "text", "text": "Analyse this image."},
                        ],
                    },
                ],
                max_tokens=1000,
            )

            raw = json.loads(resp.choices[0].message.content)
            log.info("openai.image.done", ms=int((time.monotonic() - start) * 1000))

            return ImageAnalysis(
                labels=[Label(**l) for l in raw.get("labels", [])],
                objects=[DetectedObject(**o) for o in raw.get("objects", [])],
                extracted_text=raw.get("extracted_text"),
                description=raw.get("description"),
                tags=raw.get("tags", []),
                is_safe=raw.get("is_safe", True),
                dominant_colours=[],
            )
        except Exception as e:
            raise ProviderError("openai", str(e)) from e

    # ── Video ──────────────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        start = time.monotonic()
        log.info("openai.video.start", file=video_path.name)
        audio_path = video_path.with_suffix(".wav")
        try:
            # Extract audio
            from moviepy.editor import VideoFileClip
            VideoFileClip(str(video_path)).audio.write_audiofile(str(audio_path), logger=None)

            # Transcribe with Whisper
            with open(audio_path, "rb") as f:
                whisper_resp = await self._client.audio.transcriptions.create(
                    model=settings.openai_whisper_model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            transcript = whisper_resp.text
            segments = [
                TranscriptSegment(start_seconds=s.start, end_seconds=s.end, text=s.text.strip())
                for s in (whisper_resp.segments or [])
            ]

            # Analyse transcript text
            text_result = await self.analyse_text(transcript) if transcript else None

            log.info("openai.video.done", ms=int((time.monotonic() - start) * 1000))
            return VideoAnalysis(
                transcript=transcript,
                transcript_segments=segments,
                summary=text_result.summary if text_result else None,
                detected_language=text_result.detected_language if text_result else None,
                tags=text_result.keywords[:10] if text_result else [],
            )
        except Exception as e:
            raise ProviderError("openai", str(e)) from e
        finally:
            if audio_path.exists():
                audio_path.unlink()

    # ── Text ───────────────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        start = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=settings.openai_text_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text analysis system. Return ONLY valid JSON with keys: "
                            "summary (2-3 sentences), sentiment (positive/negative/neutral/mixed), "
                            "sentiment_score (float -1.0 to 1.0), "
                            "entities (array of {text,type,confidence} where type is PERSON/LOCATION/ORGANIZATION/DATE/OTHER), "
                            "keywords (array of top 10 strings), detected_language (ISO code), "
                            "topics (array of up to 5 strings)."
                        ),
                    },
                    {"role": "user", "content": f"Analyse:\n\n{text[:4000]}"},
                ],
                max_tokens=800,
            )

            raw = json.loads(resp.choices[0].message.content)
            log.info("openai.text.done", ms=int((time.monotonic() - start) * 1000))

            return TextAnalysis(
                summary=raw.get("summary"),
                sentiment=Sentiment(raw.get("sentiment", "neutral")),
                sentiment_score=raw.get("sentiment_score", 0.0),
                entities=[Entity(**e) for e in raw.get("entities", [])],
                keywords=raw.get("keywords", []),
                detected_language=raw.get("detected_language"),
                word_count=len(text.split()),
                topics=raw.get("topics", []),
            )
        except Exception as e:
            raise ProviderError("openai", str(e)) from e
