"""
providers/openai_provider.py — OpenAI Implementation
──────────────────────────────────────────────────────
Uses GPT-4o for image and text analysis, Whisper for audio transcription.

Models used:
  - gpt-4o         → Image description, label extraction, text analysis
  - gpt-4o-mini    → Text-only tasks (cheaper, faster)
  - whisper-1      → Speech-to-text for video audio

Pricing (as of 2025, pay-per-use):
  - GPT-4o:        $5 / 1M input tokens
  - GPT-4o-mini:   $0.15 / 1M input tokens
  - Whisper:       $0.006 / minute of audio

Free tier: New OpenAI accounts get $5 credit — enough for hundreds of analyses.
Get your key at: https://platform.openai.com/api-keys
"""

import asyncio
import base64
import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, DetectedObject, Entity, Sentiment, TranscriptSegment
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class OpenAIProvider(BaseProvider):
    """
    AI analysis using OpenAI's GPT-4o vision model and Whisper.

    Initialisation is lazy — the client is only created when first needed,
    so importing this class doesn't fail if no API key is set.
    """

    def __init__(self):
        if not settings.openai_available:
            raise ProviderError("openai", "No API key configured. Set OPENAI_API_KEY in .env")
        # AsyncOpenAI is thread-safe and handles connection pooling internally
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_info(self) -> str:
        return f"{settings.openai_vision_model} + {settings.openai_whisper_model}"

    # ── Image Analysis ────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """
        Send image to GPT-4o Vision and parse the structured response.

        Strategy: We encode the image as base64 and ask GPT-4o to return
        a JSON object with specific fields. Using JSON mode ensures we
        always get parseable output rather than freeform text.
        """
        start = time.monotonic()
        log.info("openai.image.start", path=str(image_path))

        try:
            # Read image and encode as base64 for the API
            image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
            mime_type = self._get_mime_type(image_path)

            response = await self._client.chat.completions.create(
                model=settings.openai_vision_model,
                response_format={"type": "json_object"},  # Forces valid JSON output
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional content analysis system. "
                            "Analyse the provided image and return ONLY a JSON object with these exact keys:\n"
                            "- labels: array of {name, confidence} objects (top 10 scene/content labels)\n"
                            "- objects: array of {name, confidence} objects (detected objects)\n"
                            "- extracted_text: string of any text visible in the image, or null\n"
                            "- description: one sentence describing the image\n"
                            "- tags: array of strings (searchable keywords for a media library)\n"
                            "- is_safe: boolean (false if graphic violence, adult content, or hate speech)\n"
                            "Confidence values must be between 0.0 and 1.0."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                            },
                            {"type": "text", "text": "Analyse this image."}
                        ]
                    }
                ],
                max_tokens=1000,
            )

            # Parse the JSON response
            raw = json.loads(response.choices[0].message.content)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("openai.image.done", elapsed_ms=elapsed_ms)

            return ImageAnalysis(
                labels=[Label(**l) for l in raw.get("labels", [])],
                objects=[DetectedObject(**o) for o in raw.get("objects", [])],
                extracted_text=raw.get("extracted_text"),
                description=raw.get("description"),
                tags=raw.get("tags", []),
                is_safe=raw.get("is_safe", True),
                dominant_colours=[],  # GPT-4o doesn't extract hex colours reliably
            )

        except Exception as e:
            log.error("openai.image.failed", error=str(e))
            raise ProviderError("openai", str(e)) from e

    # ── Video Analysis ────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        """
        Transcribe video audio with Whisper, then analyse content with GPT-4o.

        Two-step process:
        1. Extract audio from video → send to Whisper API for transcription
        2. Send transcript to GPT-4o-mini for summarisation and tagging
        """
        start = time.monotonic()
        log.info("openai.video.start", path=str(video_path))

        try:
            # Step 1: Transcribe audio using Whisper
            transcript_text, segments = await self._transcribe_audio(video_path)

            # Step 2: Analyse the transcript text
            text_analysis = await self.analyse_text(transcript_text) if transcript_text else None

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("openai.video.done", elapsed_ms=elapsed_ms)

            return VideoAnalysis(
                transcript=transcript_text,
                transcript_segments=segments,
                summary=text_analysis.summary if text_analysis else None,
                detected_language=text_analysis.detected_language if text_analysis else None,
                tags=text_analysis.keywords[:10] if text_analysis else [],
                scene_labels=[],  # Would require frame extraction — left as extension point
            )

        except Exception as e:
            log.error("openai.video.failed", error=str(e))
            raise ProviderError("openai", str(e)) from e

    async def _transcribe_audio(self, video_path: Path) -> tuple[str, list[TranscriptSegment]]:
        """
        Extract audio from video and send to Whisper API.
        Returns (full_transcript, list_of_timed_segments).
        """
        # Extract audio to a temporary WAV file using moviepy
        audio_path = video_path.with_suffix(".wav")
        try:
            # Import here to avoid slow startup time if moviepy isn't needed
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))
            clip.audio.write_audiofile(str(audio_path), logger=None)
            clip.close()
        except Exception as e:
            raise ProviderError("openai", f"Could not extract audio from video: {e}") from e

        try:
            with open(audio_path, "rb") as audio_file:
                # verbose_json gives us word-level timestamps
                response = await self._client.audio.transcriptions.create(
                    model=settings.openai_whisper_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            segments = [
                TranscriptSegment(
                    start_seconds=seg.start,
                    end_seconds=seg.end,
                    text=seg.text.strip(),
                )
                for seg in (response.segments or [])
            ]

            return response.text, segments

        finally:
            # Always clean up the temporary audio file
            if audio_path.exists():
                audio_path.unlink()

    # ── Text Analysis ─────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        """
        Analyse text using GPT-4o-mini for cost efficiency.
        Extracts summary, sentiment, named entities, and keywords.
        """
        start = time.monotonic()
        log.info("openai.text.start", word_count=len(text.split()))

        try:
            response = await self._client.chat.completions.create(
                model=settings.openai_text_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text analysis system. Analyse the provided text and return "
                            "ONLY a JSON object with these exact keys:\n"
                            "- summary: 2-3 sentence summary\n"
                            "- sentiment: one of 'positive', 'negative', 'neutral', 'mixed'\n"
                            "- sentiment_score: float from -1.0 (very negative) to 1.0 (very positive)\n"
                            "- entities: array of {text, type, confidence} where type is "
                            "PERSON/LOCATION/ORGANIZATION/DATE/OTHER\n"
                            "- keywords: array of the 10 most important words or phrases\n"
                            "- detected_language: ISO 639-1 language code (e.g. 'en', 'de', 'fr')\n"
                            "- topics: array of up to 5 high-level topic categories"
                        )
                    },
                    {"role": "user", "content": f"Analyse this text:\n\n{text[:4000]}"}
                    # Truncate to 4000 chars to stay within token limits
                ],
                max_tokens=800,
            )

            raw = json.loads(response.choices[0].message.content)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("openai.text.done", elapsed_ms=elapsed_ms)

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
            log.error("openai.text.failed", error=str(e))
            raise ProviderError("openai", str(e)) from e

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_mime_type(path: Path) -> str:
        """Map file extension to MIME type for the API request."""
        return {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")
