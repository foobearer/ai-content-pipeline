"""
providers/google_provider.py — Google Cloud Implementation
────────────────────────────────────────────────────────────
Uses three Google Cloud APIs:
  - Cloud Vision API     → Image labels, OCR, object localisation, safe search
  - Cloud Speech-to-Text → Video/audio transcription
  - Cloud Natural Language API → Sentiment, entities, syntax analysis

Setup:
  1. Create a Google Cloud project at https://console.cloud.google.com
  2. Enable: Vision API, Speech-to-Text API, Natural Language API
  3. Create a service account key (JSON file)
  4. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json in .env

Full guide: docs/SETUP.md#google-cloud

Free tier (generous):
  - Vision API:    1,000 images/month free
  - Speech API:    60 minutes/month free
  - NL API:        5,000 units/month free
"""

import asyncio
import time
from pathlib import Path

from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, DetectedObject, BoundingBox, Entity, Sentiment, TranscriptSegment
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class GoogleProvider(BaseProvider):
    """
    AI analysis using Google Cloud Vision, Speech-to-Text, and Natural Language APIs.

    Google's specialised APIs are often more accurate for specific tasks
    (e.g. document OCR, logo detection) than general-purpose LLMs.
    """

    def __init__(self):
        if not settings.google_available:
            raise ProviderError(
                "google",
                "No credentials configured. Set GOOGLE_APPLICATION_CREDENTIALS in .env. "
                "See docs/SETUP.md#google-cloud for setup instructions."
            )
        # Import here — these are optional dependencies
        # If google-cloud packages aren't installed, we get a clear ImportError
        try:
            from google.cloud import vision, speech, language_v1
            self._vision = vision.ImageAnnotatorClient()
            self._speech = speech.SpeechClient()
            self._language = language_v1.LanguageServiceClient()
            self._vision_types = vision
            self._speech_types = speech
            self._language_types = language_v1
        except ImportError as e:
            raise ProviderError("google", f"Google Cloud libraries not installed: {e}") from e

    @property
    def name(self) -> str:
        return "google"

    @property
    def model_info(self) -> str:
        return "Cloud Vision API + Speech-to-Text v2 + Natural Language API"

    # ── Image Analysis ────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """
        Send image to Google Cloud Vision API.

        We request multiple feature types in a single API call to minimise
        round trips and cost:
          - LABEL_DETECTION      → general scene labels
          - OBJECT_LOCALIZATION  → objects with bounding boxes
          - TEXT_DETECTION       → OCR
          - IMAGE_PROPERTIES     → dominant colours
          - SAFE_SEARCH_DETECTION→ content safety check
        """
        start = time.monotonic()
        log.info("google.image.start", path=str(image_path))

        try:
            image_bytes = image_path.read_bytes()
            image = self._vision_types.Image(content=image_bytes)

            # Request all features in one API call (one request = one unit of quota)
            features = [
                self._vision_types.Feature(type_=self._vision_types.Feature.Type.LABEL_DETECTION, max_results=15),
                self._vision_types.Feature(type_=self._vision_types.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
                self._vision_types.Feature(type_=self._vision_types.Feature.Type.TEXT_DETECTION),
                self._vision_types.Feature(type_=self._vision_types.Feature.Type.IMAGE_PROPERTIES),
                self._vision_types.Feature(type_=self._vision_types.Feature.Type.SAFE_SEARCH_DETECTION),
            ]

            # Run in a thread pool — the Google SDK is synchronous
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._vision.annotate_image({"image": image, "features": features})
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("google.image.done", elapsed_ms=elapsed_ms)

            # ── Parse labels ──────────────────────────────────────────────
            labels = [
                Label(name=l.description, confidence=round(l.score, 3))
                for l in response.label_annotations
            ]

            # ── Parse detected objects with bounding boxes ─────────────────
            objects = []
            for obj in response.localized_object_annotations:
                # Google returns normalised coordinates (0.0–1.0)
                # We store them as-is; the frontend can scale to actual pixels
                vertices = obj.bounding_poly.normalized_vertices
                if vertices:
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    bbox = BoundingBox(
                        x=int(min(x_coords) * 1000),   # Store as per-mille for JSON
                        y=int(min(y_coords) * 1000),
                        width=int((max(x_coords) - min(x_coords)) * 1000),
                        height=int((max(y_coords) - min(y_coords)) * 1000),
                    )
                else:
                    bbox = None

                objects.append(DetectedObject(
                    name=obj.name,
                    confidence=round(obj.score, 3),
                    bounding_box=bbox,
                ))

            # ── Parse OCR text ────────────────────────────────────────────
            extracted_text = None
            if response.text_annotations:
                # First annotation is the full text; rest are individual words
                extracted_text = response.text_annotations[0].description.strip()

            # ── Parse dominant colours ─────────────────────────────────────
            dominant_colours = []
            if response.image_properties_annotation:
                colours = response.image_properties_annotation.dominant_colors.colors
                for colour in sorted(colours, key=lambda c: c.pixel_fraction, reverse=True)[:5]:
                    r, g, b = int(colour.color.red), int(colour.color.green), int(colour.color.blue)
                    dominant_colours.append(f"#{r:02x}{g:02x}{b:02x}")

            # ── Parse safe search ──────────────────────────────────────────
            safe = response.safe_search_annotation
            # VERY_UNLIKELY=1, UNLIKELY=2, POSSIBLE=3, LIKELY=4, VERY_LIKELY=5
            is_safe = all(
                getattr(safe, field) <= 3  # Allow up to POSSIBLE for safety flags
                for field in ["adult", "violence", "racy"]
            )

            # Build searchable tags from label names
            tags = [l.name for l in response.label_annotations if l.score > 0.7]

            return ImageAnalysis(
                labels=labels,
                objects=objects,
                extracted_text=extracted_text or None,
                dominant_colours=dominant_colours,
                tags=tags,
                is_safe=is_safe,
                description=None,  # Vision API doesn't generate descriptions — use labels
            )

        except Exception as e:
            log.error("google.image.failed", error=str(e))
            raise ProviderError("google", str(e)) from e

    # ── Video Analysis ────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        """
        Transcribe video using Google Speech-to-Text, then analyse with NL API.

        For longer videos (>60s), use Google's LongRunningRecognize which
        processes asynchronously and supports files up to 480 minutes.
        For shorter clips, we use synchronous Recognize.
        """
        start = time.monotonic()
        log.info("google.video.start", path=str(video_path))

        try:
            # Step 1: Extract audio
            audio_path = await self._extract_audio(video_path)

            try:
                # Step 2: Transcribe
                transcript_text, segments = await self._transcribe(audio_path)

                # Step 3: Analyse transcript text
                text_analysis = await self.analyse_text(transcript_text) if transcript_text else None

                elapsed_ms = int((time.monotonic() - start) * 1000)
                log.info("google.video.done", elapsed_ms=elapsed_ms)

                return VideoAnalysis(
                    transcript=transcript_text,
                    transcript_segments=segments,
                    summary=text_analysis.summary if text_analysis else None,
                    detected_language=text_analysis.detected_language if text_analysis else None,
                    tags=text_analysis.keywords[:10] if text_analysis else [],
                    scene_labels=[],
                )
            finally:
                if audio_path.exists():
                    audio_path.unlink()

        except Exception as e:
            log.error("google.video.failed", error=str(e))
            raise ProviderError("google", str(e)) from e

    async def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from video file using moviepy."""
        audio_path = video_path.with_suffix(".wav")
        try:
            from moviepy.editor import VideoFileClip
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: VideoFileClip(str(video_path)).audio.write_audiofile(
                    str(audio_path), fps=16000, logger=None  # 16kHz is optimal for Speech API
                )
            )
        except Exception as e:
            raise ProviderError("google", f"Audio extraction failed: {e}") from e
        return audio_path

    async def _transcribe(self, audio_path: Path) -> tuple[str, list[TranscriptSegment]]:
        """Send audio to Google Speech-to-Text API."""
        audio_bytes = audio_path.read_bytes()
        audio = self._speech_types.RecognitionAudio(content=audio_bytes)
        config = self._speech_types.RecognitionConfig(
            encoding=self._speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_time_offsets=True,      # Get word-level timestamps
            enable_automatic_punctuation=True,
            model="video",                       # Optimised model for video audio
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._speech.recognize(config=config, audio=audio)
        )

        full_text = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )

        # Build segments from results (Google Speech returns per-phrase segments)
        segments = []
        for result in response.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            if alt.words:
                start_s = alt.words[0].start_time.total_seconds()
                end_s = alt.words[-1].end_time.total_seconds()
                segments.append(TranscriptSegment(
                    start_seconds=round(start_s, 2),
                    end_seconds=round(end_s, 2),
                    text=alt.transcript.strip(),
                    confidence=round(alt.confidence, 3),
                ))

        return full_text, segments

    # ── Text Analysis ─────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        """
        Analyse text using Google Natural Language API.
        Highly accurate for entity extraction and sentiment analysis.
        """
        start = time.monotonic()
        log.info("google.text.start", word_count=len(text.split()))

        try:
            document = self._language_types.Document(
                content=text[:5000],  # API limit is much higher but we cap for cost
                type_=self._language_types.Document.Type.PLAIN_TEXT,
            )

            # Run entity, sentiment, and syntax analysis concurrently
            entity_task = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._language.analyze_entities(request={"document": document})
            )
            sentiment_task = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._language.analyze_sentiment(request={"document": document})
            )

            entity_response, sentiment_response = await asyncio.gather(entity_task, sentiment_task)

            # ── Parse entities ─────────────────────────────────────────────
            entity_type_map = {
                0: "OTHER", 1: "PERSON", 2: "LOCATION", 3: "ORGANIZATION",
                4: "EVENT", 5: "WORK_OF_ART", 6: "CONSUMER_GOOD", 7: "OTHER",
            }
            entities = [
                Entity(
                    text=e.name,
                    type=entity_type_map.get(e.type_, "OTHER"),
                    confidence=round(e.salience, 3),
                )
                for e in entity_response.entities[:15]
                if e.salience > 0.01
            ]

            # ── Parse sentiment ────────────────────────────────────────────
            score = sentiment_response.document_sentiment.score
            if score > 0.25:
                sentiment = Sentiment.POSITIVE
            elif score < -0.25:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL

            # Keywords = top entity names by salience
            keywords = [e.name for e in entity_response.entities[:10]]

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("google.text.done", elapsed_ms=elapsed_ms)

            return TextAnalysis(
                summary=None,           # NL API doesn't summarise — would need Vertex AI
                sentiment=sentiment,
                sentiment_score=round(score, 3),
                entities=entities,
                keywords=keywords,
                detected_language=entity_response.language,
                word_count=len(text.split()),
                topics=[],
            )

        except Exception as e:
            log.error("google.text.failed", error=str(e))
            raise ProviderError("google", str(e)) from e
