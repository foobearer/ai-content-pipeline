"""
providers/google_provider.py — Google Cloud Vision + Speech + NL
─────────────────────────────────────────────────────────────────
APIs used:
  Cloud Vision API         → labels, OCR, object detection, safe search
  Cloud Speech-to-Text     → video/audio transcription
  Cloud Natural Language   → entities, sentiment

Setup guide: docs/SETUP.md
Free tier: 1,000 images/month · 60 min audio/month · 5,000 NL units/month
"""

import asyncio
import time
from pathlib import Path

from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, DetectedObject, BoundingBox, Entity, Sentiment, TranscriptSegment,
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class GoogleProvider(BaseProvider):

    def __init__(self):
        if not settings.google_available:
            raise ProviderError(
                "google",
                "No credentials found. Set GOOGLE_APPLICATION_CREDENTIALS in backend/.env. "
                "See docs/SETUP.md for full setup instructions."
            )
        try:
            from google.cloud import vision, speech, language_v1
            self._vision_client = vision.ImageAnnotatorClient()
            self._speech_client = speech.SpeechClient()
            self._language_client = language_v1.LanguageServiceClient()
            self._vision = vision
            self._speech = speech
            self._language = language_v1
        except ImportError as e:
            raise ProviderError("google", f"Google Cloud libraries not installed: {e}") from e

    @property
    def name(self) -> str:
        return "google"

    @property
    def model_info(self) -> str:
        return "Cloud Vision API + Speech-to-Text + Natural Language API"

    # ── Image ──────────────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """
        Single API call requesting all features at once:
        labels, objects, OCR, colours, and safe search.
        This minimises quota usage (one request = one billing unit).
        """
        start = time.monotonic()
        log.info("google.image.start", file=image_path.name)
        try:
            image = self._vision.Image(content=image_path.read_bytes())
            features = [
                self._vision.Feature(type_=self._vision.Feature.Type.LABEL_DETECTION, max_results=15),
                self._vision.Feature(type_=self._vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
                self._vision.Feature(type_=self._vision.Feature.Type.TEXT_DETECTION),
                self._vision.Feature(type_=self._vision.Feature.Type.IMAGE_PROPERTIES),
                self._vision.Feature(type_=self._vision.Feature.Type.SAFE_SEARCH_DETECTION),
            ]

            # Vision SDK is synchronous — run in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._vision_client.annotate_image({"image": image, "features": features}),
            )

            labels = [Label(name=l.description, confidence=round(l.score, 3)) for l in response.label_annotations]

            objects = []
            for obj in response.localized_object_annotations:
                verts = obj.bounding_poly.normalized_vertices
                bbox = BoundingBox(
                    x=int(min(v.x for v in verts) * 1000),
                    y=int(min(v.y for v in verts) * 1000),
                    width=int((max(v.x for v in verts) - min(v.x for v in verts)) * 1000),
                    height=int((max(v.y for v in verts) - min(v.y for v in verts)) * 1000),
                ) if verts else None
                objects.append(DetectedObject(name=obj.name, confidence=round(obj.score, 3), bounding_box=bbox))

            extracted_text = response.text_annotations[0].description.strip() if response.text_annotations else None

            colours = []
            if response.image_properties_annotation:
                for c in sorted(response.image_properties_annotation.dominant_colors.colors,
                                key=lambda x: x.pixel_fraction, reverse=True)[:5]:
                    r, g, b = int(c.color.red), int(c.color.green), int(c.color.blue)
                    colours.append(f"#{r:02x}{g:02x}{b:02x}")

            safe = response.safe_search_annotation
            is_safe = all(getattr(safe, f) <= 3 for f in ["adult", "violence", "racy"])

            log.info("google.image.done", ms=int((time.monotonic() - start) * 1000))
            return ImageAnalysis(
                labels=labels, objects=objects, extracted_text=extracted_text,
                dominant_colours=colours, tags=[l.name for l in labels if l.confidence > 0.7],
                is_safe=is_safe,
            )
        except Exception as e:
            raise ProviderError("google", str(e)) from e

    # ── Video ──────────────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        start = time.monotonic()
        audio_path = video_path.with_suffix(".wav")
        try:
            from moviepy.editor import VideoFileClip
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: VideoFileClip(str(video_path)).audio.write_audiofile(str(audio_path), fps=16000, logger=None),
            )

            audio_bytes = audio_path.read_bytes()
            audio = self._speech.RecognitionAudio(content=audio_bytes)
            config = self._speech.RecognitionConfig(
                encoding=self._speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="video",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._speech_client.recognize(config=config, audio=audio)
            )

            transcript = " ".join(r.alternatives[0].transcript for r in response.results if r.alternatives)
            segments = [
                TranscriptSegment(
                    start_seconds=round(r.alternatives[0].words[0].start_time.total_seconds(), 2),
                    end_seconds=round(r.alternatives[0].words[-1].end_time.total_seconds(), 2),
                    text=r.alternatives[0].transcript.strip(),
                    confidence=round(r.alternatives[0].confidence, 3),
                )
                for r in response.results
                if r.alternatives and r.alternatives[0].words
            ]

            text_result = await self.analyse_text(transcript) if transcript else None
            log.info("google.video.done", ms=int((time.monotonic() - start) * 1000))

            return VideoAnalysis(
                transcript=transcript, transcript_segments=segments,
                summary=text_result.summary if text_result else None,
                detected_language=text_result.detected_language if text_result else None,
                tags=text_result.keywords[:10] if text_result else [],
            )
        except Exception as e:
            raise ProviderError("google", str(e)) from e
        finally:
            if audio_path.exists():
                audio_path.unlink()

    # ── Text ───────────────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        start = time.monotonic()
        try:
            doc = self._language.Document(
                content=text[:5000], type_=self._language.Document.Type.PLAIN_TEXT
            )
            entity_resp, sentiment_resp = await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._language_client.analyze_entities(request={"document": doc})
                ),
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._language_client.analyze_sentiment(request={"document": doc})
                ),
            )

            type_map = {0: "OTHER", 1: "PERSON", 2: "LOCATION", 3: "ORGANIZATION",
                        4: "EVENT", 5: "WORK_OF_ART", 6: "CONSUMER_GOOD"}
            entities = [
                Entity(text=e.name, type=type_map.get(e.type_, "OTHER"), confidence=round(e.salience, 3))
                for e in entity_resp.entities[:15] if e.salience > 0.01
            ]

            score = sentiment_resp.document_sentiment.score
            sentiment = (Sentiment.POSITIVE if score > 0.25 else
                         Sentiment.NEGATIVE if score < -0.25 else Sentiment.NEUTRAL)

            log.info("google.text.done", ms=int((time.monotonic() - start) * 1000))
            return TextAnalysis(
                sentiment=sentiment, sentiment_score=round(score, 3),
                entities=entities, keywords=[e.name for e in entity_resp.entities[:10]],
                detected_language=entity_resp.language, word_count=len(text.split()),
            )
        except Exception as e:
            raise ProviderError("google", str(e)) from e
