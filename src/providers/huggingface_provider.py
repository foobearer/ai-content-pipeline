"""
providers/huggingface_provider.py — HuggingFace Local Models
──────────────────────────────────────────────────────────────
Runs entirely on your local machine — no API key, no cost, no data leaving
your computer. Perfect for development, demos, and privacy-sensitive content.

Models used (downloaded automatically on first run, ~2GB total):
  - Salesforce/blip-image-captioning-base  → Image captions (~900MB)
  - openai/whisper-base                    → Audio transcription (~150MB)
  - facebook/bart-large-cnn                → Text summarisation (~1.6GB)
  - cardiffnlp/twitter-roberta-base-sentiment-latest → Sentiment (~500MB)
  - dslim/bert-base-NER                    → Named entity recognition (~400MB)

Trade-offs vs cloud providers:
  ✅ Free, private, works offline
  ⚠️  Slower (especially on CPU — GPU makes a big difference)
  ⚠️  Less accurate than GPT-4o or Google's specialised APIs
  ⚠️  First run downloads ~2GB of model weights

Tip: Set HF_HOME in .env to a fast SSD path for better performance.
"""

import asyncio
import time
from pathlib import Path
from functools import lru_cache

from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, Entity, Sentiment, TranscriptSegment
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class HuggingFaceProvider(BaseProvider):
    """
    AI analysis using local HuggingFace Transformers models.

    Models are loaded lazily — only downloaded/loaded when first used.
    This keeps startup time fast even if a model hasn't been used yet.
    """

    def __init__(self):
        # No credentials needed — just confirm transformers is installed
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ProviderError(
                "huggingface",
                "transformers library not installed. Run: pip install transformers torch"
            )
        self._pipelines: dict = {}  # Lazy cache for loaded pipelines

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_info(self) -> str:
        return "BLIP + Whisper-base + BART-large-CNN + RoBERTa-sentiment + BERT-NER"

    # ── Lazy pipeline loader ──────────────────────────────────────────────────

    async def _get_pipeline(self, task: str, model: str):
        """
        Load a HuggingFace pipeline, caching it after the first load.

        Loading a model takes 2–10 seconds the first time.
        Subsequent calls for the same model return instantly from cache.
        We run loading in a thread pool because it's CPU-bound.
        """
        if task not in self._pipelines:
            log.info("hf.model.loading", task=task, model=model)
            from transformers import pipeline

            # run_in_executor offloads the blocking model load to a thread
            # so we don't block the async event loop
            self._pipelines[task] = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline(task, model=model, token=settings.hf_token or None)
            )
            log.info("hf.model.loaded", task=task)

        return self._pipelines[task]

    # ── Image Analysis ────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """
        Analyse an image using BLIP for captioning and ViT for classification.

        BLIP (Bootstrapping Language-Image Pre-training) can generate natural
        language captions AND answer questions about image content.
        """
        start = time.monotonic()
        log.info("hf.image.start", path=str(image_path))

        try:
            from PIL import Image as PILImage

            image = PILImage.open(image_path).convert("RGB")

            # Load BLIP captioning pipeline
            captioner = await self._get_pipeline(
                "image-to-text",
                "Salesforce/blip-image-captioning-base"
            )

            # Load image classification pipeline
            classifier = await self._get_pipeline(
                "image-classification",
                "google/vit-base-patch16-224"
            )

            # Run both pipelines concurrently
            caption_result, class_result = await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(None, lambda: captioner(image)),
                asyncio.get_event_loop().run_in_executor(None, lambda: classifier(image)),
            )

            # Try OCR if pytesseract is installed (optional)
            extracted_text = await self._ocr(image) if self._pytesseract_available() else None

            # Extract dominant colours using PIL (no model needed)
            dominant_colours = self._extract_colours(image)

            description = caption_result[0]["generated_text"] if caption_result else None

            labels = [
                Label(name=r["label"].replace("_", " "), confidence=round(r["score"], 3))
                for r in (class_result or [])[:10]
            ]

            tags = [l.name for l in labels if l.confidence > 0.1]
            if description:
                # Extract nouns from description as additional tags
                tags += [w for w in description.split() if len(w) > 4 and w.isalpha()]

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("hf.image.done", elapsed_ms=elapsed_ms)

            return ImageAnalysis(
                labels=labels,
                objects=[],             # Object detection requires a separate model (DETR)
                extracted_text=extracted_text,
                dominant_colours=dominant_colours,
                description=description,
                tags=list(set(tags))[:15],
                is_safe=None,           # Content safety requires a specialised classifier
            )

        except Exception as e:
            log.error("hf.image.failed", error=str(e))
            raise ProviderError("huggingface", str(e)) from e

    async def _ocr(self, image) -> str | None:
        """Run Tesseract OCR on image. Returns None if no text found."""
        try:
            import pytesseract
            text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image).strip()
            )
            return text if len(text) > 3 else None
        except Exception:
            return None

    @staticmethod
    def _pytesseract_available() -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _extract_colours(image) -> list[str]:
        """
        Extract dominant colours from an image using PIL quantization.
        No ML model needed — just colour-space analysis.
        """
        try:
            # Resize for speed, quantize to find dominant palette
            small = image.resize((150, 150))
            quantized = small.quantize(colors=5)
            palette = quantized.getpalette()[:15]  # 5 colours × 3 channels (RGB)

            colours = []
            for i in range(0, 15, 3):
                r, g, b = palette[i], palette[i+1], palette[i+2]
                colours.append(f"#{r:02x}{g:02x}{b:02x}")
            return colours
        except Exception:
            return []

    # ── Video Analysis ────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        """
        Transcribe video using local Whisper model, then analyse transcript.

        Whisper runs entirely locally — audio never leaves your machine.
        The 'base' model balances speed vs. accuracy; 'small' or 'medium'
        are more accurate but slower.
        """
        start = time.monotonic()
        log.info("hf.video.start", path=str(video_path))

        try:
            # Extract audio
            audio_path = video_path.with_suffix(".wav")
            try:
                from moviepy.editor import VideoFileClip
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: VideoFileClip(str(video_path)).audio.write_audiofile(
                        str(audio_path), fps=16000, logger=None
                    )
                )
            except Exception as e:
                raise ProviderError("huggingface", f"Audio extraction failed: {e}") from e

            try:
                # Load Whisper pipeline
                asr = await self._get_pipeline(
                    "automatic-speech-recognition",
                    "openai/whisper-base"
                )

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: asr(str(audio_path), return_timestamps=True)
                )

                transcript_text = result.get("text", "")
                chunks = result.get("chunks", [])

                segments = [
                    TranscriptSegment(
                        start_seconds=round(chunk["timestamp"][0] or 0, 2),
                        end_seconds=round(chunk["timestamp"][1] or 0, 2),
                        text=chunk["text"].strip(),
                    )
                    for chunk in chunks
                    if chunk.get("text", "").strip()
                ]

                # Analyse the transcript
                text_analysis = await self.analyse_text(transcript_text) if transcript_text else None

                elapsed_ms = int((time.monotonic() - start) * 1000)
                log.info("hf.video.done", elapsed_ms=elapsed_ms)

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
            log.error("hf.video.failed", error=str(e))
            raise ProviderError("huggingface", str(e)) from e

    # ── Text Analysis ─────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        """
        Analyse text using BART (summarisation), RoBERTa (sentiment), BERT (NER).

        Three separate models are used, each specialised for its task.
        They run sequentially to avoid exceeding RAM on low-spec machines.
        """
        start = time.monotonic()
        log.info("hf.text.start", word_count=len(text.split()))

        try:
            truncated = text[:1024]  # Keep within model token limits

            # ── Summarisation ──────────────────────────────────────────────
            summariser = await self._get_pipeline(
                "summarization",
                "facebook/bart-large-cnn"
            )
            summary_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: summariser(
                    truncated,
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )
            )
            summary = summary_result[0]["summary_text"] if summary_result else None

            # ── Sentiment ──────────────────────────────────────────────────
            sentiment_pipe = await self._get_pipeline(
                "sentiment-analysis",
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            sentiment_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sentiment_pipe(truncated[:512])
            )

            label_map = {"positive": Sentiment.POSITIVE, "negative": Sentiment.NEGATIVE, "neutral": Sentiment.NEUTRAL}
            raw_label = sentiment_result[0]["label"].lower() if sentiment_result else "neutral"
            sentiment = label_map.get(raw_label, Sentiment.NEUTRAL)
            sentiment_score = sentiment_result[0]["score"] if sentiment_result else 0.5
            if sentiment == Sentiment.NEGATIVE:
                sentiment_score = -sentiment_score

            # ── Named Entity Recognition ───────────────────────────────────
            ner_pipe = await self._get_pipeline(
                "ner",
                "dslim/bert-base-NER"
            )
            ner_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ner_pipe(truncated[:512])
            )

            # Merge consecutive tokens that belong to the same entity
            entities = self._merge_ner_tokens(ner_result or [])

            # ── Keywords ───────────────────────────────────────────────────
            # Simple TF-IDF-style keyword extraction without a model
            keywords = self._extract_keywords(text)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("hf.text.done", elapsed_ms=elapsed_ms)

            return TextAnalysis(
                summary=summary,
                sentiment=sentiment,
                sentiment_score=round(sentiment_score, 3),
                entities=entities[:10],
                keywords=keywords[:10],
                detected_language=None,    # Would need a separate langdetect model
                word_count=len(text.split()),
                topics=[],
            )

        except Exception as e:
            log.error("hf.text.failed", error=str(e))
            raise ProviderError("huggingface", str(e)) from e

    @staticmethod
    def _merge_ner_tokens(tokens: list) -> list[Entity]:
        """
        BERT NER returns word-pieces (e.g. ['Jo', '##hn', 'Doe']).
        This merges consecutive tokens that belong to the same entity.
        """
        entities = []
        current_entity = None
        current_text = ""

        for token in tokens:
            word = token["word"].lstrip("##")
            entity_type = token["entity"].replace("B-", "").replace("I-", "")

            if token["entity"].startswith("B-") or current_entity != entity_type:
                if current_entity and current_text:
                    entities.append(Entity(
                        text=current_text,
                        type=current_entity,
                        confidence=round(token["score"], 3)
                    ))
                current_entity = entity_type
                current_text = word
            else:
                current_text += word

        if current_entity and current_text:
            entities.append(Entity(text=current_text, type=current_entity, confidence=0.8))

        return entities

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
        """
        Simple keyword extraction without a model.
        Counts word frequency, filters stop words, returns top N.
        """
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "this", "that", "these", "those", "it", "its", "as", "not", "no",
        }
        words = [
            w.lower().strip(".,!?;:\"'()[]")
            for w in text.split()
            if len(w) > 3
        ]
        freq: dict[str, int] = {}
        for word in words:
            if word not in stop_words and word.isalpha():
                freq[word] = freq.get(word, 0) + 1

        return sorted(freq, key=freq.get, reverse=True)[:top_n]
