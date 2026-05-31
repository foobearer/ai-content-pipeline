"""
providers/huggingface_provider.py — Local HuggingFace Models
──────────────────────────────────────────────────────────────
Runs entirely on your machine. No API key, no cost, no data sent externally.

Models (auto-downloaded ~2GB on first run):
  Salesforce/blip-image-captioning-base  → image captions
  google/vit-base-patch16-224            → image classification
  openai/whisper-base                    → speech transcription
  facebook/bart-large-cnn                → text summarisation
  cardiffnlp/twitter-roberta-base-sentiment-latest → sentiment
  dslim/bert-base-NER                    → named entity recognition

Trade-offs vs cloud providers:
  ✅ Free, private, works offline
  ⚠️  Slower on CPU (GPU speeds things up significantly)
  ⚠️  First run downloads ~2GB of model weights
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.config import settings
from src.models.schemas import (
    ImageAnalysis, VideoAnalysis, TextAnalysis,
    Label, Entity, Sentiment, TranscriptSegment,
)
from src.providers.base import BaseProvider, ProviderError
from src.utils.logger import get_logger

log = get_logger(__name__)


class HuggingFaceProvider(BaseProvider):
    """
    Each pipeline is loaded lazily and cached in _pipelines.
    This means startup is instant and models only load when actually needed.
    """

    def __init__(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ProviderError("huggingface", "Run: pip install transformers torch")
        self._pipelines: dict = {}

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_info(self) -> str:
        return "BLIP + ViT + Whisper-base + BART + RoBERTa + BERT-NER (local)"

    async def _load_pipeline(self, task: str, model: str):
        """Load a pipeline once and cache it. Runs model loading in a thread."""
        if task not in self._pipelines:
            log.info("hf.model.loading", task=task, model=model)
            from transformers import pipeline
            self._pipelines[task] = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline(task, model=model, token=settings.hf_token or None),
            )
            log.info("hf.model.ready", task=task)
        return self._pipelines[task]

    # ── Image ──────────────────────────────────────────────────────────────────

    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        start = time.monotonic()
        log.info("hf.image.start", file=image_path.name)
        try:
            from PIL import Image as PILImage
            image = PILImage.open(image_path).convert("RGB")

            captioner, classifier = await asyncio.gather(
                self._load_pipeline("image-to-text", "Salesforce/blip-image-captioning-base"),
                self._load_pipeline("image-classification", "google/vit-base-patch16-224"),
            )

            caption_out, class_out = await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(None, lambda: captioner(image)),
                asyncio.get_event_loop().run_in_executor(None, lambda: classifier(image)),
            )

            description = caption_out[0]["generated_text"] if caption_out else None
            labels = [
                Label(name=r["label"].replace("_", " "), confidence=round(r["score"], 3))
                for r in (class_out or [])[:10]
            ]

            ocr_text = await self._run_ocr(image)
            colours = self._dominant_colours(image)

            tags = list({l.name for l in labels if l.confidence > 0.1})
            if description:
                tags += [w for w in description.split() if len(w) > 4 and w.isalpha()]

            log.info("hf.image.done", ms=int((time.monotonic() - start) * 1000))
            return ImageAnalysis(
                labels=labels, objects=[], extracted_text=ocr_text,
                dominant_colours=colours, description=description, tags=tags[:15],
            )
        except Exception as e:
            raise ProviderError("huggingface", str(e)) from e

    async def _run_ocr(self, image) -> str | None:
        try:
            import pytesseract
            text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pytesseract.image_to_string(image).strip()
            )
            return text if len(text) > 3 else None
        except Exception:
            return None

    @staticmethod
    def _dominant_colours(image) -> list[str]:
        try:
            small = image.resize((150, 150))
            palette = small.quantize(colors=5).getpalette()[:15]
            return [f"#{palette[i]:02x}{palette[i+1]:02x}{palette[i+2]:02x}" for i in range(0, 15, 3)]
        except Exception:
            return []

    # ── Video ──────────────────────────────────────────────────────────────────

    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        start = time.monotonic()
        log.info("hf.video.start", file=video_path.name)
        audio_path = video_path.with_suffix(".wav")
        try:
            from moviepy.editor import VideoFileClip
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: VideoFileClip(str(video_path)).audio.write_audiofile(
                    str(audio_path), fps=16000, logger=None
                ),
            )

            asr = await self._load_pipeline("automatic-speech-recognition", "openai/whisper-base")
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: asr(str(audio_path), return_timestamps=True)
            )

            transcript = result.get("text", "")
            segments = [
                TranscriptSegment(
                    start_seconds=round(c["timestamp"][0] or 0, 2),
                    end_seconds=round(c["timestamp"][1] or 0, 2),
                    text=c["text"].strip(),
                )
                for c in result.get("chunks", []) if c.get("text", "").strip()
            ]

            text_result = await self.analyse_text(transcript) if transcript else None
            log.info("hf.video.done", ms=int((time.monotonic() - start) * 1000))

            return VideoAnalysis(
                transcript=transcript, transcript_segments=segments,
                summary=text_result.summary if text_result else None,
                detected_language=text_result.detected_language if text_result else None,
                tags=text_result.keywords[:10] if text_result else [],
            )
        except Exception as e:
            raise ProviderError("huggingface", str(e)) from e
        finally:
            if audio_path.exists():
                audio_path.unlink()

    # ── Text ───────────────────────────────────────────────────────────────────

    async def analyse_text(self, text: str) -> TextAnalysis:
        start = time.monotonic()
        truncated = text[:1024]
        try:
            summariser, sentiment_pipe, ner_pipe = await asyncio.gather(
                self._load_pipeline("summarization", "facebook/bart-large-cnn"),
                self._load_pipeline("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
                self._load_pipeline("ner", "dslim/bert-base-NER"),
            )

            summary_out, sentiment_out, ner_out = await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: summariser(truncated, max_length=130, min_length=30, do_sample=False)
                ),
                asyncio.get_event_loop().run_in_executor(None, lambda: sentiment_pipe(truncated[:512])),
                asyncio.get_event_loop().run_in_executor(None, lambda: ner_pipe(truncated[:512])),
            )

            summary = summary_out[0]["summary_text"] if summary_out else None

            label_map = {"positive": Sentiment.POSITIVE, "negative": Sentiment.NEGATIVE, "neutral": Sentiment.NEUTRAL}
            raw_label = sentiment_out[0]["label"].lower() if sentiment_out else "neutral"
            sentiment = label_map.get(raw_label, Sentiment.NEUTRAL)
            score = sentiment_out[0]["score"] if sentiment_out else 0.5
            sentiment_score = -score if sentiment == Sentiment.NEGATIVE else score

            entities = self._merge_ner_tokens(ner_out or [])
            keywords = self._extract_keywords(text)

            log.info("hf.text.done", ms=int((time.monotonic() - start) * 1000))
            return TextAnalysis(
                summary=summary, sentiment=sentiment, sentiment_score=round(sentiment_score, 3),
                entities=entities[:10], keywords=keywords[:10], word_count=len(text.split()),
            )
        except Exception as e:
            raise ProviderError("huggingface", str(e)) from e

    @staticmethod
    def _merge_ner_tokens(tokens: list) -> list[Entity]:
        """Reassemble BERT wordpiece tokens into full entity names."""
        entities, current_text, current_type = [], "", None
        for token in tokens:
            word = token["word"].lstrip("##")
            etype = token["entity"].replace("B-", "").replace("I-", "")
            if token["entity"].startswith("B-") or current_type != etype:
                if current_type and current_text:
                    entities.append(Entity(text=current_text, type=current_type, confidence=round(token["score"], 3)))
                current_type, current_text = etype, word
            else:
                current_text += word
        if current_type and current_text:
            entities.append(Entity(text=current_text, type=current_type, confidence=0.8))
        return entities

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
        """Frequency-based keyword extraction without a model."""
        stop = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by",
                "from","is","was","are","were","be","been","have","has","had","do","does",
                "did","this","that","these","those","it","its","as","not","no"}
        freq: dict[str, int] = {}
        for w in text.split():
            w = w.lower().strip(".,!?;:\"'()[]")
            if len(w) > 3 and w.isalpha() and w not in stop:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]  # type: ignore
