"""
main.py — FastAPI Application
───────────────────────────────
Run:  uvicorn src.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import uuid
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models.schemas import (
    AnalysisResult, HealthResponse,
    Provider, ContentType, ProviderInfo,
    ImageAnalysis, VideoAnalysis, TextAnalysis,
)
from src.providers import get_provider
from src.providers.base import ProviderError
from src.utils.file_handler import save_upload, cleanup
from src.utils.logger import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multi-modal AI content analysis pipeline. Supports images, videos, text, and documents. "
        "Switch between OpenAI, Google Cloud, and HuggingFace providers.\n\n"
        "Built by [Joycee Catamora Paragas](https://joycee.dev)"
    ),
)

# Allow the Vite dev server (port 5173) and any production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Returns which providers are configured and ready."""
    return HealthResponse(providers=settings.provider_status(), default_provider=settings.default_provider)


@app.get("/providers", tags=["System"])
async def list_providers():
    """Returns all providers with their capabilities and configuration status."""
    return {
        "providers": [
            {
                "id": "openai", "name": "OpenAI", "configured": settings.openai_available,
                "models": "GPT-4o + Whisper-1", "requires_key": True,
                "capabilities": ["image", "video", "text"],
                "key_url": "https://platform.openai.com/api-keys",
            },
            {
                "id": "google", "name": "Google Cloud", "configured": settings.google_available,
                "models": "Vision API + Speech-to-Text + NL API", "requires_key": True,
                "capabilities": ["image", "video", "text"],
                "key_url": "https://console.cloud.google.com",
            },
            {
                "id": "huggingface", "name": "HuggingFace", "configured": True,
                "models": "BLIP + Whisper-base + BART + RoBERTa + BERT-NER",
                "requires_key": False,
                "capabilities": ["image", "video", "text"],
                "note": "Runs locally. First use downloads ~2GB of models.",
            },
        ]
    }


# ── Analysis endpoints ─────────────────────────────────────────────────────────

@app.post("/analyse/image", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_image(
    file: UploadFile = File(..., description="JPEG, PNG, GIF, or WebP"),
    provider: Provider = Form(Provider.HUGGINGFACE),
):
    """Analyse an image: labels, objects, OCR, dominant colours, description, tags."""
    return await _run(file, provider, ContentType.IMAGE)


@app.post("/analyse/video", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_video(
    file: UploadFile = File(..., description="MP4, MOV, AVI, or WebM"),
    provider: Provider = Form(Provider.HUGGINGFACE),
):
    """Analyse a video: transcript, timestamped segments, tags, summary."""
    return await _run(file, provider, ContentType.VIDEO)


@app.post("/analyse/text", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_text(
    file: UploadFile = File(..., description="TXT, PDF, or DOCX"),
    provider: Provider = Form(Provider.HUGGINGFACE),
):
    """Analyse a document: summary, sentiment, entities, keywords, topics."""
    return await _run(file, provider, ContentType.TEXT)


@app.post("/analyse/auto", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_auto(
    file: UploadFile = File(..., description="Any supported file — type is auto-detected"),
    provider: Provider = Form(Provider.HUGGINGFACE),
):
    """Auto-detect file type and route to the correct analysis pipeline."""
    return await _run(file, provider, content_type=None)


# ── Core orchestrator ──────────────────────────────────────────────────────────

async def _run(
    upload: UploadFile,
    provider_choice: Provider,
    content_type: ContentType | None,
) -> AnalysisResult:
    job_id = uuid.uuid4().hex
    start = time.monotonic()
    path: Path | None = None

    log.info("job.start", job_id=job_id, provider=provider_choice, file=upload.filename)

    try:
        path, detected_type = await save_upload(upload)
        resolved = content_type or detected_type

        try:
            ai = get_provider(provider_choice)
        except ProviderError as e:
            raise HTTPException(status_code=400, detail=str(e))

        img_result: ImageAnalysis | None = None
        vid_result: VideoAnalysis | None = None
        txt_result: TextAnalysis | None = None

        if resolved == ContentType.IMAGE:
            img_result = await ai.analyse_image(path)
        elif resolved == ContentType.VIDEO:
            vid_result = await ai.analyse_video(path)
        elif resolved in (ContentType.TEXT, ContentType.DOCUMENT):
            raw_text = await _read_text(path, resolved)
            txt_result = await ai.analyse_text(raw_text)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {resolved}")

        ms = int((time.monotonic() - start) * 1000)
        log.info("job.done", job_id=job_id, ms=ms)

        return AnalysisResult(
            job_id=job_id,
            content_type=resolved,
            filename=upload.filename,
            provider=ProviderInfo(name=provider_choice, model_used=ai.model_info, processing_time_ms=ms),
            image_analysis=img_result,
            video_analysis=vid_result,
            text_analysis=txt_result,
        )

    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.error("job.failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path:
            cleanup(path)


async def _read_text(path: Path, content_type: ContentType) -> str:
    """Extract raw text from a TXT, PDF, or DOCX file."""
    if content_type == ContentType.TEXT:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".pdf":
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF support: pip install PyPDF2")
    if path.suffix in (".doc", ".docx"):
        try:
            import docx
            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(status_code=400, detail="DOCX support: pip install python-docx")
    raise HTTPException(status_code=400, detail=f"Cannot extract text from {path.suffix}")
