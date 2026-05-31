"""
main.py — FastAPI Application Entry Point
───────────────────────────────────────────
This is where all the pieces come together.

Run with:
    uvicorn src.main:app --reload          # Development (auto-reloads on save)
    uvicorn src.main:app --host 0.0.0.0   # Production

API docs auto-generated at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import uuid
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.models.schemas import (
    AnalysisResult, AnalysisRequest, HealthResponse,
    Provider, ContentType, ProviderInfo
)
from src.providers import get_provider
from src.providers.base import ProviderError
from src.utils.file_handler import save_upload, cleanup
from src.utils.logger import setup_logging, get_logger

# ── Initialise logging ─────────────────────────────────────────────────────────
setup_logging()
log = get_logger(__name__)

# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A multi-modal AI content analysis pipeline that supports images, videos, "
        "and text/documents. Supports OpenAI, Google Cloud, and HuggingFace providers.\n\n"
        "Built by [Joycee Catamora Paragas](https://joycee.dev)"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow the frontend to call the API ──────────────────────────────────
# In production, replace "*" with your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────────
# Serve the React frontend from the /frontend directory
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the React frontend app."""
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": f"{settings.app_name} API is running. Visit /docs for the API reference."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check which AI providers are configured and ready.

    Use this to verify your API keys are working before running analyses.
    HuggingFace is always available (no key required).
    """
    return HealthResponse(
        status="ok",
        providers=settings.provider_status(),
        default_provider=settings.default_provider,
    )


@app.get("/providers", tags=["System"])
async def list_providers():
    """
    List all available providers with their configuration status and capabilities.
    """
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "configured": settings.openai_available,
                "models": "GPT-4o Vision + Whisper-1",
                "capabilities": ["image", "video", "text"],
                "requires_key": True,
                "key_url": "https://platform.openai.com/api-keys",
            },
            {
                "id": "google",
                "name": "Google Cloud",
                "configured": settings.google_available,
                "models": "Vision API + Speech-to-Text + Natural Language API",
                "capabilities": ["image", "video", "text"],
                "requires_key": True,
                "key_url": "https://console.cloud.google.com",
            },
            {
                "id": "huggingface",
                "name": "HuggingFace",
                "configured": True,
                "models": "BLIP + Whisper-base + BART + RoBERTa + BERT-NER",
                "capabilities": ["image", "video", "text"],
                "requires_key": False,
                "note": "Runs locally. First use downloads ~2GB of models.",
            },
        ]
    }


@app.post("/analyse/image", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, GIF, WebP)"),
    provider: Provider = Form(Provider.HUGGINGFACE, description="Which AI provider to use"),
):
    """
    Analyse an uploaded image.

    Returns:
    - **labels**: Scene and content classification labels with confidence scores
    - **objects**: Detected objects, optionally with bounding box coordinates
    - **extracted_text**: Any text visible in the image (OCR)
    - **dominant_colours**: Hex colour codes of the most prominent colours
    - **description**: Natural language description of the image
    - **tags**: Flat list of searchable keywords for media library indexing
    - **is_safe**: Whether the image passes content safety checks
    """
    return await _run_analysis(file, provider, ContentType.IMAGE)


@app.post("/analyse/video", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI, WebM)"),
    provider: Provider = Form(Provider.HUGGINGFACE, description="Which AI provider to use"),
):
    """
    Analyse an uploaded video.

    Extracts audio, transcribes speech, and generates searchable tags.

    Returns:
    - **transcript**: Full text transcript of all spoken audio
    - **transcript_segments**: Timestamped chunks — great for building search indexes
    - **summary**: Brief description of the video's content
    - **tags**: Searchable keywords derived from audio and visual content
    - **detected_language**: ISO language code of the spoken language
    """
    return await _run_analysis(file, provider, ContentType.VIDEO)


@app.post("/analyse/text", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_text_file(
    file: UploadFile = File(..., description="Text file or document (TXT, PDF, DOCX)"),
    provider: Provider = Form(Provider.HUGGINGFACE, description="Which AI provider to use"),
):
    """
    Analyse a text file or document.

    Returns:
    - **summary**: 2–3 sentence condensed version of the content
    - **sentiment**: positive / negative / neutral / mixed
    - **sentiment_score**: -1.0 (very negative) to +1.0 (very positive)
    - **entities**: Named entities — people, places, organisations
    - **keywords**: Top 10 most important words/phrases
    - **topics**: High-level topic categories
    """
    return await _run_analysis(file, provider, ContentType.TEXT)


@app.post("/analyse/auto", response_model=AnalysisResult, tags=["Analysis"])
async def analyse_auto(
    file: UploadFile = File(..., description="Any supported file — type is auto-detected"),
    provider: Provider = Form(Provider.HUGGINGFACE, description="Which AI provider to use"),
):
    """
    Upload any file and let the pipeline auto-detect the content type.

    Useful for building upload UIs where users can drop any file type.
    The pipeline detects the file format from its content (magic bytes),
    not just the file extension.
    """
    return await _run_analysis(file, provider, content_type=None)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run_analysis(
    upload: UploadFile,
    provider_choice: Provider,
    content_type: ContentType | None,
) -> AnalysisResult:
    """
    Core analysis orchestrator — shared by all /analyse/* endpoints.

    Flow:
    1. Save the upload to a temp file
    2. Auto-detect content type if not specified
    3. Initialise the chosen AI provider
    4. Route to the correct analysis method
    5. Clean up the temp file
    6. Return the structured result
    """
    job_id = uuid.uuid4().hex
    start = time.monotonic()
    temp_path: Path | None = None

    log.info("analysis.start", job_id=job_id, provider=provider_choice, filename=upload.filename)

    try:
        # Step 1 & 2: Save upload, detect content type
        temp_path, detected_type = await save_upload(upload)
        resolved_type = content_type or detected_type

        # Step 3: Get the provider
        try:
            ai_provider = get_provider(provider_choice)
        except ProviderError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Step 4: Route to the right analysis method
        image_analysis = None
        video_analysis = None
        text_analysis = None

        if resolved_type == ContentType.IMAGE:
            image_analysis = await ai_provider.analyse_image(temp_path)

        elif resolved_type == ContentType.VIDEO:
            video_analysis = await ai_provider.analyse_video(temp_path)

        elif resolved_type in (ContentType.TEXT, ContentType.DOCUMENT):
            # Extract text from the file
            raw_text = await _extract_text(temp_path, resolved_type)
            text_analysis = await ai_provider.analyse_text(raw_text)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {resolved_type}")

        elapsed_ms = int((time.monotonic() - start) * 1000)
        log.info("analysis.complete", job_id=job_id, elapsed_ms=elapsed_ms)

        return AnalysisResult(
            job_id=job_id,
            content_type=resolved_type,
            filename=upload.filename,
            provider=ProviderInfo(
                name=provider_choice,
                model_used=ai_provider.model_info,
                processing_time_ms=elapsed_ms,
            ),
            image_analysis=image_analysis,
            video_analysis=video_analysis,
            text_analysis=text_analysis,
        )

    except ProviderError as e:
        log.error("analysis.provider_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=502, detail=str(e))

    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions as-is

    except Exception as e:
        log.error("analysis.unexpected_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    finally:
        # Always clean up the temp file, even if analysis failed
        if temp_path:
            cleanup(temp_path)


async def _extract_text(path: Path, content_type: ContentType) -> str:
    """
    Extract raw text from a file based on its type.

    For .txt: read directly
    For .pdf: use PyPDF2 to extract text from pages
    For .docx: use python-docx to extract paragraphs
    """
    if content_type == ContentType.TEXT:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read text file: {e}")

    elif content_type == ContentType.DOCUMENT:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                import PyPDF2
                text_parts = []
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                return "\n".join(text_parts)
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="PDF support requires PyPDF2. Run: pip install PyPDF2"
                )

        elif suffix in (".doc", ".docx"):
            try:
                import docx
                doc = docx.Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="Word document support requires python-docx. Run: pip install python-docx"
                )

    raise HTTPException(status_code=400, detail=f"Cannot extract text from {path.suffix} files")
