"""
utils/file_handler.py — Upload Validation and Storage
───────────────────────────────────────────────────────
Validates files by reading magic bytes (not just the extension),
saves to a temp directory, and cleans up after analysis.
"""

import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import UploadFile, HTTPException

from src.config import settings
from src.models.schemas import ContentType
from src.utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_TYPES: dict[str, ContentType] = {
    "image/jpeg": ContentType.IMAGE, "image/png": ContentType.IMAGE,
    "image/gif": ContentType.IMAGE, "image/webp": ContentType.IMAGE,
    "video/mp4": ContentType.VIDEO, "video/mpeg": ContentType.VIDEO,
    "video/quicktime": ContentType.VIDEO, "video/webm": ContentType.VIDEO,
    "text/plain": ContentType.TEXT, "application/pdf": ContentType.DOCUMENT,
    "application/msword": ContentType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.DOCUMENT,
}


async def save_upload(upload: UploadFile) -> tuple[Path, ContentType]:
    """
    Validate and persist an uploaded file to the temp directory.
    Returns (saved_path, detected_content_type).
    Raises HTTPException 400/413 on validation failure.
    """
    content = await upload.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_file_size_mb}MB, got {len(content)/1024/1024:.1f}MB."
        )

    content_type = _detect_type(content, upload.filename or "")
    if content_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{Path(upload.filename or '').suffix}'. "
                   "Supported: JPEG, PNG, GIF, WebP, MP4, MOV, AVI, WebM, TXT, PDF, DOC, DOCX."
        )

    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "file").suffix.lower() or ".bin"
    path = temp_dir / f"{uuid.uuid4().hex}{suffix}"

    async with aiofiles.open(path, "wb") as f:
        await f.write(content)

    log.info("file.saved", path=str(path), bytes=len(content), type=content_type)
    return path, content_type


def cleanup(path: Path) -> None:
    """Delete a temp file silently — never raises."""
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        log.warning("file.cleanup_failed", path=str(path), error=str(e))


def _detect_type(content: bytes, filename: str) -> Optional[ContentType]:
    """Detect content type from magic bytes with extension fallback."""
    # Magic byte checks
    if content[:3] == b"\xff\xd8\xff":
        return ContentType.IMAGE
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ContentType.IMAGE
    if content[:4] == b"GIF8":
        return ContentType.IMAGE
    if content[:4] == b"%PDF":
        return ContentType.DOCUMENT

    # python-magic (optional)
    try:
        import magic
        mime = magic.from_buffer(content[:1024], mime=True)
        if mime in ALLOWED_TYPES:
            return ALLOWED_TYPES[mime]
    except Exception:
        pass

    # Extension fallback
    ext_map = {
        ".jpg": ContentType.IMAGE, ".jpeg": ContentType.IMAGE, ".png": ContentType.IMAGE,
        ".gif": ContentType.IMAGE, ".webp": ContentType.IMAGE,
        ".mp4": ContentType.VIDEO, ".mov": ContentType.VIDEO,
        ".avi": ContentType.VIDEO, ".webm": ContentType.VIDEO,
        ".txt": ContentType.TEXT, ".pdf": ContentType.DOCUMENT,
        ".doc": ContentType.DOCUMENT, ".docx": ContentType.DOCUMENT,
    }
    return ext_map.get(Path(filename).suffix.lower())
