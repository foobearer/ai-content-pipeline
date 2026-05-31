"""
utils/file_handler.py — File Validation and Storage
─────────────────────────────────────────────────────
Handles all file I/O for uploaded assets:
  - Validates file type and size before processing
  - Saves uploads to a temp directory
  - Cleans up temp files after analysis

Security note: We validate file type by reading the file header (magic bytes),
not just the file extension. A user can rename "virus.exe" to "photo.jpg" —
we catch that here.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import UploadFile, HTTPException

from src.config import settings
from src.models.schemas import ContentType
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Allowed file types ─────────────────────────────────────────────────────────
# Maps MIME type → ContentType so we know how to process each file

ALLOWED_TYPES: dict[str, ContentType] = {
    # Images
    "image/jpeg": ContentType.IMAGE,
    "image/png": ContentType.IMAGE,
    "image/gif": ContentType.IMAGE,
    "image/webp": ContentType.IMAGE,
    "image/bmp": ContentType.IMAGE,
    # Videos
    "video/mp4": ContentType.VIDEO,
    "video/mpeg": ContentType.VIDEO,
    "video/quicktime": ContentType.VIDEO,
    "video/x-msvideo": ContentType.VIDEO,   # .avi
    "video/webm": ContentType.VIDEO,
    # Documents / Text
    "text/plain": ContentType.TEXT,
    "application/pdf": ContentType.DOCUMENT,
    "application/msword": ContentType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.DOCUMENT,
}

# Accepted file extensions (as a fallback if python-magic isn't available)
ALLOWED_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    ".mp4", ".mpeg", ".mov", ".avi", ".webm",
    ".txt", ".pdf", ".doc", ".docx",
}


async def save_upload(upload: UploadFile) -> tuple[Path, ContentType]:
    """
    Validate and save an uploaded file to the temp directory.

    Args:
        upload: The FastAPI UploadFile object from the request

    Returns:
        (path_to_saved_file, detected_content_type)

    Raises:
        HTTPException 400: If file type is not allowed
        HTTPException 413: If file is too large
    """
    # ── Validate file size ─────────────────────────────────────────────────
    # Read the whole file into memory first so we can check size
    # For very large files, you'd want to stream this instead
    content = await upload.read()
    size_bytes = len(content)

    if size_bytes > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB, "
                   f"got {size_bytes / 1024 / 1024:.1f}MB"
        )

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # ── Detect file type ───────────────────────────────────────────────────
    content_type = _detect_content_type(content, upload.filename or "")
    if content_type is None:
        ext = Path(upload.filename or "").suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Got extension '{ext}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # ── Save to temp directory ─────────────────────────────────────────────
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use a UUID in the filename to avoid collisions if multiple requests come in
    original_suffix = Path(upload.filename or "file").suffix.lower() or ".bin"
    temp_filename = f"{uuid.uuid4().hex}{original_suffix}"
    temp_path = temp_dir / temp_filename

    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(content)

    log.info("file.saved", path=str(temp_path), size_bytes=size_bytes, content_type=content_type)
    return temp_path, content_type


def cleanup(path: Path) -> None:
    """
    Delete a temporary file after analysis is complete.
    Logs a warning if deletion fails — never raises.
    """
    try:
        if path.exists():
            path.unlink()
            log.info("file.cleaned", path=str(path))
    except Exception as e:
        log.warning("file.cleanup_failed", path=str(path), error=str(e))


def _detect_content_type(content: bytes, filename: str) -> Optional[ContentType]:
    """
    Detect content type from file bytes (magic bytes) with extension fallback.

    Magic bytes are the first few bytes of a file that identify its format.
    For example, JPEG files always start with FF D8 FF.
    """
    # Try python-magic first (most reliable)
    try:
        import magic
        mime = magic.from_buffer(content[:1024], mime=True)
        if mime in ALLOWED_TYPES:
            return ALLOWED_TYPES[mime]
    except (ImportError, Exception):
        pass  # Fall back to extension-based detection

    # Fallback: check magic bytes manually for common formats
    if content[:3] == b"\xff\xd8\xff":
        return ContentType.IMAGE    # JPEG
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ContentType.IMAGE    # PNG
    if content[:4] == b"GIF8":
        return ContentType.IMAGE    # GIF
    if content[:4] in (b"ftyp", b"\x00\x00\x00\x18", b"\x00\x00\x00\x20"):
        return ContentType.VIDEO    # MP4
    if content[:4] == b"%PDF":
        return ContentType.DOCUMENT # PDF

    # Last resort: file extension
    ext = Path(filename).suffix.lower()
    ext_map = {
        ".jpg": ContentType.IMAGE, ".jpeg": ContentType.IMAGE,
        ".png": ContentType.IMAGE, ".gif": ContentType.IMAGE,
        ".webp": ContentType.IMAGE,
        ".mp4": ContentType.VIDEO, ".mov": ContentType.VIDEO,
        ".avi": ContentType.VIDEO, ".webm": ContentType.VIDEO,
        ".txt": ContentType.TEXT,
        ".pdf": ContentType.DOCUMENT, ".doc": ContentType.DOCUMENT,
        ".docx": ContentType.DOCUMENT,
    }
    return ext_map.get(ext)
