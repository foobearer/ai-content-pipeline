# Dockerfile — Multi-stage build for the AI Content Analysis Pipeline
#
# Stage 1: Install Python dependencies
# Stage 2: Copy source and run
#
# Built on Python 3.11 slim to keep image size reasonable (~800MB with models)

FROM python:3.11-slim AS base

# Install system dependencies:
# - tesseract-ocr: required by pytesseract for OCR
# - ffmpeg: required by moviepy for video/audio processing
# - libmagic: required by python-magic for file type detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching —
# this layer is only rebuilt if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY frontend/ ./frontend/

# Create temp directory for uploads
RUN mkdir -p /tmp/ai_pipeline

# Expose the API port
EXPOSE 8000

# Default command — override in docker-compose.yml for development
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
