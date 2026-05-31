# 🤖 Multi-Modal AI Content Analysis Pipeline

> **Portfolio Project** by [Joycee Catamora Paragas](https://joycee.dev)  
> Inspired by real-world AI content analysis work at Extreme Reach UK Ltd  
> Built with Python · FastAPI · React · OpenAI · Google Cloud · HuggingFace

---

## What This Does

This project is a **full-stack, multi-modal AI content analysis pipeline** that can automatically process:

| Input Type | What it extracts |
|------------|-----------------|
| 🖼️ **Images** | Objects, labels, text (OCR), dominant colours, scene classification |
| 🎬 **Video** | Transcript, scene tags, content classification, key moments |
| 📄 **Documents / Text** | Summary, keywords, sentiment, entities, language detection |

It mirrors the kind of system used in production ad-tech and media platforms to automatically index, classify, and search large libraries of digital assets — removing the need for manual tagging.

**Key achievement this replicates:** Search results delivered 40% faster by removing manual transcription and indexing.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│         (Upload UI + Results Dashboard)              │
└─────────────────┬───────────────────────────────────┘
                  │ HTTP (multipart/form-data)
┌─────────────────▼───────────────────────────────────┐
│              FastAPI Backend (Python)                │
│                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   Router   │  │  Processor   │  │   Storage   │  │
│  │  /analyse  │→ │   Pipeline   │→ │  (Results)  │  │
│  └────────────┘  └──────┬───────┘  └─────────────┘  │
│                         │                            │
│              ┌──────────▼──────────┐                 │
│              │   Provider Factory  │                 │
│              └──┬──────┬───────┬───┘                 │
└─────────────────┼──────┼───────┼────────────────────┘
                  │      │       │
        ┌─────────▼┐ ┌───▼────┐ ┌▼──────────┐
        │  OpenAI  │ │Google  │ │HuggingFace│
        │ Vision + │ │Cloud   │ │(local,    │
        │ Whisper  │ │Vision  │ │no key)    │
        └──────────┘ └────────┘ └───────────┘
```

---

## Project Structure

```
ai-content-pipeline/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment variables template
├── docker-compose.yml          ← Run everything with one command
│
├── src/
│   ├── main.py                 ← FastAPI app entry point
│   ├── config.py               ← Configuration & provider settings
│   │
│   ├── providers/              ← AI provider implementations
│   │   ├── __init__.py
│   │   ├── base.py             ← Abstract base class (interface)
│   │   ├── openai_provider.py  ← OpenAI Vision + Whisper
│   │   ├── google_provider.py  ← Google Cloud Vision + Speech
│   │   └── huggingface_provider.py ← Local HuggingFace models
│   │
│   ├── processors/             ← Media type processors
│   │   ├── __init__.py
│   │   ├── image_processor.py  ← Image analysis pipeline
│   │   ├── video_processor.py  ← Video analysis pipeline
│   │   └── text_processor.py   ← Text/document analysis pipeline
│   │
│   ├── models/                 ← Pydantic data models
│   │   ├── __init__.py
│   │   └── schemas.py          ← Request/response schemas
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py     ← File validation & temp storage
│       └── logger.py           ← Structured logging
│
├── frontend/
│   ├── index.html              ← Single-file React app (no build needed)
│   └── styles.css
│
├── tests/
│   ├── test_providers.py       ← Unit tests for each provider
│   ├── test_processors.py      ← Integration tests
│   └── conftest.py             ← Pytest fixtures
│
├── docs/
│   ├── SETUP.md                ← Detailed setup guide per provider
│   ├── API.md                  ← API endpoint documentation
│   └── ARCHITECTURE.md         ← Deep dive into design decisions
│
└── samples/                    ← Sample files to test with
    ├── sample.jpg
    ├── sample.txt
    └── README.md
```

---

## Quick Start

### Option A — HuggingFace only (no API key needed, runs locally)

```bash
# 1. Clone the repo
git clone https://github.com/foobearer/ai-content-pipeline.git
cd ai-content-pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file (no keys needed for HuggingFace)
cp .env.example .env

# 5. Start the server
uvicorn src.main:app --reload

# 6. Open your browser
# API docs:     http://localhost:8000/docs
# Frontend:     http://localhost:8000
```

### Option B — With OpenAI

```bash
# After step 4 above, edit .env and add:
OPENAI_API_KEY=sk-your-key-here

# Get your key at: https://platform.openai.com/api-keys
# Free tier: $5 credit on signup
```

### Option C — With Google Cloud

```bash
# After step 4 above, add to .env:
GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json

# Setup guide: docs/SETUP.md#google-cloud
```

### Option D — Docker (easiest, everything included)

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up
# Visit http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/health` | Health check + provider status |
| `GET` | `/providers` | List available providers |
| `POST` | `/analyse/image` | Analyse an uploaded image |
| `POST` | `/analyse/video` | Analyse a video (URL or upload) |
| `POST` | `/analyse/text` | Analyse text or document |
| `POST` | `/analyse/auto` | Auto-detect type and analyse |
| `GET` | `/results/{job_id}` | Get analysis results by ID |

Full API docs available at `http://localhost:8000/docs` (Swagger UI auto-generated).

---

## Provider Comparison

| Feature | OpenAI | Google Cloud | HuggingFace |
|---------|--------|--------------|-------------|
| Image labels | ✅ GPT-4V | ✅ Vision API | ✅ BLIP/ViT |
| OCR (text in images) | ✅ | ✅ | ✅ Tesseract |
| Video transcript | ✅ Whisper | ✅ Speech-to-Text | ✅ Whisper local |
| Text summary | ✅ GPT-4 | ✅ Natural Language | ✅ BART |
| Sentiment | ✅ | ✅ | ✅ |
| API key required | ✅ Yes | ✅ Yes | ❌ No |
| Cost | Pay-per-use | Pay-per-use | Free (local) |
| Speed | Fast | Fast | Slower (CPU) |
| Best for | Quality | Google ecosystem | Privacy / offline |

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — async Python web framework
- [Pydantic](https://docs.pydantic.dev/) — data validation and schemas
- [Uvicorn](https://www.uvicorn.org/) — ASGI server
- [Pillow](https://pillow.readthedocs.io/) — image processing
- [MoviePy](https://zulko.github.io/moviepy/) — video processing
- [python-multipart](https://andrew-d.github.io/python-multipart/) — file uploads

**AI Providers**
- [OpenAI Python SDK](https://github.com/openai/openai-python) — GPT-4V, Whisper
- [Google Cloud Client Libraries](https://cloud.google.com/python/docs/reference) — Vision, Speech, NL
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — local models

**Frontend**
- Vanilla React (CDN, no build step) — keeps it simple to run
- No framework dependencies — just open the HTML file

---

## Licence

MIT — free to use, modify, and share.

---

*Built by Joycee Catamora Paragas · [joycee.dev](https://joycee.dev) · [github.com/foobearer](https://github.com/foobearer)*
