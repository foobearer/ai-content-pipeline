"""
tests/test_providers.py — Provider Unit Tests
──────────────────────────────────────────────
Tests for each AI provider using mocked API calls.

We mock the external APIs so tests:
  - Run fast (no network calls)
  - Work without API keys
  - Are deterministic (same result every time)
  - Don't cost money

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=src    # With coverage report
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# ── HuggingFace Provider Tests ─────────────────────────────────────────────────
# HuggingFace doesn't need API keys, so we can test more thoroughly

class TestHuggingFaceProvider:
    """Tests for the HuggingFace local provider."""

    def test_keyword_extraction(self):
        """
        _extract_keywords should return the most frequent meaningful words,
        filtering out stop words and short tokens.
        """
        from src.providers.huggingface_provider import HuggingFaceProvider
        text = "machine learning models are transforming natural language processing and computer vision"
        keywords = HuggingFaceProvider._extract_keywords(text, top_n=5)

        # Should include meaningful long words
        assert len(keywords) <= 5
        # Should not include stop words
        assert "are" not in keywords
        assert "and" not in keywords

    def test_ner_token_merging(self):
        """
        BERT NER splits words into subword tokens (wordpieces).
        _merge_ner_tokens should reassemble them into complete entity names.
        """
        from src.providers.huggingface_provider import HuggingFaceProvider
        tokens = [
            {"word": "Jo", "entity": "B-PER", "score": 0.99},
            {"word": "##hn", "entity": "I-PER", "score": 0.98},
            {"word": "London", "entity": "B-LOC", "score": 0.97},
        ]
        entities = HuggingFaceProvider._merge_ner_tokens(tokens)

        entity_texts = [e.text for e in entities]
        assert "John" in entity_texts
        assert "London" in entity_texts

    def test_colour_extraction(self):
        """
        _extract_colours should return a list of hex colour strings.
        """
        from PIL import Image
        from src.providers.huggingface_provider import HuggingFaceProvider

        # Create a simple test image (100×100 solid red)
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        colours = HuggingFaceProvider._extract_colours(img)

        assert isinstance(colours, list)
        assert len(colours) > 0
        # Each colour should be a valid hex string
        for colour in colours:
            assert colour.startswith("#")
            assert len(colour) == 7  # #rrggbb


# ── OpenAI Provider Tests ──────────────────────────────────────────────────────

class TestOpenAIProvider:
    """Tests for the OpenAI provider — all external calls are mocked."""

    def test_init_without_key_raises(self):
        """Should raise ProviderError immediately if no API key is configured."""
        from src.providers.base import ProviderError
        with patch("src.config.settings.openai_api_key", ""):
            with patch("src.config.settings.openai_available", False):
                from src.providers.openai_provider import OpenAIProvider
                with pytest.raises(ProviderError, match="No API key"):
                    OpenAIProvider()

    def test_mime_type_detection(self):
        """_get_mime_type should map file extensions to correct MIME types."""
        from src.providers.openai_provider import OpenAIProvider
        assert OpenAIProvider._get_mime_type(Path("photo.jpg")) == "image/jpeg"
        assert OpenAIProvider._get_mime_type(Path("image.PNG")) == "image/png"
        assert OpenAIProvider._get_mime_type(Path("anim.gif")) == "image/gif"
        assert OpenAIProvider._get_mime_type(Path("unknown.xyz")) == "image/jpeg"  # fallback

    @pytest.mark.asyncio
    async def test_analyse_text_success(self, tmp_path):
        """
        analyse_text should call the OpenAI API and parse the JSON response
        into a TextAnalysis object.
        """
        mock_response_json = {
            "summary": "A test summary.",
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "entities": [{"text": "London", "type": "LOCATION", "confidence": 0.9}],
            "keywords": ["test", "summary"],
            "detected_language": "en",
            "topics": ["testing"],
        }

        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = str(mock_response_json).replace("'", '"')

        with patch("src.config.settings.openai_available", True), \
             patch("src.config.settings.openai_api_key", "sk-test123"), \
             patch("openai.AsyncOpenAI") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_client_class.return_value = mock_client

            from src.providers.openai_provider import OpenAIProvider
            import json
            mock_completion.choices[0].message.content = json.dumps(mock_response_json)

            provider = OpenAIProvider()
            result = await provider.analyse_text("This is a test document about London.")

            assert result.summary == "A test summary."
            assert result.sentiment.value == "positive"
            assert len(result.entities) == 1
            assert result.entities[0].text == "London"


# ── File Handler Tests ─────────────────────────────────────────────────────────

class TestFileHandler:
    """Tests for file validation and storage utilities."""

    def test_detect_jpeg_from_magic_bytes(self):
        """Should identify JPEG from the FF D8 FF magic bytes header."""
        from src.utils.file_handler import _detect_content_type
        from src.models.schemas import ContentType

        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        result = _detect_content_type(jpeg_header, "photo.jpg")
        assert result == ContentType.IMAGE

    def test_detect_png_from_magic_bytes(self):
        """Should identify PNG from the 8-byte PNG signature."""
        from src.utils.file_handler import _detect_content_type
        from src.models.schemas import ContentType

        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = _detect_content_type(png_header, "image.png")
        assert result == ContentType.IMAGE

    def test_detect_pdf_from_magic_bytes(self):
        """Should identify PDF from the %PDF header."""
        from src.utils.file_handler import _detect_content_type
        from src.models.schemas import ContentType

        pdf_header = b"%PDF-1.4" + b"\x00" * 100
        result = _detect_content_type(pdf_header, "document.pdf")
        assert result == ContentType.DOCUMENT

    def test_unknown_type_returns_none(self):
        """Should return None for unrecognised file types."""
        from src.utils.file_handler import _detect_content_type

        random_bytes = b"\x00\x01\x02\x03" * 100
        result = _detect_content_type(random_bytes, "unknown.xyz")
        assert result is None


# ── Schema Tests ───────────────────────────────────────────────────────────────

class TestSchemas:
    """Tests that our Pydantic schemas validate correctly."""

    def test_label_confidence_must_be_0_to_1(self):
        """Confidence scores outside 0–1 should raise a validation error."""
        from pydantic import ValidationError
        from src.models.schemas import Label

        with pytest.raises(ValidationError):
            Label(name="cat", confidence=1.5)   # Too high

        with pytest.raises(ValidationError):
            Label(name="cat", confidence=-0.1)  # Too low

        # Valid case should not raise
        label = Label(name="cat", confidence=0.95)
        assert label.name == "cat"

    def test_provider_enum_values(self):
        """Provider enum should only accept the three valid values."""
        from src.models.schemas import Provider

        assert Provider("openai") == Provider.OPENAI
        assert Provider("google") == Provider.GOOGLE
        assert Provider("huggingface") == Provider.HUGGINGFACE

        with pytest.raises(ValueError):
            Provider("anthropic")  # Not a valid provider
