"""
providers/base.py — Abstract Base Class for AI Providers
──────────────────────────────────────────────────────────
This defines the interface (contract) that every provider must implement.

Why use an abstract base class?
  - All three providers (OpenAI, Google, HuggingFace) have the same methods
  - The processor classes don't need to know WHICH provider they're using
  - Adding a new provider later only requires implementing this interface
  - This pattern is called the "Strategy Pattern"

Design rule: The processors call these methods. The providers implement them.
The main app wires them together based on the user's choice.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from src.models.schemas import ImageAnalysis, VideoAnalysis, TextAnalysis


class BaseProvider(ABC):
    """
    Abstract base class that all AI providers must inherit from.

    Every subclass must implement all four methods below.
    If a method is not implemented, Python will raise a TypeError
    when you try to instantiate the class — a fast, clear failure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable provider name.
        Used in logs and API responses.
        Example: "openai", "google", "huggingface"
        """
        ...

    @property
    @abstractmethod
    def model_info(self) -> str:
        """
        The specific model(s) this provider is using.
        Used in API responses so callers know exactly what processed their data.
        Example: "gpt-4o + whisper-1"
        """
        ...

    @abstractmethod
    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """
        Analyse an image file and return structured results.

        Args:
            image_path: Path to the image file on disk (already validated)

        Returns:
            ImageAnalysis with labels, objects, OCR text, colours, tags

        Raises:
            ProviderError: If the API call fails or returns unusable data
        """
        ...

    @abstractmethod
    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        """
        Analyse a video file: extract audio, transcribe speech, classify visuals.

        Args:
            video_path: Path to the video file on disk

        Returns:
            VideoAnalysis with transcript, tags, summary, duration

        Raises:
            ProviderError: If processing fails
        """
        ...

    @abstractmethod
    async def analyse_text(self, text: str) -> TextAnalysis:
        """
        Analyse a text string: summarise, extract entities, detect sentiment.

        Args:
            text: Raw text content (from form input or extracted from a document)

        Returns:
            TextAnalysis with summary, sentiment, entities, keywords

        Raises:
            ProviderError: If analysis fails
        """
        ...

    def __repr__(self) -> str:
        return f"<Provider: {self.name} using {self.model_info}>"


class ProviderError(Exception):
    """
    Raised when an AI provider fails to process a request.

    Carries the provider name so error messages are clear:
        ProviderError("openai", "API rate limit exceeded")
        → "OpenAI provider error: API rate limit exceeded"
    """

    def __init__(self, provider_name: str, message: str):
        self.provider_name = provider_name
        self.message = message
        super().__init__(f"{provider_name.title()} provider error: {message}")
