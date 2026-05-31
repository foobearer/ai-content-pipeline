"""
providers/base.py — Abstract Provider Interface
─────────────────────────────────────────────────
Defines the contract every AI provider must fulfil.
The Strategy Pattern: processors call these methods without knowing
which concrete provider is behind them.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from src.models.schemas import ImageAnalysis, VideoAnalysis, TextAnalysis


class BaseProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier e.g. 'openai'"""
        ...

    @property
    @abstractmethod
    def model_info(self) -> str:
        """Specific models in use e.g. 'gpt-4o + whisper-1'"""
        ...

    @abstractmethod
    async def analyse_image(self, image_path: Path) -> ImageAnalysis:
        """Analyse an image file. Raises ProviderError on failure."""
        ...

    @abstractmethod
    async def analyse_video(self, video_path: Path) -> VideoAnalysis:
        """Extract and analyse video audio + visuals. Raises ProviderError on failure."""
        ...

    @abstractmethod
    async def analyse_text(self, text: str) -> TextAnalysis:
        """Analyse raw text. Raises ProviderError on failure."""
        ...

    def __repr__(self) -> str:
        return f"<Provider: {self.name} | {self.model_info}>"


class ProviderError(Exception):
    """Raised when a provider API call fails."""

    def __init__(self, provider_name: str, message: str):
        self.provider_name = provider_name
        self.message = message
        super().__init__(f"{provider_name.title()} error: {message}")
