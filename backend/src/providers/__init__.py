"""
providers/__init__.py — Provider Factory
─────────────────────────────────────────
Single place that maps a Provider enum value to its implementation class.
Providers are imported lazily so missing optional dependencies don't crash
the app at startup — only when that provider is actually requested.
"""

from src.models.schemas import Provider
from src.providers.base import BaseProvider, ProviderError


def get_provider(provider: Provider) -> BaseProvider:
    """
    Instantiate and return the requested AI provider.

    Raises:
        ProviderError: Provider isn't configured (missing API key / credentials)
        ValueError: Unknown provider name
    """
    if provider == Provider.OPENAI:
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if provider == Provider.GOOGLE:
        from src.providers.google_provider import GoogleProvider
        return GoogleProvider()

    if provider == Provider.HUGGINGFACE:
        from src.providers.huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider()

    raise ValueError(f"Unknown provider: {provider}")
