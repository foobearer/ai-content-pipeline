"""
providers/__init__.py — Provider Factory
─────────────────────────────────────────
The factory function is the single place in the codebase that decides
which provider class to instantiate.

Usage:
    from src.providers import get_provider
    provider = get_provider(Provider.OPENAI)
    result = await provider.analyse_image(path)
"""

from src.models.schemas import Provider
from src.providers.base import BaseProvider, ProviderError


def get_provider(provider: Provider) -> BaseProvider:
    """
    Instantiate and return the requested AI provider.

    Args:
        provider: Which provider to use (from the Provider enum)

    Returns:
        A fully initialised provider instance

    Raises:
        ProviderError: If the provider isn't configured (missing API key)
        ValueError: If an unknown provider name is passed

    Example:
        provider = get_provider(Provider.HUGGINGFACE)
        result = await provider.analyse_text("Hello world")
    """
    if provider == Provider.OPENAI:
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    elif provider == Provider.GOOGLE:
        from src.providers.google_provider import GoogleProvider
        return GoogleProvider()

    elif provider == Provider.HUGGINGFACE:
        from src.providers.huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider()

    else:
        raise ValueError(f"Unknown provider: {provider}")
