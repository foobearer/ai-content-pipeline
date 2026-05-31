"""
config.py — Application Configuration
──────────────────────────────────────
All settings are loaded from environment variables (or a .env file).

We use pydantic-settings so every value is type-checked and validated
on startup — the app will fail immediately with a clear error message
if a required value is missing or malformed, rather than failing later
in a confusing way.

Usage:
    from src.config import settings

    if settings.openai_api_key:
        # OpenAI is available
        ...
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.models.schemas import Provider


class Settings(BaseSettings):
    """
    Application settings — values are read from environment variables.
    Falls back to defaults if a variable isn't set.
    """

    # ── pydantic-settings config ───────────────────────────────────────────
    # Tell pydantic-settings to also look in a .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # OPENAI_API_KEY and openai_api_key both work
        extra="ignore",         # Don't crash on unknown env vars
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_name: str = "Multi-Modal AI Content Analysis Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    default_provider: Provider = Provider.HUGGINGFACE
    max_file_size_mb: int = Field(50, ge=1, le=500)
    temp_dir: str = "/tmp/ai_pipeline"

    # ── OpenAI ────────────────────────────────────────────────────────────
    # Optional — leave empty if you don't have an OpenAI key
    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o"
    openai_text_model: str = "gpt-4o-mini"
    openai_whisper_model: str = "whisper-1"

    # ── Google Cloud ──────────────────────────────────────────────────────
    # Optional — leave empty if you don't have Google credentials
    google_application_credentials: str = ""
    google_cloud_project: str = ""
    google_cloud_region: str = "us-central1"

    # ── HuggingFace ───────────────────────────────────────────────────────
    # No API key required for local models
    hf_home: str = ""          # Override cache location
    hf_token: str = ""         # Only needed for private models
    transformers_verbosity: str = "info"

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB setting to bytes for comparison against uploaded files."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def openai_available(self) -> bool:
        """True if an OpenAI API key has been provided."""
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def google_available(self) -> bool:
        """True if Google Cloud credentials have been configured."""
        return bool(self.google_application_credentials or self.google_cloud_project)

    @property
    def huggingface_available(self) -> bool:
        """
        HuggingFace is always available — it runs locally with no API key.
        Returns True unconditionally.
        """
        return True

    def provider_status(self) -> dict[str, bool]:
        """
        Return a dict showing which providers are ready to use.
        Used by the /health endpoint.
        """
        return {
            "openai": self.openai_available,
            "google": self.google_available,
            "huggingface": self.huggingface_available,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Return the global Settings instance.

    lru_cache() ensures this is only created once — subsequent calls
    return the same object. This avoids re-reading .env on every request.

    Usage:
        from src.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Convenience export so callers can just do `from src.config import settings`
settings = get_settings()
