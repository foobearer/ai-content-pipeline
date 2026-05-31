"""
config.py — Application Settings
──────────────────────────────────
Reads all configuration from environment variables / .env file.
Using pydantic-settings means every value is validated at startup.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.models.schemas import Provider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Multi-Modal AI Content Analysis Pipeline"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    default_provider: Provider = Provider.HUGGINGFACE
    max_file_size_mb: int = Field(50, ge=1, le=500)
    temp_dir: str = "/tmp/ai_pipeline"

    # OpenAI — leave empty if not used
    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o"
    openai_text_model: str = "gpt-4o-mini"
    openai_whisper_model: str = "whisper-1"

    # Google Cloud — leave empty if not used
    google_application_credentials: str = ""
    google_cloud_project: str = ""

    # HuggingFace — no key required
    hf_token: str = ""

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def openai_available(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def google_available(self) -> bool:
        return bool(self.google_application_credentials or self.google_cloud_project)

    @property
    def huggingface_available(self) -> bool:
        return True  # Always available — runs locally

    def provider_status(self) -> dict[str, bool]:
        return {
            "openai": self.openai_available,
            "google": self.google_available,
            "huggingface": self.huggingface_available,
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
