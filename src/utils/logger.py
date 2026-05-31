"""
utils/logger.py — Structured Logging
──────────────────────────────────────
Uses structlog to produce consistent, machine-readable log output.

In development (LOG_LEVEL=DEBUG): pretty coloured console output
In production (LOG_LEVEL=INFO):   JSON output, one line per event

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("analysis.start", provider="openai", file="photo.jpg")
    log.error("analysis.failed", error=str(e))
"""

import logging
import sys
import structlog
from src.config import settings


def setup_logging() -> None:
    """
    Configure structlog. Call this once at application startup.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure standard library logging (used by uvicorn, etc.)
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout,
        format="%(message)s",
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if sys.stdout.isatty():
        # Pretty output for local development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON output for production / log aggregators
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a bound logger for a module."""
    return structlog.get_logger(name)
