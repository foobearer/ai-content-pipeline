"""
utils/logger.py — Structured Logging
──────────────────────────────────────
Pretty console output in dev, JSON in production.
"""

import logging
import sys
import structlog
from src.config import settings


def setup_logging() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        stream=sys.stdout, format="%(message)s")
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
    ]
    structlog.configure(processors=processors, wrapper_class=structlog.stdlib.BoundLogger,
                        context_class=dict, logger_factory=structlog.stdlib.LoggerFactory(),
                        cache_logger_on_first_use=True)


def get_logger(name: str):
    return structlog.get_logger(name)
