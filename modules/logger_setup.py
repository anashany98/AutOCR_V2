"""
Centralised Loguru configuration for AutOCR.

When imported, this module configures Loguru to emit structured logs to both
stdout and a rotating file in ``logs/autocr.log``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from loguru import logger

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    colorize=True,
    enqueue=True,
)
logger.add(
    os.path.join(LOG_DIR, "autocr.log"),
    rotation="5 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    enqueue=True,
)
logger.info("🧩 Logger initialized.")


class InterceptHandler(logging.Handler):
    """
    Route standard logging records through Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:  # pragma: no cover - defensive
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)


__all__ = ["logger"]

