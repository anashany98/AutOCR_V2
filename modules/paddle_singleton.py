"""
Singleton helper for PaddleOCR PP-Structure engines.

The paddle runtime raises a ``RuntimeError`` when PP-Structure is
initialised more than once within the same process. This module
exposes ``get_paddle_ocr`` to lazily create and reuse a single
instance across the application.
"""

from __future__ import annotations

import os
import platform
from typing import Dict

from loguru import logger

try:
    from paddleocr import PPStructureV3  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    PPStructureV3 = None  # type: ignore[assignment]
    logger.warning(
        "PaddleOCR import failed; PP-Structure features disabled. "
        "Ensure paddleocr>=3.3 and paddlepaddle>=3.2 are installed. Details: %s",
        exc,
    )

_pp_instance = None


def _log_env_snapshot() -> None:
    """Emit the environment variables that influence Paddle runtime."""
    tracked_keys = (
        "CUDA_VISIBLE_DEVICES",
        "PADDLEOCR_DISABLE_VLM",
        "FLAGS_use_cuda",
        "FLAGS_fraction_of_gpu_memory_to_use",
    )
    snapshot: Dict[str, str] = {key: os.getenv(key, "<unset>") for key in tracked_keys}
    logger.info("Paddle env snapshot: {}", snapshot)


def _log_platform_snapshot() -> None:
    """Record host/platform details once per boot to aid debugging."""
    logger.info(
        "Runtime platform: python {} on {} {}",
        platform.python_version(),
        platform.system(),
        platform.release(),
    )


def _log_paddle_runtime() -> None:
    """Best-effort introspection of the Paddle runtime state."""
    try:
        import paddle

        logger.info("PaddlePaddle version: {}", paddle.__version__)
        logger.info("Paddle compiled with CUDA: {}", paddle.is_compiled_with_cuda())
        if paddle.is_compiled_with_cuda():
            logger.info("CUDA device count: {}", paddle.device.cuda.device_count())
            logger.info("Current Paddle device: {}", paddle.get_device())
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("Could not log Paddle runtime context: {}", exc)


def get_paddle_ocr():
    """
    Return a process-wide PaddleOCR PP-Structure instance.
    """
    global _pp_instance

    if _pp_instance is None:
        if PPStructureV3 is None:  # pragma: no cover - defensive
            return None
        try:
            # PPStructureV3 may have compatibility issues - try basic initialization first
            logger.info("Attempting to initialize PPStructureV3...")
            _log_platform_snapshot()
            _log_env_snapshot()
            _log_paddle_runtime()

            # Set environment variables that might help
            os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")  # Disable VLM components
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Ensure GPU 0 is visible
            logger.info("Post-set env snapshot (may include defaults applied):")
            _log_env_snapshot()

            _pp_instance = PPStructureV3()
            logger.info("PaddleOCR (PPStructureV3) loaded successfully.")
        except RuntimeError as exc:
            message = str(exc)
            if "PDX has already been initialized" in message:
                logger.warning("PaddleOCR already initialised; using existing instance.")
            else:  # pragma: no cover - pass through unexpected runtime errors
                logger.error("Failed to initialize PPStructureV3: {}", exc)
                logger.error("Exception type: {}", type(exc).__name__)
                import traceback
                logger.error("Full traceback:\n{}", traceback.format_exc())
                logger.warning("PPStructureV3 failed, layout/table detection will fallback to naive methods")
                return None
        except Exception as exc:
            logger.error("Unexpected error initializing PPStructureV3: {}", exc)
            logger.error("Exception type: {}", type(exc).__name__)
            import traceback
            logger.error("Full traceback:\n{}", traceback.format_exc())
            logger.warning("PPStructureV3 failed, layout/table detection will fallback to naive methods")
            return None
    else:
        logger.debug("Reusing cached PPStructureV3 instance.")
    return _pp_instance


__all__ = ["get_paddle_ocr"]
