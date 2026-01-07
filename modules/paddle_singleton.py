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
    from paddleocr import PaddleOCR  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore[assignment]
    logger.warning(
        "PaddleOCR import failed; PP-Structure features disabled. "
        "Ensure paddleocr is installed. Details: %s",
        exc,
    )


import threading

_pp_instance = None
_pp_lock = threading.Lock()

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


def get_paddle_ocr(gpu_id: int = 0):
    """
    Return a process-wide PaddleOCR PP-Structure instance.
    Thread-safe implementation.
    """
    global _pp_instance

    if _pp_instance is None:
        if PaddleOCR is None:  # pragma: no cover - defensive
            return None
            
        with _pp_lock:
            # Double-check locking pattern
            if _pp_instance is not None:
                logger.debug("Reusing cached PPStructureV3 instance (acquired via lock).")
                return _pp_instance
                
            try:
                # PPStructureV3 may have compatibility issues - try basic initialization first
                logger.info("Attempting to initialize PPStructureV3...")
                _log_platform_snapshot()
                _log_env_snapshot()
                _log_paddle_runtime()

                # Set environment variables that might help
                os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")  # Disable VLM components
                # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # REMOVED: Let Paddle handle it or set externally
                logger.info("Post-set env snapshot (may include defaults applied):")
                _log_env_snapshot()

                # PPStructure is the correct class for layout/table analysis in newer PaddleOCR versions
                from paddleocr import PPStructure
                
                logger.info("Initializing PPStructure...")
                _pp_instance = PPStructure(gpu_id=gpu_id, use_gpu=True, show_log=False)
                logger.info("PPStructure loaded successfully.")
            except Exception as exc:
                logger.error("Unexpected error initializing PPStructure: {}", exc)
                logger.error("Exception type: {}", type(exc).__name__)
                # Fallback to CPU if GPU failed
                try:
                    from paddleocr import PPStructure
                    logger.info("Retrying PPStructure on CPU...")
                    _pp_instance = PPStructure(use_gpu=False, show_log=False)
                    return _pp_instance
                except Exception:
                    import traceback
                    logger.error("Full traceback:\n{}", traceback.format_exc())
                    logger.warning("Structure engine failed.")
                    return None
    else:
        logger.debug("Reusing cached PPStructureV3 instance.")
    return _pp_instance


__all__ = ["get_paddle_ocr"]
