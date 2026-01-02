"""
Automatic OCR self-test executed during application startup.
"""

from __future__ import annotations

import os

from loguru import logger

from .ocr_cascade import MultiOCR


def run_startup_test() -> None:
    """
    Run a quick OCR pass on the bundled sample invoice.
    """
    sample = os.path.join("tests", "sample_invoice.png")
    if not os.path.exists(sample):
        logger.warning("⚠️ Startup test skipped (sample file missing).")
        return

    try:
        ocr = MultiOCR()
        result = ocr.run(sample)
        text = result[0].get("text", "") if result else ""
        if text.strip():
            logger.success("✅ OCR self-test passed. Extracted text:")
            logger.info(text[:200])
        else:
            logger.error("❌ OCR self-test produced empty result.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("❌ OCR self-test failed: {}", exc)


__all__ = ["run_startup_test"]

