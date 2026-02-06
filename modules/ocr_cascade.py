"""
Multi-engine OCR cascade with GPU auto-detection.

The cascade executes PaddleOCR first, then EasyOCR and finally
Tesseract as a last resort.  This keeps compatibility with legacy
components while offering resilience when individual engines fail.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from loguru import logger
from PIL import Image

from .paddle_singleton import get_ppstructure_v3_instance

try:
    import easyocr  # type: ignore
except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore[assignment]
    logger.error("❌ EasyOCR import failed: {}", exc)

try:
    import pytesseract  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]
    logger.error("❌ pytesseract import failed: {}", exc)

try:
    import torch  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class MultiOCR:
    """
    Run OCR engines in a cascade: PaddleOCR → EasyOCR → Tesseract.
    """

    def __init__(self, langs: Sequence[str] | None = None) -> None:
        self.langs = list(langs) if langs else ["en", "es"]
        self.gpu_available = bool(torch and torch.cuda.is_available())  # type: ignore[union-attr]
        logger.info("🧠 GPU available: {}", self.gpu_available)
        logger.info("🔤 OCR cascade languages: {}", self.langs)

        self._paddle_ocr = None

        if easyocr is None:
            raise ImportError("EasyOCR is required for the OCR cascade.")
        self.easy = easyocr.Reader(self.langs, gpu=self.gpu_available)  # type: ignore[misc]
        logger.info("✅ EasyOCR initialized.")

    def run(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Execute the OCR cascade and return normalised results.
        """
        # Lazy-load Paddle to avoid import errors at module load time
        if self._paddle_ocr is None:
            try:
                self._paddle_ocr = get_ppstructure_v3_instance()
            except Exception as exc:
                logger.warning("Could not initialize PaddleOCR: {}", exc)
                self._paddle_ocr = False  # Mark as failed, don't retry

        if self._paddle_ocr:
            try:
                logger.info("▶️ Running PaddleOCR (PPStructureV3) on {}", image_path)
                # PPStructureV3 returns a list of blocks
                results = self._paddle_ocr(image_path)
                if results:
                    texts = []
                    for block in results:
                        res = block.get("res")
                        if res and isinstance(res, list):
                            for item in res:
                                if isinstance(item, (list, tuple)) and len(item) == 2:
                                    # item: [box, (text, score)]
                                    data = item[1]
                                    if data and isinstance(data, (list, tuple)):
                                        text = (str(data[0]) or "").strip()
                                        if text:
                                            texts.append(text)
                    
                    if texts:
                        logger.success("📄 PaddleOCR (Structural) succeeded.")
                        return [{"text": "\n".join(texts)}]
            except Exception as exc:  # pragma: no cover - Paddle runtime errors
                logger.warning("⚠️ PaddleOCR failed: {}", exc)
                logger.opt(exception=exc).debug("PaddleOCR exception stacktrace")

        try:
            logger.info("▶️ Running EasyOCR on {}", image_path)
            text_blocks = self.easy.readtext(image_path, detail=0)
            if text_blocks:
                logger.success("📄 EasyOCR succeeded.")
                return [{"text": "\n".join(text_blocks)}]
        except Exception as exc:  # pragma: no cover - EasyOCR runtime errors
            logger.warning("⚠️ EasyOCR failed: {}", exc)

        if pytesseract is not None:
            try:
                logger.info("▶️ Running Tesseract on {}", image_path)
                with Image.open(image_path) as image:
                    text = pytesseract.image_to_string(image, lang="spa+eng")
                if text.strip():
                    logger.success("📄 Tesseract succeeded.")
                    return [{"text": text}]
            except Exception as exc:  # pragma: no cover - pytesseract runtime errors
                logger.error("❌ All OCR engines failed: {}", exc)

        logger.error("❌ OCR cascade produced empty result.")
        return [{"text": ""}]


__all__ = ["MultiOCR"]
