#!/usr/bin/env python3
"""
AutOCR_V2 GPU/CPU OCR Health Check
-----------------------------------
This script runs during Docker startup or CI/CD to verify:
 - PaddlePaddle installation and GPU availability
 - PaddleOCR model loading (document + textline)
 - EasyOCR fallback functionality
 - Poppler / pdf2image PDF rendering
 - Basic OCR text extraction test

Author: AutOCR_V2 Diagnostic Module
"""

import os
import sys
import shutil
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

def check_paddle():
    try:
        import paddle
        version = paddle.__version__
        cuda = paddle.is_compiled_with_cuda()
        logger.info(f"✅ PaddlePaddle version {version}, GPU compiled: {cuda}")
        return True
    except Exception as e:
        logger.error(f"❌ PaddlePaddle check failed: {e}")
        return False

def check_paddleocr():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="en", use_angle_cls=True)
        _ = ocr.ocr("tests/sample_invoice.png", cls=True)
        logger.info("✅ PaddleOCR functional and models downloadable.")
        return True
    except Exception as e:
        logger.error(f"❌ PaddleOCR check failed: {e}")
        return False

def check_easyocr():
    try:
        import easyocr
        reader = easyocr.Reader(["en", "es"], gpu=True)
        _ = reader.readtext("tests/sample_invoice.png")
        logger.info("✅ EasyOCR functional and GPU accessible.")
        return True
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR fallback only (CPU or error): {e}")
        return False

def check_pdf2image():
    try:
        from pdf2image import convert_from_path
        if shutil.which("pdfinfo") is None:
            raise FileNotFoundError("Poppler (pdfinfo) not found in PATH")
        pages = convert_from_path("tests/pdf-test.pdf", first_page=1, last_page=1)
        logger.info(f"✅ Poppler/pdf2image functional: {len(pages)} page(s) converted.")
        return True
    except Exception as e:
        logger.error(f"❌ PDF2Image/Poppler check failed: {e}")
        return False

def run_all():
    logger.info("🔍 Running AutOCR_V2 environment health check...")
    results = {
        "paddle": check_paddle(),
        "paddleocr": check_paddleocr(),
        "easyocr": check_easyocr(),
        "pdf2image": check_pdf2image(),
    }

    passed = all(results.values())
    logger.info(f"🧩 Overall health: {'✅ PASSED' if passed else '❌ FAILED'}")
    for key, value in results.items():
        logger.info(f" - {key:12s}: {'OK' if value else 'FAIL'}")

    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    run_all()
