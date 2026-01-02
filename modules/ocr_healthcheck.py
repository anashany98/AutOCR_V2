#!/usr/bin/env python3
"""
AutOCR_V2 OCR Environment Health Check
--------------------------------------
Runs at container startup to verify:
 - PaddlePaddle installation and GPU capability
 - PaddleOCR model loading (document + textline)
 - EasyOCR fallback
 - Poppler/pdf2image presence
 - Optional CUDA detection for Paddle/EasyOCR
"""

import sys
import shutil
from loguru import logger


def check_paddle():
    try:
        import paddle
        version = paddle.__version__
        cuda = paddle.is_compiled_with_cuda()
        devices = getattr(paddle.device.cuda, "device_count", lambda: 0)()
        logger.info(f"✅ PaddlePaddle {version} | CUDA built: {cuda} | GPUs detected: {devices}")
        return True
    except Exception as e:
        logger.error(f"❌ PaddlePaddle check failed: {e}")
        return False


def check_paddleocr():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="en", use_angle_cls=True)
        _ = ocr.ocr("tests/sample_invoice.png")
        logger.info("✅ PaddleOCR functional and model initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ PaddleOCR test failed: {e}")
        return False


def check_easyocr():
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        reader = easyocr.Reader(["en", "es"], gpu=use_gpu)
        _ = reader.readtext("tests/sample_invoice.png")
        logger.info(f"✅ EasyOCR operational (GPU: {use_gpu}).")
        return True
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR check failed or GPU unavailable: {e}")
        return False


def check_pdf2image():
    try:
        from pdf2image import convert_from_path
        if shutil.which("pdfinfo") is None:
            raise FileNotFoundError("Poppler not installed or missing in PATH.")
        pages = convert_from_path("tests/pdf-test.pdf", first_page=1, last_page=1)
        logger.info(f"✅ Poppler/pdf2image functional: {len(pages)} page(s) converted.")
        return True
    except Exception as e:
        logger.error(f"❌ PDF2Image check failed: {e}")
        return False


def run_healthcheck():
    logger.info("🔍 Running AutOCR_V2 OCR environment diagnostic...")
    results = {
        "paddle": check_paddle(),
        "paddleocr": check_paddleocr(),
        "easyocr": check_easyocr(),
        "pdf2image": check_pdf2image(),
    }

    passed = all(results.values())
    summary = "✅ PASSED" if passed else "❌ FAILED"
    logger.info(f"🧩 OCR health summary: {summary}")

    for k, v in results.items():
        logger.info(f" - {k:12s}: {'OK' if v else 'FAIL'}")

    if not passed:
        logger.warning("⚠️ Some OCR backends failed to initialise correctly. Check logs.")
    return passed


if __name__ == "__main__":
    run_healthcheck()
