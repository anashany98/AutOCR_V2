#!/usr/bin/env python3
"""
Test script to debug PPStructureV3 initialization issues.
"""

import os
import sys
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

def test_ppstructure_initialization():
    """Test PPStructureV3 initialization with detailed logging."""
    logger.info("=== PPStructureV3 Initialization Test ===")

    # Check imports
    try:
        from paddleocr import PPStructureV3
        logger.success("✅ PPStructureV3 import successful")
    except ImportError as e:
        logger.error("❌ PPStructureV3 import failed: {}", e)
        return False

    # Check PaddlePaddle
    try:
        import paddle
        logger.info("PaddlePaddle version: {}", paddle.__version__)
        logger.info("CUDA compiled: {}", paddle.is_compiled_with_cuda())
        if paddle.is_compiled_with_cuda():
            logger.info("CUDA device count: {}", paddle.device.cuda.device_count())
            logger.info("Current device: {}", paddle.get_device())
    except Exception as e:
        logger.error("❌ PaddlePaddle check failed: {}", e)
        return False

    # Set environment variables
    os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logger.info("Environment variables set: PADDLEOCR_DISABLE_VLM=1, CUDA_VISIBLE_DEVICES=0")

    # Try initialization
    try:
        logger.info("Attempting PPStructureV3()...")
        instance = PPStructureV3()
        logger.success("✅ PPStructureV3 initialized successfully!")
        return True
    except Exception as e:
        logger.error("❌ PPStructureV3 initialization failed: {}", e)
        logger.error("Exception type: {}", type(e).__name__)
        logger.error("Full traceback:\n{}", traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_ppstructure_initialization()
    sys.exit(0 if success else 1)