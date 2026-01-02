import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_paddle")

def test_ppstructure():
    logger.info("Attempting to import paddleocr...")
    try:
        from paddleocr import PPStructure
    except ImportError as e:
        logger.error(f"Failed to import paddleocr: {e}")
        return

    logger.info("Initializing PPStructure(show_log=True, image_orientation=True)...")
    try:
        # Try compact init first
        table_engine = PPStructure(show_log=True, image_orientation=True)
        logger.info("✅ PPStructure initialized successfully!")
    except Exception as e:
        logger.error("❌ PPStructure failed to initialize.")
        logger.exception(e)
        
        # Check for common missing deps
        try:
            import shapely
            logger.info(f"Shapely version: {shapely.__version__}")
        except ImportError:
            logger.warning("⚠️ Shapely is NOT installed. This is often required for layout analysis.")

        try:
            import paddle
            logger.info(f"Paddle version: {paddle.__version__}")
        except ImportError:
            logger.error("PaddlePaddle is NOT installed.")

if __name__ == "__main__":
    test_ppstructure()
