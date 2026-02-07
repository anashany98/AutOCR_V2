import logging
import sys
import os
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from modules.engines.florence_wrapper import FlorenceOCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFlorence")

def test_florence():
    logger.info("Starting Florence-2 Verification...")
    
    # We'll use a placeholder or any image in the project
    # Searching for an image
    test_image = "web_app/static/img/logo.png" # Safe fallback if exists
    if not os.path.exists(test_image):
        # Let's find any image
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(root, file)
                    break
            if test_image: break

    if not os.path.exists(test_image):
        logger.error("No test image found in the project.")
        return

    logger.info(f"Testing with image: {test_image}")
    
    try:
        engine = FlorenceOCREngine()
        # Test detection
        logger.info("Running Object Detection (<OD>)...")
        with Image.open(test_image) as img:
            results = engine.run_task(img.convert("RGB"), "<OD>")
            logger.info(f"Detection Results: {results}")

        # Test captioning
        logger.info("Running Dense Captioning (<DENSE_REGION_CAPTION>)...")
        caption = engine.caption_image(test_image)
        logger.info(f"Caption: {caption}")
        
        logger.info("✅ Florence-2 Verification PASSED.")
    except Exception as e:
        logger.error(f"❌ Florence-2 Verification FAILED: {e}")

if __name__ == "__main__":
    test_florence()
