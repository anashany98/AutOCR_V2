import torch
import sys
import os
from pathlib import Path

# Fix paths for local imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.engines.paddle_vl_wrapper import PaddleVLOCHEngine
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestVLM")

def test_vlm_validation():
    print("--- Testing VLM Visual Validation ---")
    
    config = {
        "enabled": True,
        "use_gpu": True,
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.5"
    }
    
    engine = PaddleVLOCHEngine(config, logger=logger)
    
    # Text to validate (Intentional error)
    incorrect_text = "This is a test of a completely different text that is not in the image."
    correct_text = "ocr" # The dummy image is a blue block, but let's see what happens
    
    # Create a dummy image with some text-like features if possible
    # For now, let's use the same dummy image strategy as before
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    
    print("\n[Scenario 1] Validating INCORRECT text...")
    result1 = engine.visual_validate(img, incorrect_text)
    print(f"Match: {result1.get('is_valid')}")
    print(f"Score: {result1.get('score')}")
    print(f"Feedback: {result1.get('feedback')}")
    print(f"Raw: {result1.get('raw_response')}")

    print("\n[Scenario 2] Validating CORRECT (empty/placeholder) text...")
    result2 = engine.visual_validate(img, correct_text)
    print(f"Match: {result2.get('is_valid')}")
    print(f"Score: {result2.get('score')}")
    print(f"Feedback: {result2.get('feedback')}")
    print(f"Raw: {result2.get('raw_response')}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_vlm_validation()
    else:
        print("CUDA not available, but VLM engine will fallback to CPU automatically.")
        test_vlm_validation()
