import sys
from pathlib import Path
import os
import traceback

# Ensure modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def debug_paddle():
    try:
        print("Importing PaddleOCR...")
        from paddleocr import PPStructureV3
        import paddle
        
        print(f"Paddle version: {paddle.__version__}")
        print(f"Paddle compiled with CUDA: {paddle.is_compiled_with_cuda() if hasattr(paddle, 'is_compiled_with_cuda') else 'N/A'}")
        
        print("\nAttempting to initialize PPStructureV3(gpu_id=0)...")
        try:
            # Try with zero arguments first to see what happens
            instance = PPStructureV3()
            print("Successfully initialized PPStructureV3 with no arguments!")
        except Exception as e:
            print("\nInitialization failed with no arguments:")
            traceback.print_exc()

        print("\nAttempting to initialize PPStructureV3(gpu_id=0)...")
        try:
            instance = PPStructureV3(gpu_id=0)
            print("Successfully initialized PPStructureV3(gpu_id=0)!")
        except Exception as e:
            print("\nInitialization failed with gpu_id=0:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"\nOuter Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_paddle()
