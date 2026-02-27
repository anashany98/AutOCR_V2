import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.services import get_vision_manager, get_logger

def test():
    print("Testing get_vision_manager...")
    try:
        vm = get_vision_manager()
        if vm:
            print("VisionManager loaded successfully via services.")
        else:
            print("VisionManager returned None.")
            # Check if logger has info
            # We can't easily check logger buffer here unless we mock it or read file
            print("Check logs for errors.")
            
    except Exception as e:
        print(f"CRITICAL: get_vision_manager raised {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
