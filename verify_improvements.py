
import sys
import os
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def verify_improvements():
    print("Verifying improvements...")
    errors = []

    # 1. Verify Task Import
    try:
        from modules.tasks import rebuild_vision_index_task
        print("[OK] rebuild_vision_index_task imported.")
    except ImportError as e:
        errors.append(f"Failed to import rebuild_vision_index_task: {e}")

    # 2. Verify MinerU Logic
    try:
        from modules.engines.mineru_wrapper import get_mineru_engine
        engine = get_mineru_engine({})
        # Create dummy file
        with open("test_dummy.pdf", "wb") as f:
            f.write(b"%PDF-1.4 empty")
        
        is_complex = engine.is_complex_document("test_dummy.pdf")
        print(f"[OK] MinerU heuristic check ran. Result for dummy: {is_complex}")
        
    except Exception as e:
        errors.append(f"MinerU check failed: {e}")
    finally:
        if os.path.exists("test_dummy.pdf"):
            os.remove("test_dummy.pdf")

    # 3. Verify Route Existence (Static Check)
    try:
        from web_app.routes.main_routes import system_status
        print("[OK] system_status route function exists.")
    except ImportError:
        errors.append("system_status function not found in main_routes.")

    if errors:
        print("\nERRORS FOUND:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED.")

if __name__ == "__main__":
    verify_improvements()
