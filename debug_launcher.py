import sys
import os
import logging
import traceback

# Set universal fix for OpenMP conflicts just in case
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO)

print("--- DIAGNOSTIC START ---")

if os.name == "nt":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    print(f"CUDA bin path: {cuda_bin}")
    if os.path.exists(cuda_bin):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(cuda_bin)
                print("Added CUDA bin to os.add_dll_directory")
            except Exception as e:
                print(f"Failed to add CUDA bin: {e}")

# Check Torch Lib
try:
    from pathlib import Path
    possible_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    print(f"Checking Torch Lib at: {possible_torch_lib}")
    if possible_torch_lib.exists():
        if hasattr(os, "add_dll_directory"):
             os.add_dll_directory(str(possible_torch_lib))
             print("Added Torch lib to os.add_dll_directory")
        # List first 5 files
        files = list(possible_torch_lib.glob("*.dll"))[:5]
        print(f"Found {len(files)} DLLs in torch/lib, e.g. {[f.name for f in files]}")
    else:
        print("Torch lib directory NOT FOUND.")
except Exception as e:
    print(f"Error checking torch lib: {e}")

# TEST 1: TORCH IMPORT
print("\n[TEST 1] Attempting 'import torch'...")
try:
    import torch
    print(f"✅ Torch imported. Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Torch Import Failed: {e}")
except OSError as e:
    print(f"❌ Torch OS Error (DLL?): {e}")
    # traceback.print_exc()

# TEST 2: PADDLE IMPORT
print("\n[TEST 2] Attempting 'import paddle'...")
try:
    import paddle
    print(f"✅ Paddle imported. Version: {paddle.__version__}")
    device = paddle.device.get_device()
    print(f"   Device: {device}")
except ImportError as e:
    print(f"❌ Paddle Import Failed: {e}")
except OSError as e:
    print(f"❌ Paddle OS Error (DLL?): {e}")
    traceback.print_exc()

# TEST 3: SINGLETON
print("\n[TEST 3] Calling Singleton...")
try:
    from modules.paddle_singleton import get_ppstructure_v3_instance
    instance = get_ppstructure_v3_instance()
    if instance:
        print("✅ Singleton SUCCESS")
    else:
        print("❌ Singleton returned None")
except Exception as e:
    print(f"❌ Singleton Exception: {e}")
    traceback.print_exc()

print("--- DIAGNOSTIC END ---")
