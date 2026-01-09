import os
import sys
from pathlib import Path

print("--- DIAGNOSTIC START ---")
print(f"Python: {sys.version}")
print(f"OS: {os.name}")

# Add DLL directory for torch if on Windows
if os.name == "nt":
    # Add torch libs
    possible_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if possible_torch_lib.exists():
        print(f"Adding Torch DLL directory: {possible_torch_lib}")
        os.add_dll_directory(str(possible_torch_lib))
        
    # Add CUDA bin directory (CRITICAL FIX)
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    if os.path.exists(cuda_bin):
        print(f"Adding CUDA DLL directory: {cuda_bin}")
        os.add_dll_directory(cuda_bin)
    else:
        print(f"WARNING: CUDA bin directory not found at {cuda_bin}")

try:
    print("Attempting to import paddle...")
    import paddle
    print(f"Paddle version: {getattr(paddle, '__version__', 'unknown')}")
    paddle.utils.run_check()
    print("Paddle device check successful.")
    
    print("Attempting to import PPStructureV3...")
    from paddleocr import PPStructureV3
    print("Import successful. Initializing engine...")
    
    os.environ["PADDLEOCR_DISABLE_VLM"] = "1"
    paddle.set_device("gpu")
    
    engine = PPStructureV3()
    print("PPStructureV3 initialized successfully!")
    
except Exception as e:
    print("\n!!! ERROR DETECTED !!!")
    import traceback
    traceback.print_exc()

print("--- DIAGNOSTIC END ---")
