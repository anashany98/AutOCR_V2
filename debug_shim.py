import os
import sys
from pathlib import Path

print("--- FINAL SHIM DIAGNOSTIC START ---")
print(f"Python: {sys.version}")

# Add shim directory explicitly
shim_dir = Path(r"C:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\venv311\Lib\site-packages\paddle\libs")
if shim_dir.exists():
    print(f"Adding Shim DLL directory: {shim_dir}")
    os.add_dll_directory(str(shim_dir))

# Add torch libs as usual
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.add_dll_directory(str(torch_lib))

try:
    print("Attempting to import paddle...")
    import paddle
    paddle.utils.run_check()
    print("Paddle device check successful with SHIM!")
except Exception:
    import traceback
    traceback.print_exc()

print("--- FINAL SHIM DIAGNOSTIC END ---")
