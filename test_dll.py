import os
import ctypes
from pathlib import Path

print("--- CTYPES DLL TEST START ---")

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
dll_path = os.path.join(cuda_bin, "cudnn64_8.dll")

if os.name == "nt":
    print(f"Adding DLL directory: {cuda_bin}")
    os.add_dll_directory(cuda_bin)

print(f"Testing existence of {dll_path}: {os.path.exists(dll_path)}")

try:
    print(f"Attempting to load {dll_path} via ctypes.WinDLL...")
    lib = ctypes.WinDLL(dll_path)
    print("SUCCESS: DLL loaded successfully!")
except Exception as e:
    print(f"FAILURE: Could not load DLL: {e}")
    # Try to load it without full path to see if it's in the search path
    try:
        print("Attempting to load 'cudnn64_8.dll' from search path...")
        lib = ctypes.WinDLL("cudnn64_8.dll")
        print("SUCCESS: DLL loaded from search path!")
    except Exception as e2:
         print(f"FAILURE: Could not load from search path: {e2}")

print("--- CTYPES DLL TEST END ---")
