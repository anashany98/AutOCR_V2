"""
Singleton helper for PaddleOCR PP-Structure engines.

Standardized for PaddleOCR 3.x (PPStructureV3).
"""

from __future__ import annotations

import os
import threading
import sys
from typing import Optional

import traceback
from loguru import logger

# Windows-specific DLL handling for PyTorch/PaddleOCR
if os.name == "nt":
    try:
        import sys
        import ctypes
        from pathlib import Path
        
        # Identify the probable location of the torch DLLs
        possible_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        if possible_torch_lib.exists():
            os.add_dll_directory(str(possible_torch_lib))
            
        # Add CUDA bin directory for cuDNN (PaddleOCR fix for error 126)
        possible_cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")
        if possible_cuda_bin.exists():
            os.add_dll_directory(str(possible_cuda_bin))
            # Pre-load critical dependencies to ensure they are in memory
            try:
                ctypes.WinDLL(str(possible_cuda_bin / "zlibwapi.dll"))
                ctypes.WinDLL(str(possible_cuda_bin / "cudnn64_8.dll"))
            except Exception:
                pass
    except Exception:
        pass

# Singleton state
_pp_instance = None
_pp_lock = threading.Lock()

def get_ppstructure_v3_instance():
    """
    Return a process-wide PPStructureV3 instance.
    GPU is selected via paddle.set_device("gpu").
    Returns None if initialization or dependency loading fails.
    """
    global _pp_instance

    if _pp_instance is not None:
        return _pp_instance

    with _pp_lock:
        if _pp_instance is not None:
            return _pp_instance

        try:
            # Explicit import of required backend
            import paddle
            
            # CRITICAL: Prevent paddleocr from trying to import torch and crashing due to bad DLLs
            # We want it to use Paddle only.
            if "torch" not in sys.modules:
                 sys.modules["torch"] = None

            from paddleocr import PPStructureV3 # type: ignore
            
            # Set environment flags for cleaner execution
            os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")
            
            # Explicit GPU selection as requested
            try:
                paddle.set_device("gpu")
                logger.info("Paddle device successfully set to GPU.")
            except Exception as e:
                logger.warning("GPU selection failed, continuing with default device: {}", e)

            # Clean initialization of PPStructureV3
            # No legacy hacks or artificial version detection
            logger.info("Initializing PPStructureV3 engine...")
            _pp_instance = PPStructureV3()
            logger.info("PPStructureV3 engine loaded successfully.")
            
        except (ImportError, OSError) as e:
            # Handle PyTorch/DLL failure as a soft-fail (returns None)
            logger.error("Structural engine unavailable due to missing or broken dependencies: {}", e)
            print("--- PADDLE SINGLETON SPLASH TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------------------------")
            _pp_instance = None
        except Exception as e:
            logger.error("Runtime error during PPStructureV3 initialization: {}", e)
            print("--- PADDLE SINGLETON RUNTIME TRACEBACK ---")
            traceback.print_exc()
            print("------------------------------------------")
            _pp_instance = None

    return _pp_instance

__all__ = ["get_ppstructure_v3_instance"]
