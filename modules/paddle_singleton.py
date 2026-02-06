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

        # --- PRE-FLIGHT SAFETY CHECK ---
        # PaddleOCR 2.6+ on Windows acts unstable with some CUDA versions/DLLs.
        # to prevent the MAIN process from hard-crashing (exit code 1), we test init in a disposable subprocess.
        if _pp_instance is None: # Only check once
            import subprocess
            try:
                # We interpret the result of a minimal python script
                test_cmd = [
                    sys.executable, "-c",
                    "import os; os.environ['DISABLE_MODEL_SOURCE_CHECK']='True'; "
                    "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'; "
                    "import paddle; paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu'); "
                    "from paddleocr import PPStructureV3; "
                    "PPStructureV3(show_log=False); print('OK')"
                ]
                logger.info("🛡️ Running PaddleOCR safety pre-flight check...")
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    logger.error(f"⚠️ PaddleOCR Safety Check FAILED with code {result.returncode}. Disabling Paddle to prevent crash.")
                    logger.error(f"Stderr: {result.stderr}")
                    _pp_instance = False # Marker for "Failed, do not retry"
                    return None
                    
                logger.info("✅ PaddleOCR Safety Check PASSED.")

            except Exception as e:
                logger.error(f"⚠️ PaddleOCR Safety Check Error: {e}")
                _pp_instance = False
                return None

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
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True") 
            os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
            
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
            _pp_instance = None # Retryable? Maybe not.
        except Exception as e:
            logger.error("Runtime error during PPStructureV3 initialization: {}", e)
            print("--- PADDLE SINGLETON RUNTIME TRACEBACK ---")
            traceback.print_exc()
            print("------------------------------------------")
            _pp_instance = None

    return _pp_instance if _pp_instance is not False else None

__all__ = ["get_ppstructure_v3_instance"]
