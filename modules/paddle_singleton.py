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

def _env_truthy(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

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

        # Set env flags before importing paddleocr so import-time behavior is consistent.
        os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
        # Optional compatibility switch (off by default).
        os.environ.setdefault("AUTO_OCR_DISABLE_TORCH_IMPORT", "0")

        # --- PRE-FLIGHT SAFETY CHECK ---
        # PaddleOCR 2.6+ on Windows acts unstable with some CUDA versions/DLLs.
        # to prevent the MAIN process from hard-crashing (exit code 1), we test init in a disposable subprocess.
        if _pp_instance is None: # Only check once
            import subprocess
            try:
                # We interpret the result of a minimal python script
                test_script = (
                    "import os, sys\n"
                    "os.environ['PADDLEOCR_DISABLE_VLM'] = '1'\n"
                    "os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'\n"
                    "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'\n"
                    "from paddleocr import PPStructureV3\n"
                    "print('OK')\n"
                )
                test_cmd = [sys.executable, "-c", test_script]
                logger.info("Running PaddleOCR safety pre-flight check...")
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
                
                soft_failed = False
                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    soft_fail_markers = (
                        "torch\\lib\\shm.dll",
                        "modelscope",
                        "no kernel image",
                        "cuda error",
                    )
                    if any(marker in stderr.lower() for marker in soft_fail_markers):
                        soft_failed = True
                        logger.warning(
                            "PaddleOCR safety check reported a non-fatal optional dependency issue; "
                            "continuing with safe import mode."
                        )
                        logger.warning("Paddle pre-flight stderr (truncated): {}", stderr[:600])
                    else:
                        logger.error(
                            f"PaddleOCR Safety Check FAILED with code {result.returncode}. Disabling Paddle to prevent crash."
                        )
                        logger.error("Paddle pre-flight stderr (truncated): {}", stderr[:600])
                        _pp_instance = False # Marker for "Failed, do not retry"
                        return None
                    
                if soft_failed:
                    logger.warning("PaddleOCR safety pre-flight completed with non-fatal warnings.")
                else:
                    logger.info("PaddleOCR Safety Check PASSED.")

            except Exception as e:
                logger.error(f"PaddleOCR Safety Check Error: {e}")
                _pp_instance = False
                return None

        try:
            # Optional compatibility switch: some Windows setups hard-crash when *any* code imports torch/torchvision.
            # If you hit that, set AUTO_OCR_DISABLE_TORCH_IMPORT=1 to prevent paddleocr from importing torch at import-time.
            patched_torch = False
            if _env_truthy("AUTO_OCR_DISABLE_TORCH_IMPORT", "0") and "torch" not in sys.modules:
                sys.modules["torch"] = None
                patched_torch = True

            try:
                from paddleocr import PPStructureV3  # type: ignore
            finally:
                if patched_torch:
                    sys.modules.pop("torch", None)
            
            # Clean initialization of PPStructureV3
            # No legacy hacks or artificial version detection
            logger.info("Initializing PPStructureV3 engine...")
            _pp_instance = PPStructureV3()
            logger.info("PPStructureV3 engine loaded successfully.")
            
        except (ImportError, OSError) as e:
            # Handle PyTorch/DLL failure as a soft-fail (returns None)
            logger.error("Structural engine unavailable due to missing or broken dependencies: {}", e)
            logger.debug("PPStructureV3 import traceback:\n{}", traceback.format_exc())
            _pp_instance = None # Retryable? Maybe not.
        except Exception as e:
            logger.error("Runtime error during PPStructureV3 initialization: {}", e)
            logger.debug("PPStructureV3 runtime traceback:\n{}", traceback.format_exc())
            _pp_instance = None

    return _pp_instance if _pp_instance is not False else None

__all__ = ["get_ppstructure_v3_instance"]
