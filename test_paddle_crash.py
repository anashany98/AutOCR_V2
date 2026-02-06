
import os
import sys

# CRITICAL: Prevent paddleocr from trying to import torch and crashing due to bad DLLs
sys.modules["torch"] = None

# Allow multiple OpenMP runtimes (fixes shm.dll/libomp conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable model check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLEOCR_DISABLE_VLM"] = "1"

print("Importing paddle...")
import paddle
paddle.set_device("gpu")

print("Importing paddleocr...")
try:
    from paddleocr import PPStructureV3
    print("Initializing PPStructureV3...")
    engine = PPStructureV3()
    print("✅ Success!")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
