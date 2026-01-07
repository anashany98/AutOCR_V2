
import torch
try:
    import paddle
except ImportError:
    paddle = None

print("-" * 30)
print("GPU VERIFICATION REPORT")
print("-" * 30)

# PyTorch Check
print(f"PyTorch Version: {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"Torch CUDA Available: {gpu_available}")
if gpu_available:
    print(f"Torch Device Count: {torch.cuda.device_count()}")
    print(f"Torch Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Torch is running on CPU.")

print("-" * 30)

# Paddle Check
if paddle:
    print(f"Paddle Version: {paddle.__version__}")
    print(f"Paddle Device: {paddle.device.get_device()}")
    try:
        paddle.utils.run_check()
        print("Paddle run_check() passed successfully.")
    except Exception as e:
        print(f"Paddle run_check() failed: {e}")
else:
    print("PaddlePaddle is NOT installed.")

print("-" * 30)
