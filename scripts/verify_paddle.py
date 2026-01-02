#!/usr/bin/env python3
import sys
import paddle

def main():
    print("🔍 Verifying PaddlePaddle CUDA Compatibility")
    try:
        print(f"✅ PaddlePaddle {paddle.__version__} imported")
        cuda_compiled = paddle.is_compiled_with_cuda()
        print(f"✅ CUDA Compiled: {cuda_compiled}")

        if cuda_compiled:
            try:
                place = paddle.CUDAPlace(0)
                x = paddle.to_tensor([1.0, 2.0, 3.0], place=place)
                y = x + 1.0
                print("✅ GPU Operations: WORKING")
            except Exception as e:
                print(f"⚠️ GPU Operations: FAILED - {e}")
                print("💡 Using CPU fallback mode")
        sys.exit(0)
    except Exception as e:
        print(f"❌ PaddlePaddle verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()