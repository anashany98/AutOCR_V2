import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
import sys

def test_paddlevl():
    model_id = "PaddlePaddle/PaddleOCR-VL-1.5"
    print(f"Testing {model_id}...")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Testing on CPU...")
        device = "cpu"
    else:
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        device = "cuda"

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        print("Loading model...")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).eval()
        
        print("Model loaded successfully.")
        
        # Test extraction
        # We'll use a blank image for sanity check
        img = Image.new('RGB', (100, 100), color = (73, 109, 137))
        
        if hasattr(model, "build_processor"):
            processor = model.build_processor(tokenizer)
        else:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
        prompt = "User: <|IMAGE_PLACEHOLDER|>\nocr\nAssistant: "
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        
        print("Running inference...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Inference successful. Output: '{result}'")
        return True
    except Exception as e:
        print(f"Error: {e}")
        if "no kernel image" in str(e) or "CUDA error" in str(e):
            print("\n[WARN] CUDA COMPATIBILITY ISSUE DETECTED")
            print("Your RTX 5070 (Blackwell) requires CUDA 12.8+, but installed PyTorch uses older kernels.")
            print("Falling back to CPU for development/testing...")
            
            # Fallback code
            device = "cpu"
            print(f"CUDA detected (fallback): {device}")
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map=None
            ).eval()
            print("Model loaded on CPU successfully.")
            
            print("Running inference on CPU...")
            image = img.convert("RGB")
            # For PaddleOCR-VL, the prompt often needs to include the image placeholder
            prompt = "User: <|IMAGE_PLACEHOLDER|>\nocr\nAssistant: "
            
            if hasattr(model, "build_processor"):
                processor = model.build_processor(tokenizer)
            else:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=64)
            
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\nResult (CPU): {result}")
            return True
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_paddlevl()
    sys.exit(0 if success else 1)
