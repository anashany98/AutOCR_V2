from .base import OCREngine
from typing import Tuple, Optional, Any, Sequence, Dict
from PIL import Image
import torch
import numpy as np
import logging
import os

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

class PaddleVLOCHEngine(OCREngine):
    """
    OCR Engine wrapper for PaddleOCR-VL-1.5 using Hugging Face Transformers.
    Bypasses the need for the native PaddlePaddle library on Windows by using PyTorch.
    """
    
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)
        self.model_id = config.get("model_id", "PaddlePaddle/PaddleOCR-VL-1.5")
        self.device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def initialize(self) -> bool:
        if not self.enabled:
            return False
            
        if AutoModel is None:
            if self.logger:
                self.logger.error("Transformers library not available for PaddleVLOCHEngine")
            return False

        try:
            self.logger.info(f"Loading PaddleOCR-VL-1.5 ({self.model_id}) on {self.device}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # Try loading with requested device first
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None
                ).eval()
                
                # Test a dummy generation to ensure CUDA kernels are working (catch sm_120 issues)
                if self.device == "cuda":
                    dummy_input = torch.zeros((1, 3, 224, 224), device="cuda", dtype=self.torch_dtype)
                    # This check is model-specific, but loading is usually enough. 
                    # If inference fails later, we catch it there too.
                    
            except Exception as e:
                if "no kernel image" in str(e) or "CUDA error" in str(e):
                    self.logger.warning(f"CUDA initialization failed ({e}). Falling back to CPU.")
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=self.torch_dtype,
                        device_map=None
                    ).eval()
                else:
                    raise e
            
            # PaddleOCR-VL often defines a build_processor method
            if hasattr(self.model, "build_processor"):
                self.processor = self.model.build_processor(self.tokenizer)
            else:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                
            self.logger.info("PaddleOCR-VL-1.5 loaded successfully.")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize PaddleOCR-VL: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            return False

    def extract_text(self, image: Image.Image) -> Tuple[str, float]:
        if not self.model or not self.processor:
            if not self.initialize():
                return "", 0.0
        
        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # The PaddleOCR-VL prompt format
            prompt = "User: <|IMAGE_PLACEHOLDER|>\nocr\nAssistant: "
            
            # Prepare inputs
            # The exact call depends on the processor implementation of PaddleOCR-VL
            # Usually: processor(text=prompt, images=image, return_tensors="pt")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # For generative models, we use a high confidence if it produced non-empty text
            confidence = 0.98 if result.strip() else 0.0
            return result, confidence
            
        except Exception as e:
            # Runtime CUDA error fallback
            if self.device == "cuda" and ("no kernel image" in str(e) or "CUDA error" in str(e)):
                 if self.logger:
                     self.logger.warning(f"Runtime CUDA error: {e}. Switching to CPU for future requests.")
                 self.device = "cpu"
                 self.model = self.model.to("cpu")
                 # Retry on CPU
                 try:
                     inputs = inputs.to("cpu")
                     with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            do_sample=False,
                            temperature=0.0
                        )
                     result = self.tokenizer.decode(output[0], skip_special_tokens=True)
                     confidence = 0.98 if result.strip() else 0.0
                     return result, confidence
                 except Exception as retry_e:
                     if self.logger:
                        self.logger.error(f"Fallback CPU extraction failed: {retry_e}")
                     return "", 0.0

            if self.logger:
                self.logger.error(f"PaddleOCR-VL extraction failed: {e}")
            return "", 0.0

    def extract_block(self, image: Image.Image, bbox: Sequence[int]) -> Tuple[str, float]:
        # Crop image to bbox before extraction
        left, top, right, bottom = bbox
        width, height = image.size
        # Clamp coordinates
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        
        if right <= left or bottom <= top:
            return "", 0.0
            
        crop = image.crop((left, top, right, bottom))
        return self.extract_text(crop)

    def visual_validate(self, image: Image.Image, text_to_validate: str) -> Dict[str, Any]:
        """
        Uses the VLM to validate if the provided text matches the image content.
        Returns a validation result with score and feedback.
        """
        if not self.model or not self.processor:
            if not self.initialize():
                return {"error": "Initialization failed", "match": False, "score": 0.0}

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Craft a validation prompt
            # We truncate the text to validate to avoid context issues
            truncated_text = text_to_validate[:500] if text_to_validate else "[Texto vacío]"
            
            prompt = (
                f"User: <|IMAGE_PLACEHOLDER|>\n"
                f"Validate if this OCR text accurately represents the image content: \"{truncated_text}\".\n"
                f"Provide feedback on discrepancies. "
                f"Return ONLY a JSON object: {{\"is_valid\": boolean, \"confidence_score\": float, \"discrepancies\": string|null}}\n"
                f"Assistant: "
            )

            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract JSON from response
            import json
            import re
            
            # Simple regex to find JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return {
                        "success": True,
                        "is_valid": result.get("is_valid", False),
                        "score": result.get("confidence_score", 0.0),
                        "feedback": result.get("discrepancies", ""),
                        "raw_response": response
                    }
                except:
                    pass
            
            return {
                "success": True,
                "is_valid": "true" in response.lower(),
                "score": 0.5,
                "feedback": "Could not parse JSON response",
                "raw_response": response
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"VLM Validation failed: {e}")
            return {"success": False, "error": str(e), "is_valid": False, "score": 0.0}

    def chat(self, image: Image.Image, question: str) -> str:
        """
        Perform Visual Question Answering (VQA) on the image.
        """
        if not self.model or not self.processor:
            if not self.initialize():
                return "Error: Vision model could not be initialized."
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            prompt = f"User: <|IMAGE_PLACEHOLDER|>\n{question}\nAssistant: "
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the prompt from the response if included (model dependent)
            if response.startswith(prompt):
                 response = response[len(prompt):]
            elif "Assistant: " in response:
                 response = response.split("Assistant: ")[-1]
                 
            return response.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(f"VLM Chat failed: {e}")
            return f"Error analyzing image: {str(e)}"

    def shutdown(self):
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
