import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class FlorenceOCREngine:
    """
    Wrapper for Microsoft's Florence-2 model.
    Handles Object Detection, Captioning, and specialized OCR.
    """
    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: str = None):
        self.model_id = model_id
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = None
        self.model = None
        self._loaded = False

    def load(self):
        """Lazy load the model to save memory."""
        if self._loaded and self.model.device.type == self.device:
            return
        
        # Request VRAM from orchestrator
        from modules.resource_orchestrator import ResourceOrchestrator
        ResourceOrchestrator().request_model("florence")
        ResourceOrchestrator().register_model("florence", self)

        logger.info(f"Loading Florence-2 ({self.model_id}) on {self.device}...")
        try:
            if not self.model:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True
                ).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            else:
                self.model.to(self.device)
            
            self._loaded = True
            logger.info("Florence-2 loaded successfully.")
        except Exception as e:
            if self.device == "cuda":
                logger.warning(f"Failed to load Florence-2 on GPU: {e}. Falling back to CPU.")
                self.device = "cpu"
                self.load()
            else:
                logger.error(f"Failed to load Florence-2: {e}")
                raise

    def unload(self):
        """Move to CPU to free VRAM."""
        if self.model:
            logger.info("Moving Florence-2 to CPU...")
            self.model.to("cpu")
            self._loaded = False

    def run_task(self, image: Image.Image, task_prompt: str, text_input: str = None) -> Dict[str, Any]:
        """Generic task runner for Florence-2 prompts."""
        self.load()
        
        if text_input:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """Detect generic objects (<OD>)."""
        with Image.open(image_path) as img:
            return self.run_task(img.convert("RGB"), "<OD>")

    def caption_image(self, image_path: str, dense: bool = True) -> str:
        """Generate a description of the image content (<DENSE_REGION_CAPTION>)."""
        task = "<DENSE_REGION_CAPTION>" if dense else "<CAPTION>"
        with Image.open(image_path) as img:
            result = self.run_task(img.convert("RGB"), task)
            # Post-process result to return a string
            if task == "<DENSE_REGION_CAPTION>":
                return result.get("<DENSE_REGION_CAPTION>", "")
            return result.get("<CAPTION>", "")

    def detect_furniture_and_materials(self, image_path: str) -> Dict[str, Any]:
        """Specialized task combining detection and description."""
        # For now, we use OD and filter, or use a custom prompt if supported
        # Florence-2 is good at captioning regions
        od_results = self.detect_objects(image_path)
        caption = self.caption_image(image_path)
        
        return {
            "objects": od_results.get("<OD>", []),
            "description": caption
        }
