import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, Any, List, Optional
import os
from modules.torch_compat import torch_cuda_usable
from modules.color_extractor import ColorExtractor

logger = logging.getLogger(__name__)

# Furniture categories for classification
FURNITURE_CATEGORIES = {
    "seating": {
        "es": "Asientos",
        "en": "Seating",
        "items": ["sofa", "couch", "chair", "armchair", "seat", "stool", "bench", "ottoman", "banquette", "sillón", "taburete", "banco"]
    },
    "tables": {
        "es": "Mesas",
        "en": "Tables",
        "items": ["table", "desk", "coffee table", "dining table", "side table", "console", "mesa", "escritorio", "mesita"]
    },
    "storage": {
        "es": "Almacenamiento",
        "en": "Storage",
        "items": ["shelf", "cabinet", "drawer", "bookcase", "wardrobe", "dresser", "commode", "estantería", "aparador", "armario", "cómoda"]
    },
    "lighting": {
        "es": "Iluminación",
        "en": "Lighting",
        "items": ["lamp", "light", "chandelier", "sconce", "pendant", "lámpara", "aplique", "plafón", "colgante"]
    },
    "textiles": {
        "es": "Textiles",
        "en": "Textiles",
        "items": ["carpet", "rug", "curtain", "pillow", "cushion", "blanket", "alfombra", "cortina", "cojín", "manta"]
    },
    "decor": {
        "es": "Decoración",
        "en": "Decor",
        "items": ["vase", "mirror", "frame", "plant", "flower", "pot", "clock", "sculpture", "jarrón", "espejo", "marco", "planta", "reloj"]
    },
    "beds": {
        "es": "Camas",
        "en": "Beds",
        "items": ["bed", "mattress", "cama", "colchón"]
    },
    "outdoor": {
        "es": "Exterior",
        "en": "Outdoor",
        "items": ["planter", "bench", "table", "chair", "furniture", "maceta", "banco", "mueble exterior"]
    }
}

# Material classification keywords
MATERIAL_KEYWORDS = {
    "wood": {"es": "Madera", "en": "Wood", "keywords": ["wood", "wooden", "madera", "oak", "pine", "walnut", "mahogany", "encina", "pino", "nogal"]},
    "metal": {"es": "Metal", "en": "Metal", "keywords": ["metal", "steel", "iron", "aluminum", "brass", "chrome", "metal", "acero", "hierro", "aluminio", "latón"]},
    "glass": {"es": "Vidrio", "en": "Glass", "keywords": ["glass", "glass", "vidrio", "cristal", "mirror"]},
    "fabric": {"es": "Tela", "en": "Fabric", "keywords": ["fabric", "cloth", "textile", "velvet", "linen", "cotton", "tela", "algodón", "lino", "terciopelo"]},
    "leather": {"es": "Cuero", "en": "Leather", "keywords": ["leather", "leather", "cuero", "piel"]},
    "plastic": {"es": "Plástico", "en": "Plastic", "keywords": ["plastic", "resin", "polymer", "plástico", "resina"]},
    "marble": {"es": "Mármol", "en": "Marble", "keywords": ["marble", "stone", "granite", "mármol", "piedra", "granito"]},
    "concrete": {"es": "Hormigón", "en": "Concrete", "keywords": ["concrete", "cement", "hormigón", "cemento"]}
}

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
            cuda_ok, cuda_reason = torch_cuda_usable(torch, smoke_test=False)
            self.device = "cuda" if cuda_ok else "cpu"
            if not cuda_ok:
                logger.warning("Florence GPU disabled: %s. Falling back to CPU.", cuda_reason)
        
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
                    trust_remote_code=True,
                    attn_implementation="eager"  # Force standard attention to avoid SDPA incompatibility
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
        # First get generic object detection
        od_results = self.detect_objects(image_path)
        # Get dense caption for better description
        caption = self.caption_image(image_path, dense=True)
        
        # Extract furniture items and classify them
        furniture_items = self._classify_furniture(od_results.get("<OD>", []))
        
        # Extract materials from caption
        materials = self._extract_materials(caption)
        
        # Extract color palette
        color_extractor = ColorExtractor()
        palette = color_extractor.extract_palette(image_path, k=5)
        
        return {
            "objects": furniture_items,
            "materials": materials,
            "palette": palette,
            "description": caption
        }
    
    def _classify_furniture(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """Classify detected objects into furniture categories."""
        classified = []
        
        for det in detections:
            # Get label from detection
            label = det.get("label", "").lower() if isinstance(det, dict) else ""
            
            if not label:
                continue
            
            # Find matching category
            category = self._find_category(label)
            
            if category:
                classified.append({
                    "label": label,
                    "category": category["key"],
                    "category_es": category["es"],
                    "category_en": category["en"],
                    "bbox": det.get("bbox", []),
                    "score": det.get("score", 1.0)
                })
        
        return classified
    
    def _find_category(self, label: str) -> Optional[Dict[str, Any]]:
        """Find which furniture category a label belongs to."""
        for cat_key, cat_data in FURNITURE_CATEGORIES.items():
            for item in cat_data["items"]:
                if item in label or label in item:
                    return {
                        "key": cat_key,
                        "es": cat_data["es"],
                        "en": cat_data["en"]
                    }
        return None
    
    def _extract_materials(self, text: str) -> List[Dict[str, Any]]:
        """Extract materials mentioned in the caption."""
        text_lower = text.lower() if text else ""
        found_materials = []
        
        for mat_key, mat_data in MATERIAL_KEYWORDS.items():
            for keyword in mat_data["keywords"]:
                if keyword in text_lower:
                    found_materials.append({
                        "material": mat_key,
                        "name_es": mat_data["es"],
                        "name_en": mat_data["en"]
                    })
                    break  # Only add once per material
        
        return found_materials
