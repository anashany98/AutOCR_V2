import os
import logging
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import torch

logger = logging.getLogger(__name__)

class SegmentationManager:
    """
    Handles image segmentation using SAM (Segment Anything Model).
    Useful for extracting furniture or fabric samples from photos.
    """
    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = "models/sam_vit_b_01ec64.pth"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None
        self._loaded = False

    def load(self):
        if self._loaded and next(self.model.parameters()).device.type == self.device:
            return
        
        # Request VRAM
        from modules.resource_orchestrator import ResourceOrchestrator
        ResourceOrchestrator().request_model("sam")
        ResourceOrchestrator().register_model("sam", self)

        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"SAM checkpoint not found at {self.checkpoint_path}. Downloading...")
            self._download_checkpoint()

        logger.info(f"Loading SAM ({self.model_type}) on {self.device}...")
        try:
            if not self.model:
                self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.model.to(device=self.device)
                self.predictor = SamPredictor(self.model)
            else:
                self.model.to(self.device)
                
            self._loaded = True
            logger.info("SAM loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SAM: {e}")
            raise

    def unload(self):
        if self.model:
            logger.info("Moving SAM to CPU...")
            self.model.to("cpu")
            self._loaded = False

    def _download_checkpoint(self):
        """Download weights if missing."""
        import requests
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        response = requests.get(url, stream=True)
        with open(self.checkpoint_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def segment_by_points(self, image: Image.Image, points: List[List[int]], labels: List[int]) -> List[np.ndarray]:
        """
        Segment an object based on input points.
        points: list of [x, y]
        labels: 1 for foreground, 0 for background
        """
        self.load()
        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True
        )
        
        return masks # Returns 3 candidate masks

    def extract_object(self, image_path: str, bbox: List[int]) -> str:
        """
        Prompt SAM with a box to extract an object.
        Returns the path to the extracted PNG (with transparency).
        """
        self.load()
        img = Image.open(image_path).convert("RGB")
        image_np = np.array(img)
        self.predictor.set_image(image_np)
        
        # bbox: [x1, y1, x2, y2]
        input_box = np.array(bbox)
        
        masks, _, _ = self.predictor.predict(
            box=input_box,
            multimask_output=False
        )
        
        mask = masks[0]
        
        # Create transparent image
        img_rgba = img.convert("RGBA")
        data = np.array(img_rgba)
        
        # Apply mask to alpha channel
        data[:, :, 3] = mask.astype(np.uint8) * 255
        
        result_img = Image.fromarray(data)
        
        os.makedirs("data/segments", exist_ok=True)
        output_path = f"data/segments/extracted_{os.path.basename(image_path)}"
        result_img.save(output_path)
        
        return output_path
