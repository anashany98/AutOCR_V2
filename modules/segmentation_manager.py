import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
from .torch_compat import torch_cuda_usable


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SAM segmentation requires PyTorch ('torch'). Install it (and CUDA libs if you want GPU) "
            "or disable SAM/vision segmentation features."
        ) from exc
    return torch


def _lazy_import_sam():
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SAM segmentation requires the 'segment_anything' package. Install it or disable SAM/vision segmentation."
        ) from exc
    return sam_model_registry, SamPredictor


class SegmentationManager:
    """
    Handles image segmentation using SAM (Segment Anything Model).
    Useful for extracting furniture or fabric samples from photos.
    """
    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = "models/sam_vit_b_01ec64.pth"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        # Resolve device at load time (keeps module importable without torch installed).
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self._loaded = False
        self._torch = None
        self._sam_model_registry = None
        self._SamPredictor = None

    def load(self):
        if self._loaded and self.model is not None:
            try:
                if next(self.model.parameters()).device.type == self.device:
                    return
            except Exception:
                pass

        if self._torch is None:
            self._torch = _lazy_import_torch()
        if self._sam_model_registry is None or self._SamPredictor is None:
            self._sam_model_registry, self._SamPredictor = _lazy_import_sam()

        cuda_ok, cuda_reason = torch_cuda_usable(self._torch, smoke_test=False)
        self.device = "cuda" if cuda_ok else "cpu"
        if not cuda_ok:
            logger.warning("SAM GPU disabled: %s. Falling back to CPU.", cuda_reason)
        
        # Request VRAM
        try:
            from modules.resource_orchestrator import ResourceOrchestrator

            orchestrator = ResourceOrchestrator()
            orchestrator.request_model("sam")
            orchestrator.register_model("sam", self)
        except Exception as exc:
            logger.debug("Resource orchestrator unavailable: %s", exc)

        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"SAM checkpoint not found at {self.checkpoint_path}. Downloading...")
            self._download_checkpoint()

        logger.info(f"Loading SAM ({self.model_type}) on {self.device}...")
        try:
            if not self.model:
                self.model = self._sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.model.to(device=self.device)
                self.predictor = self._SamPredictor(self.model)
            else:
                self.model.to(self.device)
                if self.predictor is None:
                    self.predictor = self._SamPredictor(self.model)
                
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

        dst = Path(self.checkpoint_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=(10, 600)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dst)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

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
