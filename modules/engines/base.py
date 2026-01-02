from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np
from PIL import Image

class OCREngine(ABC):
    """
    Abstract base class for OCR engines (e.g. Surya, Tesseract, GOT-OCR).
    """
    
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger
        self.enabled = config.get("enabled", False)

    @abstractmethod
    def initialize(self) -> bool:
        """
        Load models and resources. Return True if successful.
        """
        pass

    @abstractmethod
    def extract_text(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text from a full image. Returns (text, confidence).
        """
        pass

    @abstractmethod
    def extract_block(self, image: Image.Image, bbox: list) -> Tuple[str, float]:
        """
        Extract text from a specific crop/bbox. Returns (text, confidence).
        """
        pass
    
    def shutdown(self):
        """
        Release resources.
        """
        pass
