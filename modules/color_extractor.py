from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List
import logging

logger = logging.getLogger(__name__)

class ColorExtractor:
    """
    Extracts dominant color palette from images using K-Means clustering.
    """
    
    @staticmethod
    def rgb_to_hex(rgb) -> str:
        """Converts RGB tuple to HEX string."""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def extract_palette(self, image_path: str, k: int = 5) -> List[str]:
        """
        Processes an image and returns a list of k dominant colors in HEX format.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image at {image_path}")
                return []

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            scale_percent = 20 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            # Flatten to a list of pixels
            pixels = img.reshape((-1, 3))

            # Apply K-Means
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(pixels)

            # Get dominant colors (cluster centers)
            colors = kmeans.cluster_centers_
            
            # Convert to HEX
            hex_colors = [self.rgb_to_hex(color) for color in colors]
            
            return hex_colors

        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return []
