"""
Vectorization Manager for AutoOCR.
Converts raster images (sketches, plans) into vector formats (DXF/SVG) for CAD interoperability.
"""

import logging
import os
import math
from typing import List, Tuple, Optional
import cv2
import numpy as np

class SimpleDXFWriter:
    """A lightweight DXF writer to avoid external dependencies like ezdxf."""
    
    def __init__(self):
        self.entities = []
        
    def add_polyline(self, points: List[Tuple[float, float]], closed: bool = False):
        if len(points) < 2:
            return
        self.entities.append(("POLYLINE", points, closed))
        
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            # Header
            f.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
            # Tables
            f.write("0\nSECTION\n2\nTABLES\n0\nENDSEC\n")
            # Blocks
            f.write("0\nSECTION\n2\nBLOCKS\n0\nENDSEC\n")
            # Entities
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            for ent_type, data, closed in self.entities:
                if ent_type == "POLYLINE":
                    f.write("0\nLWPOLYLINE\n8\n0\n") # Layer 0
                    f.write(f"90\n{len(data)}\n") # Number of vertices
                    f.write(f"70\n{1 if closed else 0}\n") # Flags (1 = closed)
                    for x, y in data:
                        f.write(f"10\n{x:.4f}\n20\n{y:.4f}\n")
                    f.write("0\n")
            
            f.write("ENDSEC\n")
            f.write("0\nEOF\n")

class VectorizationManager:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def raster_to_dxf(self, image_path: str, output_path: str, min_length: int = 10):
        """
        Converts a raster image to DXF by detecting edges and contours.
        Optimized for sketches and technical plans.
        """
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Preprocessing (denoise)
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Canny Edge Detection (Auto-tuned)
            v = np.median(blurred)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(blurred, lower, upper)
            
            # Find Contours
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            dxf = SimpleDXFWriter()
            count = 0
            
            # Processing to flip Y axis (Image coords are top-left, DXF is bottom-left usually)
            height = img.shape[0]
            
            for cnt in contours:
                # Filter small noise
                if cv2.arcLength(cnt, True) < min_length:
                    continue
                
                # Simplify contour (Ramer-Douglas-Peucker)
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                points = []
                for p in approx:
                    x, y = p[0]
                    # Flip Y for CAD compatibility
                    points.append((float(x), float(height - y)))
                
                dxf.add_polyline(points, closed=True)
                count += 1
                
            dxf.save(output_path)
            self.logger.info(f"Vectorized {image_path} to {output_path}: {count} entities.")
            return True
            
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            return False

# Example usage if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vm = VectorizationManager()
    # vm.raster_to_dxf("input.jpg", "output.dxf")
