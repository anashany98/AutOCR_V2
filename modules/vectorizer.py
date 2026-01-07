
import cv2
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Vectorizer:
    def vectorize_image(self, image_path: str, output_path: str) -> bool:
        """
        Converts a raster image to a vector SVG by detecting edges and contours.
        Returns True if successful.
        """
        try:
            # 1. Read Image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Preprocessing (Blur to reduce noise)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 3. Edge Detection (Canny)
            # Automatic parameter estimation
            v = np.median(blurred)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(blurred, lower, upper)
            
            # 4. Find Contours
            # RETR_LIST retrieves all contours
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # 5. Generate SVG
            height, width = img.shape[:2]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # SVG Header
                f.write(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
                f.write(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n')
                
                # Write paths
                f.write(f'<g stroke="black" fill="none" stroke-width="1">\n')
                
                for contour in contours:
                    # Simplify contour to get straighter lines (CAD-friendly)
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) < 2:
                        continue
                        
                    points_str = " ".join([f"{p[0][0]},{p[0][1]}" for p in approx])
                    
                    # Create polyline
                    f.write(f'<polyline points="{points_str}" />\n')
                
                f.write('</g>\n')
                f.write('</svg>')
                
            return True
            
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            return False
