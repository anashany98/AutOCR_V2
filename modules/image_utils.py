import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

def enhance_image(pil_image: Image.Image, 
                 contrast: float = 1.0, 
                 brightness: float = 1.0, 
                 sharpness: float = 1.0, 
                 apply_clahe: bool = False) -> Image.Image:
    """
    Apply visual enhancements to an image.
    
    Args:
        pil_image: Source PIL Image
        contrast: Float multiplier (1.0 = original)
        brightness: Float multiplier (1.0 = original)
        sharpness: Float multiplier (1.0 = original)
        apply_clahe: Boolean to apply adaptive histogram equalization
        
    Returns:
        Enhanced PIL Image
    """
    try:
        # 1. Basic PIL Enhancements (Fast)
        img = pil_image.copy()
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
            
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)
            
        # 2. Advanced OpenCV Enhancements (CLAHE)
        if apply_clahe:
            # Convert to LAB color space
            img_np = np.array(img)
            # Handle RGBA
            if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
            lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(final)
            
        return img
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return pil_image

def detect_handwriting_probability(pil_image: Image.Image) -> float:
    """
    Estimate probability (0.0 - 1.0) that the image contains handwriting.
    Uses heuristic based on connected components & contour irregularity.
    """
    try:
        # Convert to grayscale numpy
        img_np = np.array(pil_image.convert("L"))
        
        # Binarize
        _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # Analyze contours
        irregularity_scores = []
        for cnt in contours:
            # Filter noise
            if cv2.contourArea(cnt) < 20: 
                continue
                
            # Bounding rect
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            
            # Convex Hull Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(cnt)
            solidity = float(cnt_area)/hull_area if hull_area > 0 else 0
            
            # Handwriting tends to be more irregular (lower solidity) and variable aspect ratio
            # Printed text (especially block) is very solid
            irregularity_scores.append(solidity)
            
        if not irregularity_scores:
            return 0.0
            
        avg_solidity = np.mean(irregularity_scores)
        
        # Heuristic: Printed text usually has solidity > 0.85
        # Handwriting usually has solidity < 0.75
        if avg_solidity < 0.75:
            return 0.8  # Likely handwriting
        elif avg_solidity < 0.85:
            return 0.4  # Uncertain
        else:
            return 0.1  # Likely printed
            
    except Exception as e:
        logger.error(f"Handwriting detection failed: {e}")
        return 0.0
        return 0.0

def preprocess_image_for_ocr(image_path: str, deskew: bool = True, denoise: bool = True) -> str:
    """
    Preprocess an image for better OCR results.
    Returns path to temporary processed file.
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Deskew (Straighten)
        if deskew:
            gray, angle = deskew_image(gray)
            if angle != 0:
                logger.info(f"Deskewed image by {angle:.2f} degrees")
                
        # 3. Denoise (Clean)
        if denoise:
            gray = denoise_image(gray)
            
        # Save to temp file
        import tempfile
        import os
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(temp_path, gray)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return image_path

def deskew_image(gray_img: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct text skew/rotation."""
    try:
        # Invert colors (text must be white)
        coords = np.column_stack(np.where(gray_img > 0)) # Assuming white text? No, usually black text on white.
        # Threshold to get black text
        thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return gray_img, 0.0
            
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate
        (h, w) = gray_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, angle
    except Exception:
        return gray_img, 0.0

def denoise_image(gray_img: np.ndarray) -> np.ndarray:
    """Remove noise using Fast Non-Local Means."""
    try:
        # h: parameter deciding filter strength. Higher h -> removes more noise but also removes details.
        # For OCR, 10 is usually safe.
        return cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    except Exception:
        return gray_img
