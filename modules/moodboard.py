from __future__ import annotations
import os
import math
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class MoodboardGenerator:
    """
    Generates composite moodboard images from a list of files.
    """
    
    def __init__(self, output_dir: str = "data/moodboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Fonts - load default or fallback
        self.font_path = "arial.ttf" 

    def create(self, image_paths: List[str], title: str = "Moodboard") -> Optional[str]:
        """
        Creates a moodboard from the given images.
        Returns the absolute path to the generated file.
        """
        if not image_paths:
            return None

        valid_images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGBA")
                valid_images.append(img)
            except Exception as e:
                logger.warning(f"Could not load image {p}: {e}")
        
        if not valid_images:
            return None

        # Canvas settings
        count = len(valid_images)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        
        # Target size for each slot (HD standardish)
        slot_w = 800
        slot_h = 600
        padding = 40
        header_h = 150
        
        canvas_w = (cols * slot_w) + ((cols + 1) * padding)
        canvas_h = (rows * slot_h) + ((rows + 1) * padding) + header_h
        
        # Create background (White/Off-white)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (245, 245, 240, 255))
        draw = ImageDraw.Draw(canvas)
        
        # Draw Header
        try:
            # Try to load a nicer font, fallback to default
            font = ImageFont.truetype(self.font_path, 80)
        except IOError:
            font = ImageFont.load_default()
            
        # Draw Title
        # Center text 
        # (Naive centering for PIL default font is weird, but let's try)
        draw.text((padding, padding), title, fill=(50, 50, 50, 255), font=font)
        
        # Draw Images
        for idx, img in enumerate(valid_images):
            col = idx % cols
            row = idx // cols
            
            x = padding + (col * (slot_w + padding))
            y = header_h + padding + (row * (slot_h + padding))
            
            # Resize image to fit slot (contain)
            img_ratio = img.width / img.height
            slot_ratio = slot_w / slot_h
            
            if img_ratio > slot_ratio:
                # Wider than slot
                new_w = slot_w
                new_h = int(slot_w / img_ratio)
            else:
                # Taller than slot
                new_h = slot_h
                new_w = int(slot_h * img_ratio)
                
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center in slot
            paste_x = x + (slot_w - new_w) // 2
            paste_y = y + (slot_h - new_h) // 2
            
            # Shadow/Border effect (simple rect behind)
            shadow_offset = 10
            draw.rectangle(
                [paste_x + shadow_offset, paste_y + shadow_offset, paste_x + new_w + shadow_offset, paste_y + new_h + shadow_offset],
                fill=(200, 200, 200, 100)
            )
            
            canvas.paste(img_resized, (paste_x, paste_y), img_resized)

        # Save
        filename = f"{title.replace(' ', '_')}_{len(valid_images)}i.png"
        output_path = self.output_dir / filename
        canvas.save(output_path)
        logger.info(f"Moodboard saved to {output_path}")
        
        return str(output_path.absolute())
