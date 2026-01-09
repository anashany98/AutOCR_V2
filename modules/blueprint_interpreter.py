
"""
Blueprint Interpreter for AutoOCR.

Specialized module for extracting architectural information from plan documents:
- Scale Detection (e.g., 1:50, 1/100)
- Dimension Extraction (linear measurements, areas)
- Room Detection (Kitchen, Bedroom, etc.)
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class BlueprintInterpreter:
    """Analyzes OCR text to extract blueprint-specific metadata."""

    # Common architectural scales
    SCALE_PATTERNS = [
        r'\bE\s?[:=]\s?1[:/](\d+)',  # E:1:50, E=1/100
        r'\bEscala\s?1[:/](\d+)',    # Escala 1:50
        r'\bScale\s?1[:/](\d+)',     # Scale 1:100
        r'\b1[:/](\d+)\b'            # Simple 1:50 (requires validation)
    ]

    # Area patterns (m2, sq ft)
    AREA_PATTERNS = [
        r'(\d+[.,]\d{1,2})\s?(?:m²|m2|sq\s?m)',   # 12.50 m²
        r'S\.?\s?(?:const|bhu)?[:\.]?\s?(\d+[.,]\d{1,2})', # S.Const: 120.50
    ]

    # Common room names (Multilingual support: ES/EN)
    ROOM_KEYWORDS = {
        'Cocina': ['cocina', 'kitchen', 'cocina-comedor', 'kitchenette'],
        'Baño': ['baño', 'aseo', 'bath', 'bathroom', 'toilet', 'wc'],
        'Dormitorio': ['dormitorio', 'habitación', 'alcoba', 'bedroom', 'room'],
        'Salón': ['salón', 'sala', 'living', 'comedor', 'lounge', 'estar'],
        'Entrada': ['entrada', 'hall', 'vestíbulo', 'recibidor', 'foyer'],
        'Pasillo': ['pasillo', 'distribuidor', 'corridor', 'hallway'],
        'Terraza': ['terraza', 'balcón', 'terrace', 'balcony'],
        'Lavadero': ['lavadero', 'tendedero', 'laundry', 'utility'],
        'Trastero': ['trastero', 'storage'],
        'Garaje': ['garaje', 'garage', 'aparcamiento']
    }

    def infer_metadata(self, text: str, llm_client=None, image_path: str = None) -> Dict[str, Any]:
        """
        Main entry point. Returns a dict with 'scale', 'rooms', 'areas'.
        Prioritizes:
        1. Regex (High Precision for CAD)
        2. Vision LLM (If image_path provided & Regex failed) - BEST for sketches
        3. Text LLM (If only text avail & Regex failed)
        """
        metadata = {
            "scale": self._extract_scale(text) if text else None,
            "rooms": self._extract_rooms(text) if text else [],
            "areas": self._extract_areas(text) if text else [],
            "method": "regex"
        }

        # If Regex found nothing significant, try AI
        if llm_client and (not metadata['scale'] or len(metadata['rooms']) == 0):
            import json
            
            # OPTION A: Vision (Best for Sketches)
            if image_path:
                logger.info("Regex extraction poor. Attempting Vision Sketch Analysis...")
                llm_result = llm_client.analyze_sketch_vision(image_path)
                method_tag = "vision_llm"
            
            # OPTION B: Text-only (Fallback)
            elif text:
                logger.info("Regex extraction poor. Attempting Text LLM Analysis...")
                llm_result = llm_client.analyze_sketch_ocr(text)
                method_tag = "text_llm"
            else:
                llm_result = {"success": False}

            # Process AI Result
            if llm_result.get("success"):
                try:
                    analysis_str = llm_result.get("analysis", "{}")
                    if "```" in analysis_str:
                         analysis_str = analysis_str.replace("```json", "").replace("```", "")
                    
                    ai_data = json.loads(analysis_str)
                    
                    # Merge logic
                    if not metadata['scale'] and ai_data.get('scale'):
                        metadata['scale'] = ai_data['scale']
                    if not metadata['rooms'] and ai_data.get('rooms'):
                         metadata['rooms'] = ai_data['rooms']
                    if not metadata['areas'] and ai_data.get('areas'):
                         metadata['areas'] = ai_data['areas']
                    metadata['method'] = method_tag
                except Exception as e:
                    logger.warning(f"AI parsing failed: {e}")

        return metadata

    def _extract_scale(self, text: str) -> Optional[str]:
        """Finds the most likely scale."""
        for pattern in self.SCALE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Validate denominator (standard architectural scales)
                denom = int(match.group(1))
                if denom in [10, 20, 50, 100, 200, 500, 1000]:
                    return f"1:{denom}"
        return None

    def _extract_rooms(self, text: str) -> List[str]:
        """Detects rooms mentioned in the text."""
        found_rooms = set()
        text_lower = text.lower()
        
        for standardized_name, synonyms in self.ROOM_KEYWORDS.items():
            for synonym in synonyms:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(synonym) + r'\b', text_lower):
                    found_rooms.add(standardized_name)
                    break
        
        return list(found_rooms)

    def _extract_areas(self, text: str) -> List[str]:
        """Extracts text fragments looking like area measurements."""
        areas = []
        for pattern in self.AREA_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                areas.append(match.group(0))
        return list(set(areas)) # Deduplicate
