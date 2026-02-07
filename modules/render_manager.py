import logging
import requests
import json
import base64
import os
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class RenderManager:
    """
    Manages architectural rendering using Stable Diffusion + ControlNet.
    Connects to a local Stable Diffusion WebUI (A1111) API by default.
    """
    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        self.api_url = api_url

    def generate_from_sketch(self, sketch_path: str, prompt: str, negative_prompt: str = "") -> str:
        """
        Uses ControlNet (Canny or Scribble) to generate a render from a sketch/plan.
        """
        if not os.path.exists(sketch_path):
             return ""

        with open(sketch_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')

        # Default Architectural Prompt
        full_prompt = (
            "fotorrealistic architectural render, high quality, 8k, "
            "modern interior design, cinematic lighting, " + prompt
        )
        
        payload = {
            "prompt": full_prompt,
            "negative_prompt": "blurry, low quality, distorted, messy, " + negative_prompt,
            "steps": 25,
            "cfg_scale": 7,
            "width": 1024,
            "height": 768,
            "alwayson_scripts": {
                "ControlNet": {
                    "args": [
                        {
                            "input_image": encoded_image,
                            "module": "canny",
                            "model": "control_v11p_sd15_canny [d1110820]",
                            "weight": 1.0,
                        }
                    ]
                }
            }
        }

        try:
            logger.info("Sending request to Stable Diffusion API...")
            response = requests.post(url=f'{self.api_url}/sdapi/v1/txt2img', json=payload)
            response.raise_for_status()
            
            r = response.json()
            # The API returns a list of images in base64
            image_b64 = r['images'][0]
            
            image = Image.open(BytesIO(base64.b64decode(image_b64)))
            
            os.makedirs("data/renders", exist_ok=True)
            output_path = f"data/renders/render_{os.path.basename(sketch_path)}"
            image.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Stable Diffusion render failed: {e}")
            return ""
