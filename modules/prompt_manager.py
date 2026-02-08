import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages versioned prompts for different roles and AI tasks.
    Enables swapping prompts without modifying code.
    """
    def __init__(self, prompts_dir: str = "data/prompts"):
        self.prompts_dir = prompts_dir
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.prompts = {}
        self.current_version = "v1"
        self._load_defaults()

    def _load_defaults(self):
        """Initializes default prompts if files don't exist."""
        default_prompts = {
            "v1": {
                "SYSTEM": "Eres un motor de orquestación de IA para AutOCR V2. Tu objetivo es procesar las peticiones del usuario y decidir qué herramientas ejecutar.",
                "CLIENTE": "Eres un asistente para el cliente final. Responde de forma amable, enfocándote en el diseño y la estética. No menciones costes técnicos.",
                "GESTOR": "Eres un asistente para el gestor de proyectos. Ayuda con la documentación y la coordinación de muebles. No tienes acceso a márgenes financieros.",
                "DIRECCION": "Eres un asesor financiero y estratégico. Analiza costes, rentabilidad y presupuestos con precisión.",
                "ADMIN": "Eres el administrador del sistema. Tienes control total sobre la infraestructura y los datos.",
                "OCR": "Corrige y estructura el siguiente texto extraído por OCR. Devuelve JSON.",
                "RAG_TEXT": "Busca en la base de conocimientos la respuesta a: {query}. Usa solo la información proporcionada.",
                "RAG_FINANCIAL": "Analiza los datos económicos del hotel para responder: {query}. Prioriza la precisión numérica.",
                "VISION_DESIGN": "Analiza la imagen desde una perspectiva de diseño de interiores. Detecta estilos y materiales.",
                "PRODUCT_SUGGESTION": "Basado en los elementos detectados, sugiere productos similares del catálogo."
            }
        }
        
        for version, data in default_prompts.items():
            path = os.path.join(self.prompts_dir, f"prompts_{version}.json")
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        
        self.load_version(self.current_version)

    def load_version(self, version: str):
        path = os.path.join(self.prompts_dir, f"prompts_{version}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            self.current_version = version
            logger.info(f"Prompts version {version} loaded.")
        else:
            logger.error(f"Prompt version {version} not found.")

    def get_prompt(self, key: str, **kwargs) -> str:
        prompt = self.prompts.get(key, "")
        if prompt and kwargs:
            try: return prompt.format(**kwargs)
            except: return prompt
        return prompt
