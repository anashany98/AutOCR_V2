import logging
import os
from typing import Dict, Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class LLMClient:
    """
    Cliente genérico para conectar con LLMs (OpenAI, LM Studio, Ollama, etc)
    usando la librería estándar 'openai'.
    """
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        
        self.base_url = self.config.get("base_url", "http://localhost:1234/v1")
        self.api_key = self.config.get("api_key", "lm-studio") # Default for local
        self.model = self.config.get("model", "local-model")
        self.timeout = self.config.get("timeout", 60)
        
        self._client = None
        if self.enabled:
            self._init_client()

    def _init_client(self):
        if not OpenAI:
            self.logger.warning("Librería 'openai' no instalada. Funcionalidad LLM deshabilitada.")
            self.enabled = False
            return
            
        try:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
            self.logger.info(f"LLM Client inicializado: {self.base_url} (Model: {self.model})")
        except Exception as e:
            self.logger.error(f"Error al inicializar cliente LLM: {e}")
            self.enabled = False

    def analyze_document(self, text: str, reason: str, doc_type: str = "Documento") -> Dict[str, Any]:
        """
        Envía el texto del documento al LLM para su análisis.
        """
        if not self.enabled or not self._client:
            return {"error": "LLM deshabilitado o no inicializado"}

        system_prompt = (
            "Eres un asistente administrativo experto en análisis documental. "
            "Tu tarea es extraer información clave, corregir errores de OCR obvios y resumir el contenido.\n"
            "Devuelve la respuesta en formato JSON puro."
        )

        user_prompt = (
            f"Analiza el siguiente documento clasificado como '{doc_type}'.\n"
            f"Contexto de revisión: {reason}\n\n"
            f"--- TEXTO OCR ---\n{text[:4000]}...\n-----------------" # Truncate to avoid context limit
        )

        try:
            self.logger.info(f"Enviando solicitud al LLM ({self.model})...")
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            # Basic cleanup if model returns markdown fencing
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            return {
                "success": True, 
                "analysis": content,
                "model_used": self.model
            }

        except Exception as e:
            self.logger.error(f"Error en llamada al LLM: {e}")
            return {"success": False, "error": str(e)}

    def analyze_sketch_ocr(self, text: str) -> Dict[str, Any]:
        """
        Specialized prompt for interpreting messy OCR from architectural sketches.
        """
        if not self.enabled or not self._client:
            return {"error": "LLM disabled"}

        system_prompt = (
            "You are an architect. Analyze the provided OCR text from a hand-drawn floor plan (sketch). "
            "Identify room names, rough dimensions, and potential scale even if misspelled. "
            "Return a JSON with keys: 'scale' (string or null), 'rooms' (list of strings), 'areas' (list of strings). "
            "Ignore noise."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"OCR TEXT:\n{text[:2000]}"}
                ],
                temperature=0.2,
                response_format={ "type": "json_object" } # Force JSON mode if supported
            )
            content = response.choices[0].message.content
            return {"success": True, "analysis": content}
        except Exception as e:
            self.logger.error(f"Sketch analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def analyze_sketch_vision(self, image_path: str) -> Dict[str, Any]:
        """
        Multimodal analysis: Sends the partial/sketch image directly to the VLM.
        """
        if not self.enabled or not self._client:
            return {"error": "LLM disabled"}
            
        import base64
        import mimetypes

        try:
            # Check file size/existence
            if not os.path.exists(image_path):
                 return {"success": False, "error": "Image file not found"}
                 
            # Encode image
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image'):
                mime_type = 'image/jpeg'
                
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            system_prompt = (
                "You are an expert architect analyzing a hand-drawn floor plan (sketch). "
                "Visually identify rooms (Label them if seen), detect any handwritten scale (e.g. 1:50), "
                "and estimated areas. "
                "Return JSON: {'scale': str|null, 'rooms': [str], 'areas': [str]}."
            )
            
            self.logger.info(f"Sending Vision Request for {os.path.basename(image_path)} to {self.model}...")
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content
            # Cleanup Markdown
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
                
            return {"success": True, "analysis": content}

        except Exception as e:
            self.logger.error(f"Vision analysis failed: {e}")
            return {"success": False, "error": str(e)}
