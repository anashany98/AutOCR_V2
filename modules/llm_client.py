import logging
import os
from typing import Dict, Any, Optional, List
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
        
        self.base_url = self.config.get("base_url", "http://localhost:11434/v1")
        self.api_key = self.config.get("api_key", "ollama") 
        self.model = self.config.get("model", "moondream")
        self.provider = self.config.get("provider", "ollama").lower()
        self.timeout = self.config.get("timeout", 90)
        
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

    def _get_client_and_model(self, profile: Optional[str] = None):
        """
        RESOLVES the correct Client and Model Name based on the requested profile.
        Handles switching between Text (GPU 1) and Vision (GPU 2) endpoints.
        """
        # Default values from main config
        target_model = self.model
        target_base_url = self.base_url
        
        # Override if profile exists in routing
        if profile and "routing" in self.config:
            routing = self.config["routing"]
            if profile in routing:
                prof_conf = routing[profile]
                target_model = prof_conf.get("model", target_model)
                target_base_url = prof_conf.get("base_url", target_base_url)
        
        # If base_url matches the default self._client, use it.
        # Otherwise, create a temporary client (lightweight).
        if target_base_url == self.base_url and self._client:
            return self._client, target_model
            
        try:
            # Create ad-hoc client for the specific URL
            from openai import OpenAI
            temp_client = OpenAI(
                base_url=target_base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
            return temp_client, target_model
        except Exception as e:
            self.logger.error(f"Failed to create temp client for profile '{profile}' ({target_base_url}): {e}")
            return self._client, self.model # Fallback

    def _get_cloud_failover_client(self):
        """Returns a client configured for Cloud API (e.g., OpenAI/DeepSeek) as fallback."""
        try:
            from openai import OpenAI
            # Load from env or config (hardcoded for now or env var)
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                return None, None
                
            base_url = "https://api.openai.com/v1" if "sk-" in api_key else "https://api.deepseek.com/v1"
            model = "gpt-4o-mini" if "sk-" in api_key else "deepseek-chat"
            
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
            return client, model
        except:
            return None, None

    def analyze_document(self, text: str, reason: str, doc_type: str = "Documento") -> Dict[str, Any]:
        """
        Envía el texto del documento al LLM para su análisis.
        """
        if not self.enabled:
            return {"error": "LLM deshabilitado"}

        # Get Client dynamically (Default Profile)
        client, model = self._get_client_and_model("default")
        if not client:
             return {"error": "LLM Client not initialized"}

        system_prompt = (
            "Eres un asistente administrativo experto en análisis documental. "
            "Tu tarea es extraer información clave, corregir errores de OCR obvios y resumir el contenido.\n"
            "IMPORTANTE: Devuelve la respuesta ÚNICAMENTE en formato JSON válido. No incluyas explicaciones externas."
        )

        user_prompt = (
            f"Analiza el siguiente documento ({doc_type}).\n"
            f"Contexto: {reason}\n\n"
            f"--- TEXTO OCR ---\n{text[:4000]}\n-----------------\n"
            "Extrae: proveedor, fecha, base_imponible, iva, total."
        )

        try:
            self.logger.info(f"Enviando solicitud al LLM ({model})...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            # Cleanup DeepSeek <think> tags
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            # Basic cleanup if model returns markdown fencing
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            return {
                "success": True, 
                "analysis": content,
                "model_used": model
            }

        except Exception as e:
            self.logger.error(f"Error en llamada al LLM: {e}")
            return {"success": False, "error": str(e)}

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Generic chat call. If profile is provided, it uses the specific routing config.
        """
        if not self.enabled:
            return {"error": "LLM disabled"}

        # Use helper to get correct client/model
        client, model = self._get_client_and_model(profile or "default")
        if not client:
             return {"error": "LLM Client failed to initialize"}

        sys_p = system_prompt or "You are a helpful assistant for AutOCR V2."
        
        try:
            self.logger.info(f"LLM Chat (Profile: {profile or 'default'}, Model: {model})...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            
            # 1. Cleanup DeepSeek <think> tags (the most common leak)
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            
            # Look for common reasoning start patterns (Multilingual)
            reasoning_starters = ["vale,", "analizando", "veo que", "el usuario", "the user", "let me", "i will", "firstly", "viendo el contexto", "considerando", "el mensaje", "para responder"]
            
            if any(content.lower().strip().startswith(m) for m in reasoning_starters):
                import re
                # Common markers where the real answer usually starts (Surgical extraction)
                # We look for these markers especially when preceded by a period or double newline
                triggers = [r"¡Claro!", r"Claro,", r"Entendido\.", r"Aquí tienes", r"Por supuesto", r"Hola,", r"Hola\s", r"Buen día"]
                for trigger in triggers:
                    # Look for the trigger at the start of a logical paragraph or after a period
                    match = re.search(r"(\.|\n)\s*(" + trigger + r".*)", content, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(2).strip()
                        break
                else:
                    # Fallback to paragraph logic if no surgical trigger found
                    paragraphs = content.split("\n\n")
                    if len(paragraphs) > 1:
                        content = "\n\n".join(paragraphs[1:]).strip()

            # 3. Cleanup JSON markdown
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            
            # 4. Fallback if empty
            if not content or len(content.strip()) < 2:
                content = "Lo siento, no he podido generar una respuesta coherente. ¿Puedes reformular tu pregunta?"

            return {"success": True, "analysis": content, "model": model}
        except Exception as e:
            self.logger.error(f"LLM Chat failed (Local): {e}")
            
            # FAILOVER ATTEMPT
            self.logger.warning("Local GPU failed. Attempting Cloud Failover...")
            cloud_client, cloud_model = self._get_cloud_failover_client()
            if cloud_client:
                try:
                    response = cloud_client.chat.completions.create(
                        model=cloud_model,
                        messages=[
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1
                    )
                    content = response.choices[0].message.content
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "")
                    return {"success": True, "analysis": content, "model": f"{cloud_model} (CLOUD FAILOVER)"}
                except Exception as cloud_e:
                    self.logger.error(f"Cloud Failover also failed: {cloud_e}")
            
            return {"success": False, "error": str(e)}

    def chat_stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        """
        Stream chat response deltas.

        Returns an iterator of text chunks (str). Falls back to non-streaming chat() on failure.
        """
        if not self.enabled:
            return iter(())

        client, model = self._get_client_and_model(profile or "default")
        if not client:
            return iter(())

        sys_p = system_prompt or "You are a helpful assistant for AutOCR V2."

        def _iter():
            in_think = False
            pending = ""

            def _emit_ready() -> str:
                nonlocal in_think, pending
                out_parts = []
                # Strip <think>...</think> blocks across chunk boundaries.
                while pending:
                    if in_think:
                        end_idx = pending.lower().find("</think>")
                        if end_idx == -1:
                            # Still inside think block: drop everything we have so far.
                            pending = ""
                            break
                        pending = pending[end_idx + len("</think>") :]
                        in_think = False
                        continue

                    start_idx = pending.lower().find("<think>")
                    if start_idx == -1:
                        out_parts.append(pending)
                        pending = ""
                        break

                    # Emit anything before <think>, then enter think-mode.
                    if start_idx > 0:
                        out_parts.append(pending[:start_idx])
                    pending = pending[start_idx + len("<think>") :]
                    in_think = True

                return "".join(out_parts)

            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    stream=True,
                )

                for event in stream:
                    delta = None
                    try:
                        # OpenAI-compatible streaming chunks.
                        choice = event.choices[0]
                        if getattr(choice, "delta", None) is not None:
                            delta = getattr(choice.delta, "content", None)
                        if delta is None and getattr(choice, "message", None) is not None:
                            delta = getattr(choice.message, "content", None)
                    except Exception:
                        delta = None

                    if not delta:
                        continue

                    pending += str(delta)
                    out = _emit_ready()
                    if out:
                        yield out

                # Flush any remaining non-think content.
                out = _emit_ready()
                if out:
                    yield out
            except Exception as e:
                # Fall back to non-streaming response so the chat still works.
                self.logger.error(f"LLM stream failed: {e}")
                res = self.chat(user_prompt, system_prompt=system_prompt, profile=profile)
                text = res.get("analysis", "") if isinstance(res, dict) else ""
                if text:
                    yield str(text)

        return _iter()

    def analyze_sketch_ocr(self, text: str) -> Dict[str, Any]:
        """
        Specialized prompt for interpreting messy OCR from architectural sketches.
        """
        if not self.enabled:
            return {"error": "LLM disabled"}

        client, model = self._get_client_and_model("default")
        if not client:
            return {"error": "LLM Client not initialized"}

        system_prompt = (
            "You are an architect. Analyze the provided OCR text from a hand-drawn floor plan (sketch). "
            "Identify room names, rough dimensions, and potential scale even if misspelled. "
            "Return a JSON with keys: 'scale' (string or null), 'rooms' (list of strings), 'areas' (list of strings). "
            "Ignore noise."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"OCR TEXT:\n{text[:2000]}"}
                ],
                temperature=0.2,
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
        if not self.enabled:
            return {"error": "LLM disabled"}
            
        client, model = self._get_client_and_model("vision")
        if not client:
            return {"error": "Vision LLM Client not initialized"}
            
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
            
            self.logger.info(f"Sending Vision Request for {os.path.basename(image_path)} to {model}...")
            
            response = client.chat.completions.create(
                model=model,
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

    def analyze_decor_vision(self, image_path: str) -> Dict[str, Any]:
        """
        Multimodal analysis: Acts as a Personal Shopper/Decor Advisor.
        """
        if not self.enabled:
            return {"error": "LLM disabled"}
            
        # [NEW] Use 'vision' specific profile
        client, model = self._get_client_and_model("vision")
        if not client:
             return {"error": "Vision LLM Client not initialized"}
            
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
                "Eres un experto diseñador de interiores y 'Personal Shopper'. Analiza esta imagen y proporciona:\n"
                "1. Estilo Detectado\n"
                "2. Paleta de Colores\n"
                "3. Sugerencias de Compra (3 items)\n"
                "Return JSON: {\n"
                "  'style': '...', \n"
                "  'palette': ['#Hex', '#Hex'], \n"
                "  'recommendations': [\n"
                "    {'item': 'Nombre del Mueble', 'reason': 'Por qué encaja', 'query': 'Search Term for Amazon/Google'}\n"
                "  ]\n"
                "}\n"
                "IMPORTANT: Output ONLY JSON. No markdown."
            )
            
            self.logger.info(f"Sending Decor Vision Request for {os.path.basename(image_path)} to {model}...")
            
            response = client.chat.completions.create(
                model=model,
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
                temperature=0.5,
                max_tokens=800,
            )
            
            content = response.choices[0].message.content
            # Cleanup Markdown
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            if "```json" in content:
                 content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                 content = content.split("```")[1].strip()
                 
            return {"success": True, "analysis": content}

        except Exception as e:
            self.logger.error(f"Decor vision analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        Classifies the document into a category using LLM.
        """
        if not self.enabled:
            return {"error": "LLM disabled"}
        
        # Use default text profile
        client, model = self._get_client_and_model("default")
        if not client:
            return {"error": "LLM Client not initialized"}

        prompt = (
            f"Classify the following document text into one of these categories: "
            f"[Factura, Recibo, Contrato, Presupuesto, Nomina, Identificacion, Plano, Otro].\n"
            f"If uncertain, choose 'Otro'.\n"
            f"Return ONLY JSON: {{'category': 'CategoryName', 'confidence': 0.0-1.0}}\n\n"
            f"--- TEXT ---\n{text[:1500]}\n--- END TEXT ---"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Cleanup DeepSeek <think> tags
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            # Basic cleanup if model returns markdown fencing
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            return {"success": True, "analysis": content}
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return {"success": False, "error": str(e)}

    def smart_extract(self, text: str, fields: List[str]) -> Dict[str, Any]:
        """
        Extracts specific fields from the document text.
        """
        if not self.enabled or not self._client:
            return {"error": "LLM disabled"}
            
        system_prompt = (
            f"Extract the following fields: {', '.join(fields)}.\n"
            "Return JSON with keys matching the requested fields. If not found, use null.\n"
            "IMPORTANT: Output ONLY JSON. No markdown, no thinking."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"DOCUMENT TEXT:\n{text[:4000]}"}
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
             # Cleanup DeepSeek <think> tags
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            # Basic cleanup if model returns markdown fencing
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            return {"success": True, "analysis": content}
        except Exception as e:
            self.logger.error(f"Smart extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def summarize_document(self, text: str) -> Dict[str, Any]:
        """
        Generates a concise 1-sentence summary.
        """
        if not self.enabled or not self._client:
            return {"error": "LLM disabled"}

        system_prompt = "Generate a single, concise sentence summarizing this document (e.g., 'Invoice from X for Y amount', 'Contract for Z'). Language: Spanish."

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"TEXT:\n{text[:3000]}"}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            return {"success": True, "summary": content}
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return {"success": False, "error": str(e)}
