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
        self.current_version = "v2"  # Use v2 with enhanced search prompts
        self._load_defaults()
        
        # Log warning about version change from v1 to v2
        logger.info(f"PromptManager initialized with version '{self.current_version}'. "
                   "If you were using v1, behavior may have changed.")

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
            },
            "v2": {
                "CHAT_GENERAL": "Eres un asistente inteligente de documentos. Responde las preguntas basándote en el contexto proporcionado. Si la información no está en el contexto, dilo claramente. Cita las fuentes usando el formato [Fuente N]. Responde en español.",
                "CHAT_SEARCH_CONTRACT": "Eres un asistente especializado en búsqueda de contratos. Analiza los documentos para encontrar cláusulas específicas, fechas importantes, partes involucradas y condiciones. Responde de forma precisa y cita las fuentes [Fuente N].",
                "CHAT_SEARCH_INVOICE": "Eres un asistente especializado en búsqueda de facturas. Analiza los documentos para encontrar importes, fechas de vencimiento, proveedores, conceptos y estados de pago. Proporciona los datos exactos con referencias a las fuentes.",
                "CHAT_SEARCH_PROPOSAL": "Eres un asistente especializado en propuestas comerciales. Busca información sobre precios, productos/servicios ofertados, condiciones, plazos y estado de aprobación. Incluye siempre las referencias a las fuentes.",
                "CHAT_SEARCH_VENDOR": "Eres un asistente especializado en búsqueda de proveedores. Encuentra información de contacto, servicios ofrecidos, histórico de pedidos y calificaciones. Proporciona datos de contacto completos.",
                "CHAT_SEARCH_PROJECT": "Eres un asistente de gestión de proyectos. Busca información sobre estados, presupuestos, hitos, asignaciones y documentación asociada. Proporciona resúmenes ejecutivos y referencias.",
                "CHAT_SUMMARY": "Eres un asistente de resumen. Proporciona un resumen conciso del contenido encontrado, destacando los puntos más importantes. Estructura la respuesta en secciones claras.",
                "CHAT_COMPARISON": "Eres un asistente de comparación. Cuando el usuario pide comparar documentos, identifica las diferencias clave en precios, condiciones, fechas y términos. Usa tablas para mostrar comparaciones claras.",
                "CHAT_EXTRACTION": "Eres un asistente de extracción de datos. Extrae información estructurada de los documentos: fechas, importes, nombres, códigos, referencias. Formatea como JSON o tabla según convenga.",
                "CHAT_ANSWER": "Responde a la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado. Si no tienes información suficiente, indica que no puedes responder con los datos disponibles. Cita las fuentes [Fuente N]."
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
        """Get a prompt by key, with optional formatting."""
        prompt = self.prompts.get(key, "")
        if prompt and kwargs:
            try: 
                return prompt.format(**kwargs)
            except: 
                return prompt
        return prompt
    
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all available prompts for the current version."""
        return self.prompts.copy()
    
    def get_search_prompts(self) -> Dict[str, str]:
        """Get only the search/chat prompts."""
        search_keys = [k for k in self.prompts.keys() if k.startswith("CHAT_") or k.startswith("RAG_")]
        return {k: self.prompts[k] for k in search_keys}
    
    def list_versions(self) -> list:
        """List all available prompt versions."""
        versions = []
        try:
            for f in os.listdir(self.prompts_dir):
                if f.startswith("prompts_") and f.endswith(".json"):
                    v = f.replace("prompts_", "").replace(".json", "")
                    versions.append(v)
        except:
            pass
        return sorted(versions)
