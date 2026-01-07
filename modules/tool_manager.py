import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from .db_manager import DBManager
from .exporter import export_to_csv
from .translator import Translator
from .vectorizer import Vectorizer

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages tools available to the LLM agent.
    Each tool is a function that performs an action on the system.
    """

    def __init__(self, db_manager: DBManager, project_root: str, vision_manager=None):
        self.db = db_manager
        self.vision_manager = vision_manager
        self.translator = Translator()
        self.vectorizer = Vectorizer()
        self.project_root = project_root
        self.exports_dir = os.path.join(project_root, "data", "exports")
        os.makedirs(self.exports_dir, exist_ok=True)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns tool definitions in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "export_search_results_to_csv",
                    "description": "Exporta los resultados actuales de la búsqueda o consulta a un archivo CSV para su descarga.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "La consulta o filtro utilizado para obtener los documentos (opcional)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_document_type",
                    "description": "Actualiza el tipo documental de un archivo específico (p. ej. Factura, Contrato, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "integer",
                                "description": "El ID del documento a actualizar."
                            },
                            "new_type": {
                                "type": "string",
                                "description": "El nuevo tipo asignado (Invoice, Receipt, Contract, Report, Link, Letter, Technical Plan, Imagen, Unknown)."
                            }
                        },
                        "required": ["doc_id", "new_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_image_colors",
                    "description": "Analiza los colores dominantes de un documento y devuelve sus códigos HEX y RGB.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "integer",
                                "description": "El ID del documento a analizar."
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "translate_document",
                    "description": "Traduce el contenido de texto de un documento a un idioma específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "integer",
                                "description": "El ID del documento a traducir."
                            },
                            "target_lang": {
                                "type": "string",
                                "description": "El código de idioma destino (es, en, fr, de, it, zh...)."
                            }
                        },
                        "required": ["doc_id", "target_lang"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vectorize_document",
                    "description": "Convierte un documento (plano, dibujo, esquema) en un archivo vectorial SVG editable compatible con CAD. Útil para 'calcar' o redibujar planos.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "integer",
                                "description": "El ID del documento a vectorizar."
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            }
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool by name with provided arguments."""
        logger.info(f"Executing tool: {name} with args: {arguments}")
        
        if name == "export_search_results_to_csv":
            return self._export_docs(arguments.get("query"))
        elif name == "update_document_type":
            return self._update_type(arguments.get("doc_id"), arguments.get("new_type"))
        elif name == "get_image_colors":
            return self._get_image_colors(arguments.get("doc_id"))
        elif name == "translate_document":
            return self._translate_doc(arguments.get("doc_id"), arguments.get("target_lang"))
        
        return f"Error: Tool '{name}' not found."

    def _export_docs(self, query: Optional[str]) -> str:
        # If query is provided, we might want to filter, but for now we'll export "recent" or "matching"
        # In a real agent, we'd use the RAG results or a DB search
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if query:
                # Simpler keyword search in filename/text
                cursor.execute(
                    "SELECT * FROM documents WHERE filename ILIKE %s OR text ILIKE %s LIMIT 100",
                    (f"%{query}%", f"%{query}%")
                )
            else:
                cursor.execute("SELECT * FROM documents ORDER BY datetime DESC LIMIT 100")
            
            columns = [desc[0] for desc in cursor.description]
            docs = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if not docs:
            return "No se encontraron documentos para exportar."

        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(self.exports_dir, filename)
        abs_path = export_to_csv(docs, path)
        
        if abs_path:
            # We return a message with a relative URL for the frontend to download
            rel_url = f"/data/exports/{filename}"
            return f"Exportación completada. Puedes descargar el archivo aquí: {rel_url}"
        
        return "Error al generar la exportación CSV."

    def _update_type(self, doc_id: int, new_type: str) -> str:
        try:
            self.db.update_document_type(doc_id, new_type)
            return f"Documento {doc_id} actualizado a tipo: {new_type}."
        except Exception as e:
            return f"Error al actualizar: {str(e)}"

    def _get_image_colors(self, doc_id: int) -> str:
        if not self.vision_manager:
            return "Error: El módulo de visión no está disponible."
        
        path = self.db.get_document_path(doc_id)
        if not path:
            return f"Error: No se encontró el documento ID {doc_id}."
        
        colors = self.vision_manager.analyze_colors(path)
        if not colors:
            return "No se pudieron extraer colores."
        
        # Format nice response
        response = f"Colores dominantes en Documento {doc_id}:\n"
        for c in colors:
            response += f"- [COLOR:{c['hex']}] {c['hex']} (Presencia: {c['count']})\n"
        return response

    def _translate_doc(self, doc_id: int, target_lang: str) -> str:
        # Retrieve text from DB
        doc = self.db.get_document(doc_id)
        if not doc:
            return f"Error: Documento {doc_id} no encontrado."
        
        text = doc.get('text', '')
        if not text:
            return "El documento no tiene texto extraído para traducir."
            
        translated = self.translator.translate_text(text, target_lang)
        
        # Save translation to a file optionally, or just return snippet
        # For chat experience, returning the first 500 chars is good, or saving a file.
        # Let's save a file in exports to be cleaner.
        
        filename = f"trans_{doc_id}_{target_lang}_{datetime.now().strftime('%H%M%S')}.txt"
        path = os.path.join(self.exports_dir, filename)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(translated)
                
            rel_url = f"/data/exports/{filename}"
            preview = translated[:500] + "..." if len(translated) > 500 else translated
            return f"Traducción completada ({target_lang}).\nDescargar: {rel_url}\n\nVista previa:\n{preview}"
        except Exception as e:
            return f"Error al guardar traducción: {e}"
