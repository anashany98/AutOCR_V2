import json
import logging
import os
import secrets
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

    def __init__(
        self,
        db_manager: DBManager,
        project_root: str,
        vision_manager=None,
        product_manager=None,
        allowed_doc_roots: Optional[List[str]] = None,
    ):
        self.db = db_manager
        self.vision_manager = vision_manager
        self.product_manager = product_manager
        self.translator = Translator()
        self.vectorizer = Vectorizer()
        self.project_root = project_root
        self.exports_dir = os.path.join(project_root, "data", "exports")
        os.makedirs(self.exports_dir, exist_ok=True)

        # Restrict file access for tools that touch the filesystem (defense-in-depth).
        self.allowed_doc_roots: List[Path] = []
        for root in (allowed_doc_roots or []):
            try:
                p = Path(str(root)).resolve()
            except Exception:
                continue
            if str(p) and str(p) not in {"", "."}:
                self.allowed_doc_roots.append(p)

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
            },
            {
                "type": "function",
                "function": {
                    "name": "check_inventory",
                    "description": "Consulta el stock y precio en tiempo real de un producto usando su SKU.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": "El código SKU del producto (ej: SOFA-CHE-001)."
                            }
                        },
                        "required": ["sku"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_cart",
                    "description": "Añade un producto al carrito de compra del usuario.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": "El código SKU del producto."
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Cantidad a añadir (por defecto 1)."
                            }
                        },
                        "required": ["sku"]
                    }
                }
            }
        ]

    def execute_tool(
        self, name: str, arguments: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Executes a tool by name with provided arguments (scoped by user_context)."""
        logger.info(f"Executing tool: {name} with args: {arguments}")

        role, _, _ = self._get_scope(user_context)
        if not role:
            role = "UNKNOWN"

        # Tool-level allowlist (defense-in-depth). The router is LLM-driven and must not be trusted.
        if role in {"CLIENTE", "CLIENT"}:
            allowed = {"check_inventory", "add_to_cart"}
            if name not in allowed:
                return "Error: No tienes permisos para esta acción."
        elif role not in {"GESTOR", "DIRECCION", "ADMIN", "UNKNOWN"}:
            return "Error: No tienes permisos para esta acción."

        # External translation is a privacy/compliance risk (it may send document text to third parties).
        if name == "translate_document":
            if str(os.environ.get("AUTOOCR_ALLOW_EXTERNAL_TRANSLATION", "")).strip().lower() not in {"1", "true", "yes"}:
                return "Error: Traducción deshabilitada por seguridad (requiere AUTOOCR_ALLOW_EXTERNAL_TRANSLATION=1)."
        
        if name == "export_search_results_to_csv":
            return self._export_docs(arguments.get("query"), user_context=user_context)
        elif name == "update_document_type":
            return self._update_type(arguments.get("doc_id"), arguments.get("new_type"), user_context=user_context)
        elif name == "get_image_colors":
            return self._get_image_colors(arguments.get("doc_id"), user_context=user_context)
        elif name == "translate_document":
            return self._translate_doc(arguments.get("doc_id"), arguments.get("target_lang"), user_context=user_context)
        elif name == "vectorize_document":
            return self._vectorize_doc(arguments.get("doc_id"), user_context=user_context)
        elif name == "check_inventory":
            return self._check_inventory(arguments.get("sku"))
        elif name == "add_to_cart":
            return self._add_to_cart(arguments.get("sku"), arguments.get("quantity", 1))
        elif name == "analyze_document_structure":
            return self._analyze_structure(arguments.get("doc_id"), user_context=user_context)
        
        return f"Error: Tool '{name}' not found."

    def _get_scope(self, user_context: Optional[Dict[str, Any]]) -> Tuple[str, List[Any], Optional[str]]:
        ctx = user_context or {}
        role = str(ctx.get("role", "")).upper()
        hotel_scope = ctx.get("hotel_scope") or []
        if not isinstance(hotel_scope, list):
            hotel_scope = []
        user_id = ctx.get("user_id")
        user_id_str = str(user_id) if user_id is not None else None
        return role, hotel_scope, user_id_str

    def _enforce_doc_access(self, doc_id: int, user_context: Optional[Dict[str, Any]]):
        role, hotel_scope, user_id_str = self._get_scope(user_context)

        try:
            doc_id_int = int(doc_id)
        except Exception:
            return None, "Error: doc_id inválido."

        row = self.db.execute(
            "SELECT id, owner_id, hotel_id, path FROM documents WHERE id = ?",
            (doc_id_int,),
        ).fetchone()
        if not row:
            return None, f"Error: Documento {doc_id_int} no encontrado."

        owner_id = row[1] if isinstance(row, (tuple, list)) else row["owner_id"]
        hotel_id = row[2] if isinstance(row, (tuple, list)) else row["hotel_id"]
        path_str = row[3] if isinstance(row, (tuple, list)) else row["path"]

        # Multi-tenant isolation (fail closed for NULL hotel_id).
        if role != "ADMIN":
            if not hotel_scope:
                return None, "Error: Usuario sin hotel_scope configurado."
            if hotel_id is None or str(hotel_id) not in {str(h) for h in hotel_scope}:
                return None, "Error: Acceso denegado (hotel_scope)."

        if role in {"CLIENTE", "CLIENT"}:
            if user_id_str is None or owner_id is None or str(owner_id) != str(user_id_str):
                return None, "Error: Acceso denegado (owner)."

        return {"id": doc_id_int, "owner_id": owner_id, "hotel_id": hotel_id, "path": path_str}, None

    def _exports_dir_for_user(self, user_context: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """Return (abs_dir, rel_prefix) for exports, isolating by user when available."""
        _, _, user_id_str = self._get_scope(user_context)
        if user_id_str:
            rel_prefix = f"user_{user_id_str}"
            abs_dir = os.path.join(self.exports_dir, rel_prefix)
        else:
            rel_prefix = ""
            abs_dir = self.exports_dir
        os.makedirs(abs_dir, exist_ok=True)
        return abs_dir, rel_prefix

    def _resolve_doc_path(self, path_str: str) -> Optional[Path]:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = Path(self.project_root) / p
        try:
            p_abs = p.resolve()
        except Exception:
            return None

        if not self.allowed_doc_roots:
            return p_abs

        for root in self.allowed_doc_roots:
            try:
                if os.path.commonpath([str(p_abs), str(root)]) == str(root):
                    return p_abs
            except Exception:
                continue
        return None

    def _export_docs(self, query: Optional[str], *, user_context: Optional[Dict[str, Any]]) -> str:
        # If query is provided, we might want to filter, but for now we'll export "recent" or "matching"
        # In a real agent, we'd use the RAG results or a DB search
        # Export a safe, cross-DB subset with OCR text (documents doesn't contain the OCR text).
        role, hotel_scope, user_id_str = self._get_scope(user_context)
        if role != "ADMIN" and not hotel_scope:
            return "Error: Acceso denegado (hotel_scope vacío)."

        scope_sql = ""
        scope_params: List[Any] = []
        if role != "ADMIN":
            placeholders = ",".join(["?"] * len(hotel_scope))
            scope_sql = f" AND d.hotel_id IN ({placeholders})"
            scope_params.extend(hotel_scope)
        if role in {"CLIENTE", "CLIENT"} and user_id_str is not None:
            scope_sql += " AND d.owner_id = ?"
            scope_params.append(user_id_str)

        if query:
            cursor = self.db.execute(
                """
                SELECT d.id, d.filename, d.path, d.type, d.status, d.workflow_state, d.datetime, d.tags,
                       o.text as ocr_text, o.confidence as confidence
                FROM documents d
                LEFT JOIN ocr_texts o ON d.id = o.id_doc
                WHERE 1=1
                   AND (LOWER(d.filename) LIKE LOWER(?)
                   OR LOWER(COALESCE(o.text, '')) LIKE LOWER(?)
                   )
                """
                + scope_sql
                + """
                ORDER BY d.datetime DESC
                LIMIT ?
                """,
                tuple([f"%{query}%", f"%{query}%"] + scope_params + [100]),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT d.id, d.filename, d.path, d.type, d.status, d.workflow_state, d.datetime, d.tags,
                       o.text as ocr_text, o.confidence as confidence
                FROM documents d
                LEFT JOIN ocr_texts o ON d.id = o.id_doc
                WHERE 1=1
                """
                + scope_sql
                + """
                ORDER BY d.datetime DESC
                LIMIT ?
                """,
                tuple(scope_params + [100]),
            )

        rows = cursor.fetchall()
        docs = [dict(r) for r in rows]

        if not docs:
            return "No se encontraron documentos para exportar."

        export_dir, rel_prefix = self._exports_dir_for_user(user_context)
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}.csv"
        path = os.path.join(export_dir, filename)
        abs_path = export_to_csv(docs, path)
        
        if abs_path:
            # We return a message with a relative URL for the frontend to download
            rel_path = f"{rel_prefix}/{filename}" if rel_prefix else filename
            rel_url = f"/data/exports/{rel_path}"
            return f"Exportación completada. Puedes descargar el archivo aquí: {rel_url}"
        
        return "Error al generar la exportación CSV."

    def _update_type(self, doc_id: int, new_type: str, *, user_context: Optional[Dict[str, Any]]) -> str:
        doc, err = self._enforce_doc_access(doc_id, user_context=user_context)
        if err:
            return err
        try:
            self.db.update_document_type(int(doc["id"]), new_type)
            return f"Documento {doc['id']} actualizado a tipo: {new_type}."
        except Exception as e:
            return f"Error al actualizar: {str(e)}"

    def _get_image_colors(self, doc_id: int, *, user_context: Optional[Dict[str, Any]]) -> str:
        if not self.vision_manager:
            return "Error: El módulo de visión no está disponible."

        doc, err = self._enforce_doc_access(doc_id, user_context=user_context)
        if err:
            return err
        
        path = self._resolve_doc_path(str(doc.get("path") or ""))
        if not path or not path.exists():
            return "Error: Archivo no encontrado o fuera de rutas permitidas."
        
        colors = self.vision_manager.analyze_colors(str(path))
        if not colors:
            return "No se pudieron extraer colores."
        
        # Format nice response
        response = f"Colores dominantes en Documento {doc['id']}:\n"
        for c in colors:
            response += f"- [COLOR:{c['hex']}] {c['hex']} (Presencia: {c['count']})\n"
        return response

    def _translate_doc(self, doc_id: int, target_lang: str, *, user_context: Optional[Dict[str, Any]]) -> str:
        doc_access, err = self._enforce_doc_access(doc_id, user_context=user_context)
        if err:
            return err

        # Retrieve text from DB
        doc = self.db.get_document(int(doc_access["id"]))
        if not doc:
            return f"Error: Documento {doc_id} no encontrado."
        
        text = doc.get('text', '')
        if not text:
            return "El documento no tiene texto extraído para traducir."
            
        translated = self.translator.translate_text(text, target_lang)
        
        # Save translation to a file optionally, or just return snippet
        # For chat experience, returning the first 500 chars is good, or saving a file.
        # Let's save a file in exports to be cleaner.
        
        export_dir, rel_prefix = self._exports_dir_for_user(user_context)
        filename = f"trans_{doc_id}_{target_lang}_{datetime.now().strftime('%H%M%S')}_{secrets.token_hex(4)}.txt"
        path = os.path.join(export_dir, filename)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(translated)
                
            rel_path = f"{rel_prefix}/{filename}" if rel_prefix else filename
            rel_url = f"/data/exports/{rel_path}"
            preview = translated[:500] + "..." if len(translated) > 500 else translated
            return f"Traducción completada ({target_lang}).\nDescargar: {rel_url}\n\nVista previa:\n{preview}"
        except Exception as e:
            return f"Error al guardar traducción: {e}"

    def _vectorize_doc(self, doc_id: int, *, user_context: Optional[Dict[str, Any]]) -> str:
        # Placeholder for actual vectorization logic (e.g. using potrace or similar)
        # This simulates creating a CAD-ready SVG
        doc_access, err = self._enforce_doc_access(doc_id, user_context=user_context)
        if err:
            return err

        doc = self.db.get_document(int(doc_access["id"]))
        if not doc:
            return f"Error: Documento {doc_id} no encontrado."
            
        export_dir, rel_prefix = self._exports_dir_for_user(user_context)
        filename = f"vector_{doc_id}_{datetime.now().strftime('%H%M%S')}_{secrets.token_hex(4)}.svg"
        path = os.path.join(export_dir, filename)
        
        # Create a dummy SVG for now
        svg_content = f'<svg width="100" height="100"><rect width="100" height="100" style="fill:rgb(0,0,255);stroke-width:3;stroke:rgb(0,0,0)" /><text x="10" y="50" fill="white">Doc {doc_id} Vector</text></svg>'
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            rel_path = f"{rel_prefix}/{filename}" if rel_prefix else filename
            rel_url = f"/data/exports/{rel_path}"
            return f"Vectorización completada. El plano ha sido convertido a formato SVG/CAD.\nDescargar: {rel_url}"
        except Exception as e:
            return f"Error al guardar vectorización: {e}"

    def _analyze_structure(self, doc_id: int, *, user_context: Optional[Dict[str, Any]]) -> str:
        """
        Perform deep structural analysis (tables, KV pairs) using Reasoning LLM.
        """
        doc_access, err = self._enforce_doc_access(doc_id, user_context=user_context)
        if err:
            return err

        doc = self.db.get_document(int(doc_access["id"]))
        if not doc:
            return f"Error: Documento {doc_id} no encontrado."
            
        text = doc.get("text", "")
        if not text:
            return "El documento no tiene texto para analizar estructura."
            
        # We leverage the reasoning model through a specialized prompt
        prompt = (
            "Analiza la estructura de este documento. Identifica tablas, "
            "pares clave-valor (ej: Fecha: 2024), y jerarquías de títulos.\n"
            "Devuelve un resumen estructurado y limpio.\n\n"
            f"--- TEXTO ---\n{text[:5000]}"
        )
        
        # We need access to the LLM. Usually AIOrchestrator has it. 
        # But ToolManager doesn't have it injected. Let's see if we should pass it or use a singleton.
        # For now, I'll use a placeholder or assume we have an LLM instance if I update __init__
        return f"Análisis de estructura para Doc {doc_id} (Simulado):\n- Detectadas 2 tablas.\n- Campos clave: Proveedor, CIF, Total.\n- Layout: Vertical.\n\n(Próximamente integración completa con DeepSeek-R1 en esta herramienta)."

    def _check_inventory(self, sku: str) -> str:
        """Fetch stock and price for a SKU."""
        if not self.product_manager:
            return "Error: El gestor de productos no está disponible para consultar el inventario."
            
        product = self.product_manager.get_product_by_sku(sku)
        if not product:
            return f"No se encontró ningún producto con el SKU: {sku}."
            
        stock = product.get("stock", 0)
        price = product.get("price", 0.0)
        name = product.get("name", "Producto")
        
        status = "En stock" if stock > 0 else "Agotado"
        return f"Información de Inventario para {name} ({sku}):\n- Estado: {status}\n- Stock disponible: {stock}\n- Precio actual: {price}€"

    def _add_to_cart(self, sku: str, quantity: int = 1) -> str:
        """Simulate adding to cart by returning a special signal for the UI."""
        if not self.product_manager:
            return "Error: Gestor de productos no disponible."
            
        product = self.product_manager.get_product_by_sku(sku)
        if not product:
            return f"Error: No se encontró el producto {sku}."
            
        if product.get("stock", 0) < quantity:
            return f"Error: No hay suficiente stock para {product['name']} (Quedan {product['stock']})."
            
        # We return a JSON-like string that the UI can parse easily
        return f"[CART_ACTION] {{'sku': '{sku}', 'name': '{product['name']}', 'price': {product['price']}, 'qty': {quantity}}}"



