import logging
import json
from typing import Dict, Any, List, Optional
from modules.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """
    The brain of the system. Evaluates requests, checks role-based permissions 
    (provided by back-end), and triggers the appropriate local AI tools.
    """
    def __init__(self, llm_client, prompt_manager: PromptManager, tool_manager=None, product_manager=None):
        self.llm = llm_client
        self.prompts = prompt_manager
        self.tools = tool_manager
        self.products = product_manager

    def route_request(self, user_query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Analyze user intent.
        2. Validate if the intent matches user permissions.
        3. Decide which IA task to execute.
        """
        role = user_context.get("role", "CLIENTE").upper()
        
        # Step 1: Intent Analysis (using LLM as a router)
        # We explicitly ask for tool classification if relevant
        routing_prompt = (
            "Analiza la siguiente petición de un usuario de AutOCR V2.\n"
            "Clasifica el destino entre: [OCR, RAG_TEXT, RAG_FINANCIAL, VISION_DESIGN, PRODUCT_ADVISOR, CHAT_GENERAL, TOOL_CALL].\n"
            "Si el usuario pide una ACCIÓN (traducir, exportar, añadir al carrito, cambiar tipo, etc.), usa TOOL_CALL.\n"
            "Si el usuario pregunta por MUEBLES, DECORACIÓN o buscar productos, usa PRODUCT_ADVISOR.\n"
            f"Usuario Rol: {role}\n"
            f"Petición: {user_query}\n"
            "Responde SOLO con un JSON: {'target': str, 'tool_name': str, 'reason': str, 'parameters': dict}\n"
            "Si es TOOL_CALL, especifica 'tool_name' (p.ej. translate_document, add_to_cart, export_search_results_to_csv)."
        )
        
        routing_res = self.llm.chat(
            routing_prompt, 
            system_prompt="Eres el orquestador de IA de AutOCR V2. Tu objetivo es clasificar el 'target' exacto de la petición.",
            profile="reasoning"
        )
        
        try:
            intent_data = json.loads(routing_res.get("analysis", "{}"))
            target = intent_data.get("target", "CHAT_GENERAL")
        except:
            target = "CHAT_GENERAL"
            intent_data = {}

        # Step 2: Permission Enforcement (Cross-check role vs target).
        # Never trust LLM routing: enforce an explicit allowlist here and again inside ToolManager.
        if target == "RAG_FINANCIAL" and role not in ["GESTOR", "DIRECCION", "ADMIN"]:
            logger.warning("Blocking RAG_FINANCIAL for role %s", role)
            return {"action": "DENIED", "message": "No tienes permisos para esta acción."}

        if target == "TOOL_CALL":
            tool_name = str(intent_data.get("tool_name") or "").strip()
            if role in ["GESTOR", "DIRECCION", "ADMIN"]:
                pass
            elif role in ["CLIENTE", "CLIENT"]:
                allowed = {"check_inventory", "add_to_cart"}
                if tool_name not in allowed:
                    logger.warning("Blocking TOOL_CALL '%s' for role %s", tool_name, role)
                    return {"action": "DENIED", "message": "No tienes permisos para esta acción."}
            else:
                logger.warning("Blocking TOOL_CALL for unknown role %s", role)
                return {"action": "DENIED", "message": "No tienes permisos para esta acción."}
            
        # Step 3: Trigger Tool Instruction
        task_prompt = self.prompts.get_prompt(target, query=user_query)
        
        return {
            "action": "EXECUTE",
            "tool": target,
            "tool_name": intent_data.get("tool_name"),
            "prompt": task_prompt,
            "params": intent_data.get("parameters", {})
        }

    def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a specific system tool if available.
        """
        if not self.tools:
            return {"status": "error", "message": "Tool Manager no disponible."}
        
        logger.info(f"Orchestrator executing system tool: {tool_name}")
        result = self.tools.execute_tool(tool_name, params, user_context=user_context)
        return {"status": "completed", "tool": tool_name, "output": result}

    def handle_product_advice(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles conversational product search.
        1. Extract slots (Category, Attributes).
        2. Check for missing critical info.
        3. Search or Ask.
        """
        if not self.products:
            return {"answer": "Lo siento, el módulo de productos no está disponible."}

        # 1. Extraction
        extraction_prompt = (
            "Extrae los siguientes datos de la consulta del usuario (si no existen, usa null):\n"
            "- category (ej: Sofa, Mesa, Silla, Armario)\n"
            "- color (ej: Azul, Rojo, Madera)\n"
            "- material (ej: Piel, Tela, Roble)\n"
            "- size (ej: 3 plazas, 150cm, Grande)\n"
            "- style (ej: Moderno, Clásico)\n"
            f"Consulta: {user_query}\n"
            "Responde SOLO JSON."
        )
        res = self.llm.chat(extraction_prompt, system_prompt="Eres un experto en muebles.", profile="reasoning")
        try:
            slots = json.loads(res.get("analysis", "{}"))
        except:
            slots = {}

        category = slots.get("category")
        
        # 2. Logic: If category is present but missing specific details for that category?
        # For MVP: If category is "Sofá" and 'size' or 'color' is missing -> Ask.
        # But let's be more lenient for search: Use hybrid search directly.
        
        missing = []
        if category and category.lower() in ["sofa", "sofá"]:
            if not slots.get("size") and "plaza" not in user_query.lower():
                missing.append("el tamaño o número de plazas")
            if not slots.get("color"):
                missing.append("el color")
        
        if missing and False: # DISABLED FOR NOW to allow broad search first
            return {
                "answer": f"Para buscar el {category} perfecto, necesito saber: {' y '.join(missing)}.",
                "results": []
            }

        # 3. Search
        # We construct a rich query
        search_q = user_query
        # Use embedding search
        results = self.products.search_products(search_q, k=4)
        
        if not results:
            return {"answer": "No he encontrado productos que coincidan exactamente. ¿Prueba con otra descripción?", "results": []}

        # 4. Synthesize Answer
        product_summaries = "\n".join([f"- {p['name']} ({p['price']}€): {p['description']}" for p in results])
        
        final_prompt = (
            f"El usuario busca: {user_query}\n"
            f"He encontrado estos productos:\n{product_summaries}\n\n"
            "Responde al usuario presentando las mejores opciones de forma atractiva, profesional y persuasiva. "
            "Menciona precios, disponibilidad de stock y características que resuelvan su necesidad. "
            "Anima al usuario a añadir productos al carrito si alguno le gusta."
        )
        res_final = self.llm.chat(final_prompt, system_prompt="Eres un vendedor de muebles experto y amable.")
        answer = res_final.get("analysis", "Aquí tienes algunas opciones.")

        return {
            "answer": answer,
            "results": results
        }
