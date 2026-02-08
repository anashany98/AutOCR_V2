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
    def __init__(self, llm_client, prompt_manager: PromptManager, tool_manager=None):
        self.llm = llm_client
        self.prompts = prompt_manager
        self.tools = tool_manager

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
            "Clasifica el destino entre: [OCR, RAG_TEXT, RAG_FINANCIAL, VISION_DESIGN, PRODUCT_SUGGESTION, CHAT_GENERAL, TOOL_CALL].\n"
            "Si el usuario pide una ACCIÓN (traducir, exportar, cambiar tipo, etc.), usa TOOL_CALL.\n"
            f"Usuario Rol: {role}\n"
            f"Petición: {user_query}\n"
            "Responde SOLO con un JSON: {'target': str, 'tool_name': str, 'reason': str, 'parameters': dict}\n"
            "Si es TOOL_CALL, especifica 'tool_name' (p.ej. translate_document, export_search_results_to_csv, update_document_type)."
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

        # Step 2: Permission Enforcement (Cross-check role vs target)
        if target in ["RAG_FINANCIAL", "TOOL_CALL"] and role not in ["GESTOR", "DIRECCION", "ADMIN"]:
            if target == "RAG_FINANCIAL" or intent_data.get("tool_name") == "export_search_results_to_csv":
                logger.warning(f"Blocking {target} for role {role}")
                return {
                    "action": "DENIED",
                    "message": "No tienes permisos para esta acción."
                }
            
        # Step 3: Trigger Tool Instruction
        task_prompt = self.prompts.get_prompt(target, query=user_query)
        
        return {
            "action": "EXECUTE",
            "tool": target,
            "tool_name": intent_data.get("tool_name"),
            "prompt": task_prompt,
            "params": intent_data.get("parameters", {})
        }

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a specific system tool if available.
        """
        if not self.tools:
            return {"status": "error", "message": "Tool Manager no disponible."}
        
        logger.info(f"Orchestrator executing system tool: {tool_name}")
        result = self.tools.execute_tool(tool_name, params)
        return {"status": "completed", "tool": tool_name, "output": result}
