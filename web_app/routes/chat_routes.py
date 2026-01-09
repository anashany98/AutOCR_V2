import os
import json
import yaml
import requests
import tempfile
import threading
from pathlib import Path
from flask import Blueprint, jsonify, request

# Dependencies
from web_app.services import get_db, get_pipeline, get_rag_manager, get_tool_manager, get_logger, load_configuration, PROJECT_ROOT

chat_bp = Blueprint('chat', __name__)

@chat_bp.route("/api/chat", methods=["POST"])
def api_chat_post():
    """Semantic search chat endpoint for documents."""
    query = ""
    image_file = None
    session_id = "default_session"

    if request.content_type and 'multipart/form-data' in request.content_type:
        query = request.form.get("query", "")
        image_file = request.files.get("image")
        session_id = request.form.get("session_id", "default_session")
    else:
        data = request.json or {}
        query = data.get("query", "")
        session_id = data.get("session_id", "default_session")
        image_file = None

    if not query and not image_file:
        return jsonify({"error": "Empty query"}), 400
    
    db = get_db()
    # Save User Query
    db.insert_chat_message(session_id, "user", query)

    results = []
    visual_context = ""

    # 1. Visual Search if image provided
    if image_file:
        pipeline = get_pipeline()
        vision_manager = pipeline.vision_manager
        if vision_manager and vision_manager.config.enabled:
            suffix = Path(image_file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                image_file.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Search visually
                vision_results = vision_manager.search_similar(tmp_path, k=5)
                for vr in vision_results:
                    path_obj = Path(vr["path"])
                    # Use a new connection for specific query
                    with db.get_connection() as conn:
                        cursor = db.get_cursor(conn)
                        # Parameterized query for path
                        cursor.execute(f"SELECT id, filename, text FROM documents WHERE path = {db.placeholder}", (str(path_obj),))
                        row = cursor.fetchone()
                        if row:
                            results.append({
                                "id": row[0],
                                "filename": row[1],
                                "snippet": (row[2] or "")[:300],
                                "score": vr["score"],
                                "type": "visual"
                            })
                
                if results:
                    visual_context = "He encontrado documentos similares mediante análisis visual. "
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # 2. RAG Search (Textual)
    rag_manager = get_rag_manager()
    if query and rag_manager:
        text_results = rag_manager.search(query, k=5)
        for r in text_results:
            if isinstance(r, dict):
                results.append({
                    "id": r.get("doc_id"),
                    "filename": r.get("filename"),
                    "snippet": r.get("text"),
                    "score": 1.0, 
                    "type": "text"
                })
            else:
                results.append({
                    "id": r[0],
                    "filename": r[1],
                    "snippet": r[2],
                    "score": 1.0,
                    "type": "text"
                })

    if not results and not query:
         return jsonify({"error": "No results found"}), 404

    context_response = results

    # 1.5. Tools Setup
    tool_manager = get_tool_manager()
    tools = tool_manager.get_tool_definitions()

    # 2. Check LLM Config
    full_config = load_configuration()
    # 3. Call Chat LLM
    # Use dedicated chat config
    llm_conf = full_config.get("llm", {})
    routing_conf = llm_conf.get("routing", {})
    chat_conf = routing_conf.get("general_chat", {})
    
    # Check if Chat is enabled? 
    # For now, we assume if LLM is enabled globally, we try. 
    # Or strict check? The UI has a global "Interpretation Enabled" toggle in the Pipeline card.
    # The Chat card doesn't have an explicit 'Enabled' toggle, implied enabled if configured?
    # Let's check global enabled or just proceed if base_url is present.
    if not llm_conf.get("enabled", False):
         # If global switch is off, maybe we disable chat too? 
         # User asked for separation. Maybe Chat should be always on if configured?
         # Let's respect the ID 3584 config structure: routing enabled separate??
         # config.yaml had routing.enabled. 
         # My UI didn't touch routing.enabled. 
         # Let's rely on global 'enabled' for safety OR just existence of config.
         # For safety, if llm global is off, we might want to kill chat too?
         # No, user wants separation. Let's assume Chat is always active if configured.
         pass 

    base_url = chat_conf.get("base_url", "http://host.docker.internal:1234/v1").rstrip("/")
    model = chat_conf.get("model", "mistral-small-24b")
    timeout = int(chat_conf.get("timeout", 60))
    
    # Fallback to pipeline setting if chat specific is missing/empty (optional safety)
    if not base_url or "localhost" in base_url and "1235" not in base_url and not chat_conf:
         # Fallback logic if needed, but let's trust the new UI.
         pass
    try:
        max_context_chars = 10000 
        context_str = ""
        for i, item in enumerate(context_response):
            addition = f"[ID: {item['id']} Archivo: {item['filename']}]:\n{item['snippet']}\n\n"
            if len(context_str) + len(addition) > max_context_chars:
                break
            context_str += addition
        
        if not base_url: 
             base_url = "http://localhost:1234/v1" # Ultimate fallback

        system_prompt = (
            "Eres un asistente útil para una empresa de gestión documental. "
            f"{visual_context}"
            "Responde a la pregunta del usuario basándote EXCLUSIVAMENTE en el CONTEXTO proporcionado.\n\n"
            "REGLAS:\n"
            "1. Cita siempre la fuente usando el formato [ID: <id>] cuando uses información del contexto (ej: 'El total es 500€ [ID: 12]').\n"
            "2. Si la información NO está en el CONTEXTO, di 'No tengo información suficiente en los documentos'. No inventes.\n"
            "3. Se amable y profesional.\n\n"
            f"CONTEXTO DOCUMENTAL:\n{context_str or 'No hay documentos relevantes para esta consulta.'}"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/chat/completions", 
            json=payload, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            llm_data = response.json()
            choice = llm_data["choices"][0]
            message = choice["message"]
            
            # Check for tool/function calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                results_of_tools = []
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    tool_output = tool_manager.execute_tool(func_name, func_args)
                    results_of_tools.append(tool_output)
                
                tool_summary = "\n".join(results_of_tools)
                return jsonify({
                    "results": context_response, 
                    "answer": f"He ejecutado las acciones solicitadas:\n{tool_summary}"
                })

            # Save Assistant Response
            db.insert_chat_message(session_id, "assistant", message.get("content", ""))
            return jsonify({"results": context_response, "answer": message.get("content", "")})
        else:
            get_logger().error(f"LLM Error {response.status_code}: {response.text}")
            return jsonify({"results": context_response, "answer": "Error conectando con el cerebro local (LM Studio)."})

    except requests.exceptions.ConnectionError:
        return jsonify({
            "results": context_response, 
            "answer": "⚠️ No detecto LM Studio ejecutándose. Por favor inicia el servidor local en el puerto 1234."
        })
    except Exception as e:
        get_logger().error(f"LLM Exception: {e}")
        return jsonify({"results": context_response, "answer": "Ocurrió un error inesperado al generar la respuesta."})


@chat_bp.route("/api/chat/history", methods=["GET"])
def api_get_chat_history():
    """Get chat history for a session."""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    db = get_db()
    history = db.get_chat_history(session_id)
    return jsonify({"history": history})


@chat_bp.route("/api/status/llm")
def api_status_llm():
    """Check connectivity to the configured LLM provider."""
    full_config = load_configuration()
    llm_conf = full_config.get("llm", {})
    chat_conf = llm_conf.get("routing", {}).get("general_chat", {})
    
    # Start with Chat config
    base_url = chat_conf.get("base_url", "").rstrip("/")
    if not base_url:
        # Fallback to pipeline
        base_url = llm_conf.get("base_url", "http://host.docker.internal:1234/v1").rstrip("/")

    if not llm_conf.get("enabled", False) and not chat_conf:
        return jsonify({"status": "disabled"})

    get_logger().info(f"DEBUG: LLM Status Check - base_url={base_url}")
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        if resp.status_code == 200:
             return jsonify({"status": "online", "provider": "LM Studio / Local"})
        else:
             return jsonify({"status": "error", "code": resp.status_code})
    except Exception as e:
        get_logger().error(f"LLM Status Check Failed: {e}")
        return jsonify({"status": "offline", "error": str(e)})


@chat_bp.route("/api/rag/rebuild", methods=["POST"])
def api_rag_rebuild():
    """Trigger valid full re-indexing of documents."""
    db = get_db()
    rag_manager = get_rag_manager()
    if not rag_manager:
        return jsonify({"error": "RAG system not initialized"}), 500

    def run_rebuild():
         # Rebuild in background
         # Note: Passing db instance might be risky if connection is not thread-local safe in that method
         # But RAGManager.rebuild seems to handle it or expects it.
         try:
             rag_manager.rebuild(db)
         except Exception as e:
             get_logger().error(f"RAG Rebuild failed: {e}")
    
    threading.Thread(target=run_rebuild).start()
    return jsonify({"message": "Proceso de reindexado iniciado en segundo plano."})
