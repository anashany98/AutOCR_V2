import os
import json
import yaml
import requests
import tempfile
import threading
from typing import Optional
from pathlib import Path
from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from web_app.services import get_db, get_pipeline, get_rag_manager, get_tool_manager, get_voice_manager, get_logger, load_configuration, PROJECT_ROOT

chat_bp = Blueprint('chat', __name__)

@chat_bp.route("/api/chat", methods=["POST"])
@login_required
def api_chat_post():
    """Chat endpoint for text queries. Accepts both JSON and FormData."""
    # Handle both JSON and FormData
    image_file = None
    if request.is_json:
        data = request.json or {}
        query = data.get("query", "")
        session_id = data.get("session_id", "default_session")
        hotel_id = data.get("hotel_id")
    else:
        # FormData (for image uploads)
        query = request.form.get("query", "")
        session_id = request.form.get("session_id", "default_session")
        hotel_id = request.form.get("hotel_id")
        image_file = request.files.get("image")
    
    if not query:
        return jsonify({"error": "Query required"}), 400
        
    return process_chat_query(query, session_id, hotel_id, image_file=image_file)

@chat_bp.route("/api/chat/voice", methods=["POST"])
@login_required
def api_chat_voice():
    """Endpoint for voice queries."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    session_id = request.form.get("session_id", "default_session")
    hotel_id = request.form.get("hotel_id")
    
    # Save temp file
    temp_dir = Path(tempfile.gettempdir())
    audio_path = temp_dir / f"voice_{current_user.id}_{os.urandom(4).hex()}.wav"
    audio_file.save(str(audio_path))
    
    try:
        voice_mgr = get_voice_manager()
        if not voice_mgr or not voice_mgr.enabled:
            return jsonify({"error": "Servicio de voz no disponible localmente."}), 503
            
        # Transcribe
        query = voice_mgr.transcribe(str(audio_path))
        if not query or query.startswith("Error"):
             return jsonify({"error": query or "Could not transcribe audio"}), 500
             
        # Process as normal chat
        response = process_chat_query(query, session_id, hotel_id)
        
        # Add the transcribed text to the response
        res_data = response.get_json()
        res_data["transcription"] = query
        return jsonify(res_data)
        
    finally:
        if audio_path.exists():
            os.remove(str(audio_path))

def process_chat_query(query: str, session_id: str, hotel_id: Optional[str], image_file=None):
    """Helper to process a chat query through the AI Orchestrator or Vision Engine."""
    from web_app.services import get_orchestrator, get_db, get_rag_manager, get_prompt_manager, get_pipeline, load_configuration, get_logger
    from PIL import Image
    
    orchestrator = get_orchestrator()
    db = get_db()
    
    # --- Vision Flow ---
    if image_file:
        try:
            pipeline = get_pipeline()
            # Ensure engine is initialized
            if not pipeline.engine.enabled:
                 return jsonify({"results": [], "answer": "El motor de visión está desactivado."})
            
            # Check if engine supports chat (e.g. PaddleVLOCHEngine)
            vision_engine = pipeline.engine
            
            if not hasattr(pipeline.engine, 'chat'):
                # Fallback: Try to use PaddleVLOCHEngine explicitly for VQA
                try:
                    from modules.engines.paddle_vl_wrapper import PaddleVLOCHEngine
                    # We use a simple config for the VQA engine
                    vqa_config = load_configuration().get("ocr", {})
                    # Ensure path is set to download/load model
                    if not vqa_config.get("model_id"):
                         vqa_config["model_id"] = "PaddlePaddle/PaddleOCR-VL-1.5"
                    
                    # Singleton-like or new instance? New instance for safety/simplicity here, 
                    # ideally should be cached in app context, but let's try.
                    # Warning: This might be slow on first load.
                    get_logger().info("Initializing fallback PaddleVLOCHEngine for Vision Chat...")
                    vision_engine = PaddleVLOCHEngine(vqa_config, logger=get_logger())
                    if not vision_engine.initialize():
                         return jsonify({"results": [], "answer": "Error inicializando el motor de visión (PaddleOCR-VL)."})
                except Exception as e:
                    get_logger().error(f"Fallback VQA init failed: {e}")
                    return jsonify({
                        "results": [], 
                        "answer": f"El motor principal ({pipeline.engine.__class__.__name__}) no soporta chat y el fallback falló: {e}"
                    })

            if hasattr(vision_engine, 'chat'):
                # Load image
                try:
                    img = Image.open(image_file)
                    # Handle image mode and logic
                    answer = vision_engine.chat(img, query)
                    
                    db.insert_chat_message(session_id, "user", f"[Imagen] {query}")
                    db.insert_chat_message(session_id, "assistant", answer)
                    
                    return jsonify({
                        "results": [],
                        "answer": answer,
                        "tool_output": "Vision Analysis",
                        "orchestration": {"action": "VISION_CHAT"}
                    })
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    get_logger().error(f"Vision Chat failed: {e}\n{tb}")
                    return jsonify({"results": [], "answer": f"Error procesando la imagen: {str(e)}"})
            else:
                 return jsonify({"results": [], "answer": "Motor de visión no disponible."})
        except Exception as e:
             get_logger().error(f"Vision Flow Error: {e}")
             return jsonify({"results": [], "answer": "Error interno en flujo de visión."})

    # --- Text / Orchestrator Flow ---
    user_context = {
        "role": current_user.role,
        "hotel_scope": current_user.hotel_scope,
        "current_hotel": hotel_id
    }
    
    route = orchestrator.route_request(query, user_context)
    if route["action"] == "DENIED":
        return jsonify({"results": [], "answer": route["message"]})

    try:
        target_tool = route["tool"]
        results = []
        tool_output = None
        
        if hotel_id and str(hotel_id) not in [str(h) for h in current_user.hotel_scope] and current_user.role != 'ADMIN':
             return jsonify({"error": "Hotel access denied"}), 403

        if target_tool == "TOOL_CALL" and route.get("tool_name"):
            res_tool = orchestrator.execute_tool(route["tool_name"], route["params"])
            tool_output = res_tool.get("output", "")
            answer = f"Acción ejecutada: {route['tool_name']}. \n\nResultado: {tool_output}"
        else:
            rag = get_rag_manager()
            if target_tool in ["RAG_TEXT", "RAG_FINANCIAL", "CHAT_GENERAL"]:
                results = rag.search(query, k=5, db_manager=db, hotel_id=hotel_id)
                
            context_str = ""
            for item in results:
                context_str += f"[Doc ID: {item.get('doc_id')}] Contenido: {item.get('text')}\n\n"

            
            system_prompt = get_prompt_manager().get_prompt(current_user.role)
            if not system_prompt:
                system_prompt = get_prompt_manager().get_prompt("v1", key="CLIENTE")

            instruction = f"Contexto encontrado:\n{context_str}\n\nUsuario: {query}"
            
            llm = orchestrator.llm
            res = llm._client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.3
            )
            answer = res.choices[0].message.content
        
        db.insert_chat_message(session_id, "user", query)
        db.insert_chat_message(session_id, "assistant", answer)

        return jsonify({
            "results": results,
            "answer": answer,
            "tool_output": tool_output,
            "orchestration": route
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "results": [], 
            "answer": "⚠️ No detecto LM Studio ejecutándose. Por favor inicia el servidor local en el puerto 1234."
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        get_logger().error(f"LLM Exception: {e}\n{tb}")
        return jsonify({"results": [], "answer": f"Ocurrió un error inesperado: {str(e)}"})


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
