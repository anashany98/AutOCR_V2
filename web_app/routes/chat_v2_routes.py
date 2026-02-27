"""
Document AI Chat API — Blueprint for RAG-powered chat.

New endpoints for the Document AI platform:
  POST /api/v2/chat/query          — Ask a question with RAG retrieval
  GET  /api/v2/chat/sessions       — List chat sessions
  GET  /api/v2/chat/sessions/<id>  — Get session messages
  POST /api/v2/chat/sessions       — Create a new session
  GET  /api/v2/chat/sources/<id>   — Get source chunks for a message
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import Blueprint, Response, jsonify, request, stream_with_context
from flask_login import login_required, current_user

logger = logging.getLogger(__name__)

chat_v2_bp = Blueprint("chat_v2", __name__, url_prefix="/api/v2/chat")


def _get_services():
    """Lazy import to avoid circular dependencies."""
    from web_app.services import get_db, get_llm_client
    return get_db(), get_llm_client()


def _get_context_builder():
    """Get or create the context builder singleton."""
    from web_app.services import get_db
    from modules.context_builder import ContextBuilder
    from pipeline.embedding_step import EmbeddingStep

    db = get_db()
    embedding = EmbeddingStep(db=db)
    return ContextBuilder(db=db, embedding_step=embedding)


def _get_current_user():
    """Get current authenticated user info."""
    try:
        from flask_login import current_user
        if current_user.is_authenticated:
            return {
                "id": str(getattr(current_user, "id", "")),
                "role": getattr(current_user, "role", "CLIENTE"),
                "hotel_scope": getattr(current_user, "hotel_scope", None),
                "tenant_id": getattr(current_user, "tenant_id", None),
            }
    except Exception:
        pass
    return None


# ============================================================================
# POST /api/v2/chat/query — Ask a question with RAG
# ============================================================================

@chat_v2_bp.route("/query", methods=["POST"])
@login_required
def chat_query():
    """
    Process a chat query with RAG retrieval and LLM generation.

    Request JSON:
        - query (str): The user's question
        - session_id (str, optional): Existing session ID
        - hotel_ids (list[str], optional): Filter by hotels
        - doc_type (str, optional): Filter by document type
        - stream (bool, optional): Stream the response (default: false)

    Response JSON:
        - answer (str): The LLM's response
        - sources (list): Source documents used
        - session_id (str): Session ID for follow-up
        - retrieval_time_ms (int): Retrieval latency
    """
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    tenant_id = data.get("tenant_id") or (user or {}).get("tenant_id")
    if not tenant_id:
        return jsonify({"error": "Tenant ID is required"}), 400
    hotel_ids = data.get("hotel_ids") or (user or {}).get("hotel_scope")
    session_id = data.get("session_id") or str(uuid.uuid4())
    stream_mode = data.get("stream", False)

    try:
        db, llm = _get_services()
        ctx_builder = _get_context_builder()

        # 1. Retrieve relevant context
        ctx_result = ctx_builder.retrieve(
            query,
            tenant_id=tenant_id,
            hotel_ids=hotel_ids,
            doc_type_filter=data.get("doc_type"),
        )

        # 2. Build chat history
        history = _get_session_history(db, session_id, limit=6)

        # 3. Build system prompt with context
        system_prompt = _build_system_prompt(ctx_result.context_text)

        # 4. Save user message
        _save_message(db, session_id, tenant_id, user, "user", query)

        # 5. Generate response
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})

        if stream_mode:
            return _stream_response(
                llm, messages, db, session_id, tenant_id, user,
                ctx_result, query
            )

        # Non-streaming
        t0 = time.perf_counter()
        answer = llm.chat(messages, profile="document_chat")
        gen_time_ms = int((time.perf_counter() - t0) * 1000)

        # Save assistant response
        sources_json = [
            {
                "document_id": s["document_id"],
                "filename": s["filename"],
                "pages": s["pages"],
            }
            for s in ctx_result.source_docs
        ]

        _save_message(
            db, session_id, tenant_id, user, "assistant", answer,
            sources=sources_json,
        )

        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "sources": sources_json,
            "retrieval_time_ms": ctx_result.retrieval_time_ms,
            "generation_time_ms": gen_time_ms,
            "chunks_used": len(ctx_result.chunks),
        })

    except Exception as e:
        logger.error("Chat query failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


def _stream_response(llm, messages, db, session_id, tenant_id, user, ctx_result, query):
    """Stream LLM response via SSE."""
    def generate():
        full_answer = []
        try:
            for chunk in llm.chat_stream(messages, profile="document_chat"):
                full_answer.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            answer = "".join(full_answer)
            sources_json = [
                {"document_id": s["document_id"], "filename": s["filename"], "pages": s["pages"]}
                for s in ctx_result.source_docs
            ]

            _save_message(db, session_id, tenant_id, user, "assistant", answer, sources=sources_json)

            yield f"data: {json.dumps({'type': 'done', 'sources': sources_json})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# GET/POST /api/v2/chat/sessions — Session management
# ============================================================================

@chat_v2_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List chat sessions for the current user."""
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401

    try:
        db, _ = _get_services()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, is_active, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT 50
                """,
                (user["id"],),
            )
            rows = cursor.fetchall()
            sessions = [
                {
                    "id": str(r[0]),
                    "title": r[1],
                    "is_active": r[2],
                    "created_at": str(r[3]),
                    "updated_at": str(r[4]),
                }
                for r in rows
            ]
        return jsonify({"sessions": sessions})

    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        return jsonify({"error": str(e)}), 500


@chat_v2_bp.route("/sessions", methods=["POST"])
def create_session():
    """Create a new chat session."""
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json(force=True, silent=True) or {}
    tenant_id = data.get("tenant_id") or user.get("tenant_id")
    if not tenant_id:
        return jsonify({"error": "Tenant ID is required"}), 400
    title = data.get("title", "Nueva conversación")
    hotel_id = data.get("hotel_id")

    try:
        db, _ = _get_services()
        session_id = str(uuid.uuid4())

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_sessions (id, tenant_id, user_id, hotel_id, title)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, tenant_id, user["id"], hotel_id, title),
            )
            conn.commit()

        return jsonify({"session_id": session_id, "title": title}), 201

    except Exception as e:
        logger.error("Failed to create session: %s", e)
        return jsonify({"error": str(e)}), 500


@chat_v2_bp.route("/sessions/<session_id>", methods=["GET"])
@login_required
def get_session_messages(session_id: str):
    """Get messages for a specific session."""
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401

    try:
        db, _ = _get_services()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, role, content, sources, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT 200
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = [
                {
                    "id": str(r[0]),
                    "role": r[1],
                    "content": r[2],
                    "sources": json.loads(r[3]) if r[3] else [],
                    "created_at": str(r[4]),
                }
                for r in rows
            ]
        return jsonify({"session_id": session_id, "messages": messages})

    except Exception as e:
        logger.error("Failed to get session messages: %s", e)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# GET /api/v2/chat/sources/<message_id> — Get source chunks
# ============================================================================

@chat_v2_bp.route("/sources/<message_id>", methods=["GET"])
@login_required
def get_message_sources(message_id: str):
    """Get source chunks and documents for a specific message."""
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    try:
        db, _ = _get_services()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sources FROM chat_messages WHERE id = %s",
                (message_id,),
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "Message not found"}), 404

            sources = json.loads(row[0]) if row[0] else []

        return jsonify({"message_id": message_id, "sources": sources})

    except Exception as e:
        logger.error("Failed to get message sources: %s", e)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Helpers
# ============================================================================

def _build_system_prompt(context: str) -> str:
    """Build the system prompt with RAG context."""
    if context:
        return (
            "Eres un asistente inteligente de documentos. "
            "Responde las preguntas basándote en el contexto proporcionado. "
            "Si la información no está en el contexto, dilo claramente. "
            "Cita las fuentes usando el formato [Fuente N]. "
            "Responde en español.\n\n"
            f"--- CONTEXTO ---\n{context}\n--- FIN CONTEXTO ---"
        )
    return (
        "Eres un asistente inteligente de documentos. "
        "No se encontraron documentos relevantes para esta consulta. "
        "Responde basándote en tu conocimiento general. "
        "Responde en español."
    )


def _get_session_history(db: Any, session_id: str, limit: int = 6) -> list:
    """Get recent messages from a session for conversation context."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []


def _save_message(
    db: Any,
    session_id: str,
    tenant_id: str,
    user: Optional[Dict],
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> None:
    """Save a chat message to the database."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Ensure session exists
            cursor.execute(
                """
                INSERT INTO chat_sessions (id, tenant_id, user_id, title)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = NOW()
                """,
                (
                    session_id,
                    tenant_id,
                    (user or {}).get("id", "anonymous"),
                    content[:50] + "..." if len(content) > 50 else content,
                ),
            )

            cursor.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, sources)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(sources) if sources else None,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to save message: %s", e)


__all__ = ["chat_v2_bp"]
