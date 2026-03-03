import os
import json
import re
import yaml
import requests
import tempfile
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional
from pathlib import Path
from flask import Blueprint, Response, jsonify, request, stream_with_context
from flask_login import current_user, login_required
from web_app.services import get_db, get_pipeline, get_rag_manager, get_tool_manager, get_voice_manager, get_logger, load_configuration, PROJECT_ROOT
from web_app.security.security_decorators import require_role

chat_bp = Blueprint('chat', __name__)

_LLM_STATUS_CACHE = {"ts": 0.0, "payload": None}
_LLM_STATUS_CACHE_LOCK = threading.Lock()
_CHAT_ADMISSION_LOCK = threading.Lock()
_USER_ACTIVE_REQUESTS: Dict[str, int] = {}
_USER_RATE_WINDOWS: Dict[str, Deque[float]] = {}
_CHAT_ALERT_CACHE: Dict[str, float] = {}
_CHAT_ALERT_CACHE_LOCK = threading.Lock()
_CHAT_MAINTENANCE_LOCK = threading.Lock()
_LAST_CHAT_PURGE_TS = 0.0


def _now_ts() -> float:
    import time
    return float(time.time())


def _admit_user_request(user_id: str, tuning: Dict[str, object]):
    """
    Admission control for chat endpoints.

    Enforces:
    - requests per minute per user
    - max concurrent in-flight requests per user
    """
    now = _now_ts()
    rpm = max(1, int(tuning.get("requests_per_minute", 40) or 40))
    max_concurrent = max(1, int(tuning.get("max_concurrent_requests_per_user", 2) or 2))

    with _CHAT_ADMISSION_LOCK:
        dq = _USER_RATE_WINDOWS.get(user_id)
        if dq is None:
            dq = deque()
            _USER_RATE_WINDOWS[user_id] = dq

        cutoff = now - 60.0
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= rpm:
            return False, {
                "error": "Rate limit exceeded",
                "message": "Demasiadas solicitudes en un minuto. Intenta de nuevo en unos segundos.",
            }

        active = int(_USER_ACTIVE_REQUESTS.get(user_id, 0) or 0)
        if active >= max_concurrent:
            return False, {
                "error": "Too many concurrent requests",
                "message": "Tienes demasiadas solicitudes simultaneas en curso. Espera a que termine una.",
            }

        dq.append(now)
        _USER_ACTIVE_REQUESTS[user_id] = active + 1
        return True, None


def _release_user_request(user_id: str):
    with _CHAT_ADMISSION_LOCK:
        active = int(_USER_ACTIVE_REQUESTS.get(user_id, 0) or 0)
        if active <= 1:
            _USER_ACTIVE_REQUESTS.pop(user_id, None)
        else:
            _USER_ACTIVE_REQUESTS[user_id] = active - 1


def _check_async_backpressure(db, user_id: str, tuning: Dict[str, object]):
    """Backpressure guard for async chat queue depth."""
    max_user = max(1, int(tuning.get("max_async_tasks_per_user", 4) or 4))
    max_global = max(1, int(tuning.get("max_async_tasks_global", 200) or 200))
    recent_seconds = max(60, int(tuning.get("async_recent_window_seconds", 900) or 900))

    pending_user = db.count_chat_tasks(
        user_id=user_id,
        statuses=("processing",),
        recent_seconds=recent_seconds,
    )
    if pending_user >= max_user:
        return False, {
            "error": "Async queue saturated for user",
            "message": "Tu cola de tareas de chat esta llena. Espera a que termine alguna y reintenta.",
            "pending_user_tasks": pending_user,
        }

    pending_global = db.count_chat_tasks(
        statuses=("processing",),
        recent_seconds=recent_seconds,
    )
    if pending_global >= max_global:
        return False, {
            "error": "Async queue saturated",
            "message": "La cola global de chat esta saturada temporalmente. Reintenta en breve.",
            "pending_global_tasks": pending_global,
        }

    return True, {
        "pending_user_tasks": pending_user,
        "pending_global_tasks": pending_global,
    }


def _record_chat_metric(
    db,
    *,
    user_id: str,
    endpoint: str,
    status_code: int,
    started_ts: float,
):
    try:
        queue_depth = db.count_chat_tasks(statuses=("processing",), recent_seconds=900)
    except Exception:
        queue_depth = None

    try:
        duration_ms = (_now_ts() - float(started_ts or _now_ts())) * 1000.0
    except Exception:
        duration_ms = 0.0

    try:
        db.insert_chat_metric(
            user_id=str(user_id),
            endpoint=str(endpoint),
            status_code=int(status_code),
            duration_ms=float(duration_ms),
            queue_depth=queue_depth,
        )
    except Exception:
        pass


def _maybe_persist_alert(db, *, code: str, message: str, value: float, threshold: float, cooldown_s: float):
    now = _now_ts()
    with _CHAT_ALERT_CACHE_LOCK:
        last_ts = float(_CHAT_ALERT_CACHE.get(code, 0.0) or 0.0)
        if (now - last_ts) < float(max(0.0, cooldown_s)):
            return
        _CHAT_ALERT_CACHE[code] = now
    try:
        db.insert_chat_alert(
            level="WARNING",
            code=code,
            message=message,
            metric_value=float(value),
            threshold_value=float(threshold),
        )
    except Exception:
        pass


def _emit_chat_slo_alerts(db, tuning: Dict[str, object], *, endpoint: str):
    try:
        window_minutes = max(1, int(tuning.get("metrics_window_minutes", 15) or 15))
        summary = db.get_chat_metrics_summary(window_minutes=window_minutes, endpoint=endpoint)
    except Exception:
        return

    cooldown = float(tuning.get("alert_cooldown_seconds", 300.0) or 300.0)

    p95 = float(((summary.get("latency_ms") or {}).get("p95") or 0.0))
    p99 = float(((summary.get("latency_ms") or {}).get("p99") or 0.0))
    err_rate = float(summary.get("error_rate_pct") or 0.0)
    qmax = float(((summary.get("queue_depth") or {}).get("max") or 0.0))

    p95_th = float(tuning.get("alert_p95_ms", 15000.0) or 15000.0)
    p99_th = float(tuning.get("alert_p99_ms", 30000.0) or 30000.0)
    err_th = float(tuning.get("alert_error_rate_pct", 10.0) or 10.0)
    q_th = float(tuning.get("alert_queue_depth", 50.0) or 50.0)

    if p95 > p95_th:
        _maybe_persist_alert(
            db,
            code=f"{endpoint}:p95_latency_high",
            message=f"p95 latency high on {endpoint}: {p95:.0f}ms > {p95_th:.0f}ms",
            value=p95,
            threshold=p95_th,
            cooldown_s=cooldown,
        )
    if p99 > p99_th:
        _maybe_persist_alert(
            db,
            code=f"{endpoint}:p99_latency_high",
            message=f"p99 latency high on {endpoint}: {p99:.0f}ms > {p99_th:.0f}ms",
            value=p99,
            threshold=p99_th,
            cooldown_s=cooldown,
        )
    if err_rate > err_th:
        _maybe_persist_alert(
            db,
            code=f"{endpoint}:error_rate_high",
            message=f"error rate high on {endpoint}: {err_rate:.2f}% > {err_th:.2f}%",
            value=err_rate,
            threshold=err_th,
            cooldown_s=cooldown,
        )
    if qmax > q_th:
        _maybe_persist_alert(
            db,
            code=f"{endpoint}:queue_depth_high",
            message=f"queue depth high on {endpoint}: {qmax:.0f} > {q_th:.0f}",
            value=qmax,
            threshold=q_th,
            cooldown_s=cooldown,
        )


def _purge_old_chat_state(db, tuning: Dict[str, object]):
    """Best-effort TTL cleanup of chat history/task rows."""
    try:
        history_days = float(tuning.get("history_ttl_days", 30.0) or 30.0)
        task_days = float(tuning.get("task_ttl_days", 7.0) or 7.0)
        db.purge_chat_history_older_than(days=history_days)
        db.purge_chat_tasks_older_than(days=task_days, only_terminal=True)
    except Exception:
        pass


def _maybe_run_chat_ttl_purge(db, tuning: Dict[str, object], *, min_interval_seconds: int = 600):
    """Run TTL purge at most every `min_interval_seconds` per process."""
    global _LAST_CHAT_PURGE_TS
    now = _now_ts()
    if (now - float(_LAST_CHAT_PURGE_TS or 0.0)) < float(max(30, int(min_interval_seconds))):
        return
    with _CHAT_MAINTENANCE_LOCK:
        now = _now_ts()
        if (now - float(_LAST_CHAT_PURGE_TS or 0.0)) < float(max(30, int(min_interval_seconds))):
            return
        _purge_old_chat_state(db, tuning)
        _LAST_CHAT_PURGE_TS = now


def _attach_stream_teardown(
    resp: Response,
    *,
    db,
    user_id: str,
    endpoint: str,
    started_ts: float,
    tuning: Dict[str, object],
    status_code: int = 200,
):
    """Attach metric + admission teardown to stream response close."""
    def _on_close():
        _record_chat_metric(
            db,
            user_id=user_id,
            endpoint=endpoint,
            status_code=int(status_code),
            started_ts=started_ts,
        )
        _emit_chat_slo_alerts(db, tuning, endpoint=endpoint)
        _release_user_request(user_id)

    try:
        resp.call_on_close(_on_close)
    except Exception:
        _on_close()
    return resp


def _get_chat_tuning():
    """Return chat tuning values from config with safe defaults."""
    cfg = load_configuration() or {}
    chat_conf = cfg.get("chat", {}) or {}
    if not isinstance(chat_conf, dict):
        chat_conf = {}

    def _as_int(value, default):
        try:
            v = int(value)
            return v if v > 0 else default
        except Exception:
            return default

    def _as_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    def _as_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
        return default

    return {
        "fast_router": bool(chat_conf.get("fast_router", True)),
        "rag_k": _as_int(chat_conf.get("rag_k", 3), 3),
        "rag_hybrid_search": _as_bool(chat_conf.get("rag_hybrid_search", True), True),
        "rag_rerank": _as_bool(chat_conf.get("rag_rerank", True), True),
        "rag_incremental_rebuild": _as_bool(chat_conf.get("rag_incremental_rebuild", True), True),
        "context_chars_per_chunk": _as_int(chat_conf.get("context_chars_per_chunk", 800), 800),
        "max_total_context_chars": _as_int(chat_conf.get("max_total_context_chars", 6000), 6000),
        "history_messages": _as_int(chat_conf.get("history_messages", 6), 6),
        "max_history_chars": _as_int(chat_conf.get("max_history_chars", 1400), 1400),
        "history_ttl_days": _as_float(chat_conf.get("history_ttl_days", 30), 30.0),
        "task_ttl_days": _as_float(chat_conf.get("task_ttl_days", 7), 7.0),
        "llm_profile_with_context": str(chat_conf.get("llm_profile_with_context", "document_chat")),
        "llm_profile_no_context": str(chat_conf.get("llm_profile_no_context", "general_chat")),
        "status_cache_seconds": _as_float(chat_conf.get("status_cache_seconds", 5.0), 5.0),
        "async_enabled": _as_bool(chat_conf.get("async_enabled", True), True),
        "requests_per_minute": _as_int(chat_conf.get("requests_per_minute", 40), 40),
        "max_concurrent_requests_per_user": _as_int(chat_conf.get("max_concurrent_requests_per_user", 2), 2),
        "max_async_tasks_per_user": _as_int(chat_conf.get("max_async_tasks_per_user", 4), 4),
        "max_async_tasks_global": _as_int(chat_conf.get("max_async_tasks_global", 200), 200),
        "async_recent_window_seconds": _as_int(chat_conf.get("async_recent_window_seconds", 900), 900),
        "metrics_window_minutes": _as_int(chat_conf.get("metrics_window_minutes", 15), 15),
        "alert_p95_ms": _as_float(chat_conf.get("alert_p95_ms", 15000), 15000.0),
        "alert_p99_ms": _as_float(chat_conf.get("alert_p99_ms", 30000), 30000.0),
        "alert_error_rate_pct": _as_float(chat_conf.get("alert_error_rate_pct", 10.0), 10.0),
        "alert_queue_depth": _as_float(chat_conf.get("alert_queue_depth", 50.0), 50.0),
        "alert_cooldown_seconds": _as_float(chat_conf.get("alert_cooldown_seconds", 300.0), 300.0),
    }


def _looks_like_product_query(query: str) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    keywords = (
        "sofa",
        "sofÃƒÂ¡",
        "sillÃƒÂ³n",
        "sillon",
        "silla",
        "mesa",
        "cama",
        "armario",
        "mueble",
        "decor",
        "estilo",
        "color",
        "material",
        "sku",
        "precio",
        "comprar",
        "carrito",
    )
    return any(k in q for k in keywords)


def _extract_sku_and_qty(query: str):
    import re

    q = (query or "").strip()
    if not q:
        return None, None

    sku = None
    m = re.search(r"\bsku\s*[:#]?\s*([A-Za-z0-9_-]{3,64})\b", q, re.IGNORECASE)
    if m:
        sku = m.group(1)
    else:
        # Fallback: allow "SOFA-CHE-001" or similar patterns.
        m2 = re.search(r"\b([A-Za-z]{2,10}-[A-Za-z0-9]{2,20}-[A-Za-z0-9]{2,20})\b", q)
        if m2:
            sku = m2.group(1)

    qty = None
    m3 = re.search(r"\b(?:x|qty|cantidad)\s*(\d{1,3})\b", q, re.IGNORECASE)
    if m3:
        try:
            qty = max(1, min(999, int(m3.group(1))))
        except Exception:
            qty = None

    return sku, qty


def _filter_financial_results(results, *, role: str, db):
    """Remove financial docs from RAG results for roles that must not see them."""
    role_u = str(role or "").upper()
    if role_u in {"DIRECCION", "ADMIN"}:
        return results

    ids = []
    for item in results or []:
        doc_id = item.get("doc_id") or item.get("id")
        try:
            ids.append(int(doc_id))
        except Exception:
            continue
    ids = sorted(set(ids))
    if not ids:
        return results

    placeholders = ",".join([db.placeholder] * len(ids))
    q = f"SELECT id, financial_level FROM documents WHERE id IN ({placeholders})"
    try:
        with db.get_connection() as conn:
            cur = db.get_cursor(conn)
            cur.execute(q, tuple(ids))
            rows = cur.fetchall()
    except Exception:
        return results

    restricted = set()
    for row in rows:
        doc_id = row[0] if isinstance(row, (tuple, list)) else row["id"]
        level = row[1] if isinstance(row, (tuple, list)) else row["financial_level"]
        if level and str(level).lower() != "none":
            restricted.add(int(doc_id))

    if not restricted:
        return results

    filtered = []
    for item in results or []:
        doc_id = item.get("doc_id") or item.get("id")
        try:
            if int(doc_id) in restricted:
                continue
        except Exception:
            pass
        filtered.append(item)
    return filtered


def _build_context(results, *, chars_per_chunk: int, max_total: int) -> str:
    if not results:
        return ""
    chunks = []
    total = 0
    for item in results:
        doc_id = item.get("doc_id") or item.get("id")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        snippet = text[: max(1, chars_per_chunk)]
        piece = f"[Doc ID: {doc_id}] Contenido: {snippet}\n\n"
        if total + len(piece) > max_total:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _build_recent_history_context(
    db,
    session_id: str,
    *,
    user_id: str,
    max_messages: int,
    max_chars: int,
    drop_last_user: bool = False,
) -> str:
    """Build a compact recent-history block for conversational continuity."""
    try:
        history = db.get_chat_history(session_id, limit=max(1, int(max_messages)), user_id=str(user_id)) or []
    except Exception:
        history = []

    if not history:
        return ""

    if drop_last_user and history and str(history[-1].get("role", "")).lower() == "user":
        history = history[:-1]
    if not history:
        return ""

    cap = max(0, int(max_chars))
    selected = []
    total_chars = 0

    for msg in reversed(history):
        role_raw = str(msg.get("role", "")).lower()
        role_label = "Usuario" if role_raw == "user" else "Asistente"
        content = re.sub(r"\s+", " ", str(msg.get("content") or "").strip())
        if not content:
            continue
        if len(content) > 240:
            content = content[:240].rstrip() + "..."
        line = f"{role_label}: {content}"

        if cap > 0:
            if selected and total_chars + len(line) + 1 > cap:
                break
            if not selected and len(line) > cap:
                line = line[: max(0, cap - 3)].rstrip() + "..."
            total_chars += len(line) + 1

        selected.append(line)

    if not selected:
        return ""
    selected.reverse()
    return "\n".join(selected)


def _slim_results_for_ui(results, *, snippet_chars: int = 200):
    slim = []
    for item in results or []:
        text = (item.get("text") or "").strip()
        slim.append(
            {
                "doc_id": item.get("doc_id") or item.get("id"),
                "filename": item.get("filename"),
                "score": item.get("score"),
                "text": text[:snippet_chars] if text else "",
            }
        )
    return slim


@chat_bp.route("/api/chat", methods=["POST"])
@login_required
def api_chat_post():
    """Chat endpoint for text queries. Accepts both JSON and FormData."""
    started_ts = _now_ts()
    user_id = str(current_user.id)
    tuning = _get_chat_tuning()
    db = get_db()
    endpoint = "/api/chat"

    admitted, reject_payload = _admit_user_request(user_id, tuning)
    if not admitted:
        _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=429, started_ts=started_ts)
        return jsonify(reject_payload), 429

    status_code = 200
    try:
        _maybe_run_chat_ttl_purge(db, tuning)

        image_file = None
        request_async = False
        default_session_id = f"user_{current_user.id}_default"
        if request.is_json:
            data = request.json or {}
            query = data.get("query", "")
            session_id = (data.get("session_id") or default_session_id).strip()
            hotel_id = data.get("hotel_id")
            doc_id = data.get("doc_id")
            request_async = str(data.get("async", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            query = request.form.get("query", "")
            session_id = (request.form.get("session_id") or default_session_id).strip()
            hotel_id = request.form.get("hotel_id")
            doc_id = request.form.get("doc_id")
            image_file = request.files.get("image")
            request_async = str(request.form.get("async", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}

        if not query and not image_file:
            status_code = 400
            return jsonify({"error": "Query required"}), 400

        if not session_id or len(session_id) > 128:
            status_code = 400
            return jsonify({"error": "Invalid session_id"}), 400

        user_context = {
            "role": current_user.role,
            "hotel_scope": current_user.hotel_scope,
            "current_hotel": hotel_id,
            "user_id": user_id,
        }

        if image_file:
            response = process_chat_query_sync(query, session_id, hotel_id, image_file=image_file, doc_id=doc_id)
            if isinstance(response, tuple) and len(response) >= 2:
                try:
                    status_code = int(response[1])
                except Exception:
                    status_code = 200
            else:
                try:
                    status_code = int(getattr(response, "status_code", 200) or 200)
                except Exception:
                    status_code = 200
            return response

        if request_async and tuning.get("async_enabled", True):
            allowed, queue_state = _check_async_backpressure(db, user_id, tuning)
            if not allowed:
                status_code = 429
                return jsonify(queue_state), 429
            try:
                from modules.tasks import process_chat_async, use_celery_backend

                task_res = process_chat_async(query, session_id, hotel_id, doc_id, user_context)
                task_id = str(getattr(task_res, "id", "") or "")
                if task_id:
                    backend = "celery" if use_celery_backend() else "huey"
                    db.register_chat_task(
                        task_id=task_id,
                        user_id=user_id,
                        session_id=session_id,
                        backend=backend,
                    )
                    status_code = 202
                    return jsonify({"status": "processing", "task_id": task_id, **(queue_state or {})}), 202
            except Exception as exc:
                get_logger().error("Async chat dispatch failed, falling back to sync: %s", exc)

        response = process_chat_query_sync(query, session_id, hotel_id, doc_id=doc_id)
        if isinstance(response, tuple) and len(response) >= 2:
            try:
                status_code = int(response[1])
            except Exception:
                status_code = 200
        else:
            try:
                status_code = int(getattr(response, "status_code", 200) or 200)
            except Exception:
                status_code = 200
        return response
    finally:
        _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=status_code, started_ts=started_ts)
        _emit_chat_slo_alerts(db, tuning, endpoint=endpoint)
        _release_user_request(user_id)


@chat_bp.route("/api/chat/stream", methods=["POST"])
@login_required
def api_chat_stream():
    """Streaming chat endpoint (NDJSON). Text-only for instant UX."""
    started_ts = _now_ts()
    user_id = str(current_user.id)
    tuning = _get_chat_tuning()
    db = get_db()
    endpoint = "/api/chat/stream"

    admitted, reject_payload = _admit_user_request(user_id, tuning)
    if not admitted:
        _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=429, started_ts=started_ts)
        return jsonify(reject_payload), 429

    status_code = 200
    release_in_finally = True
    try:
        _maybe_run_chat_ttl_purge(db, tuning)

        if request.is_json:
            data = request.json or {}
            query = (data.get("query") or "").strip()
            session_id = (data.get("session_id") or f"user_{current_user.id}_default").strip()
            hotel_id = data.get("hotel_id")
            doc_id = data.get("doc_id")
        else:
            status_code = 400
            return jsonify({"error": "JSON required"}), 400

        if not query:
            status_code = 400
            return jsonify({"error": "Query required"}), 400
        if not session_id or len(session_id) > 128:
            status_code = 400
            return jsonify({"error": "Invalid session_id"}), 400

        role = str(getattr(current_user, "role", "")).upper()

        scope_list = []
        for h in (getattr(current_user, "hotel_scope", []) or []):
            try:
                scope_list.append(int(h))
            except Exception:
                continue
        if role != "ADMIN" and not scope_list:
            status_code = 403
            return jsonify({"error": "User has no hotel scope configured"}), 403

        requested_hotel_id = None
        if hotel_id:
            try:
                requested_hotel_id = int(hotel_id)
            except Exception:
                status_code = 400
                return jsonify({"error": "Invalid hotel_id"}), 400
            if role != "ADMIN" and requested_hotel_id not in set(scope_list):
                status_code = 403
                return jsonify({"error": "Hotel access denied"}), 403

        effective_hotel_ids = None if role == "ADMIN" else ([requested_hotel_id] if requested_hotel_id is not None else scope_list)
        effective_owner_id = int(current_user.id) if role in {"CLIENTE", "CLIENT"} else None

        db.insert_chat_message(session_id, "user", query, user_id=user_id)

        tool_output = None
        orchestration = {"action": "EXECUTE", "tool": "CHAT_STREAM", "fast_router": True}

        sku, qty = _extract_sku_and_qty(query)
        ql = query.lower()
        if sku and any(k in ql for k in ("carrito", "add to cart", "anade al carrito", "anade al carrito")):
            tool_output = get_tool_manager().execute_tool(
                "add_to_cart",
                {"sku": sku, "quantity": qty or 1},
                user_context={
                    "role": current_user.role,
                    "hotel_scope": current_user.hotel_scope,
                    "current_hotel": hotel_id,
                    "user_id": user_id,
                },
            )
            answer = tool_output
            db.insert_chat_message(session_id, "assistant", answer, user_id=user_id)

            def _tool_stream():
                yield json.dumps({"type": "final", "answer": answer, "results": [], "tool_output": tool_output, "orchestration": orchestration}, ensure_ascii=False) + "\n"

            resp = Response(stream_with_context(_tool_stream()), mimetype="application/x-ndjson")
            release_in_finally = False
            return _attach_stream_teardown(
                resp,
                db=db,
                user_id=user_id,
                endpoint=endpoint,
                started_ts=started_ts,
                tuning=tuning,
                status_code=200,
            )

        if sku and any(k in ql for k in ("inventario", "stock", "precio", "disponible", "availability", "check inventory")):
            tool_output = get_tool_manager().execute_tool(
                "check_inventory",
                {"sku": sku},
                user_context={
                    "role": current_user.role,
                    "hotel_scope": current_user.hotel_scope,
                    "current_hotel": hotel_id,
                    "user_id": user_id,
                },
            )
            answer = tool_output
            db.insert_chat_message(session_id, "assistant", answer, user_id=user_id)

            def _tool_stream():
                yield json.dumps({"type": "final", "answer": answer, "results": [], "tool_output": tool_output, "orchestration": orchestration}, ensure_ascii=False) + "\n"

            resp = Response(stream_with_context(_tool_stream()), mimetype="application/x-ndjson")
            release_in_finally = False
            return _attach_stream_teardown(
                resp,
                db=db,
                user_id=user_id,
                endpoint=endpoint,
                started_ts=started_ts,
                tuning=tuning,
                status_code=200,
            )

        if _looks_like_product_query(query):
            try:
                from web_app.services import get_orchestrator
                orchestrator = get_orchestrator()
                advisor_res = orchestrator.handle_product_advice(
                    query,
                    {
                        "role": current_user.role,
                        "hotel_scope": current_user.hotel_scope,
                        "current_hotel": hotel_id,
                        "user_id": user_id,
                    },
                )
                results = advisor_res.get("results", [])
                answer = advisor_res.get("answer", "No encontre productos.")
            except Exception as exc:
                get_logger().error("Product advisor failed: %s", exc)
                results = []
                answer = "Error interno en busqueda de productos."

            db.insert_chat_message(session_id, "assistant", answer, user_id=user_id)

            def _product_stream():
                yield json.dumps({"type": "final", "answer": answer, "results": results, "tool_output": "Product Search", "orchestration": {"action": "EXECUTE", "tool": "PRODUCT_ADVISOR", "fast_router": True}}, ensure_ascii=False) + "\n"

            resp = Response(stream_with_context(_product_stream()), mimetype="application/x-ndjson")
            release_in_finally = False
            return _attach_stream_teardown(
                resp,
                db=db,
                user_id=user_id,
                endpoint=endpoint,
                started_ts=started_ts,
                tuning=tuning,
                status_code=200,
            )

        rag = get_rag_manager()
        results = []
        if doc_id:
            try:
                doc_id_int = int(doc_id)
            except Exception:
                status_code = 400
                return jsonify({"error": "Invalid doc_id"}), 400

            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                cursor.execute(
                    f"""
                    SELECT d.owner_id, d.hotel_id, o.text, d.filename, d.financial_level
                    FROM documents d
                    LEFT JOIN ocr_texts o ON d.id = o.id_doc
                    WHERE d.id = {db.placeholder}
                    """,
                    (doc_id_int,),
                )
                row = cursor.fetchone()
            if not row:
                status_code = 404
                return jsonify({"results": [], "answer": "Documento no encontrado."}), 404

            owner_id = row[0]
            doc_hotel_id = row[1]
            text_val = row[2] or ""
            filename_val = row[3] or ""
            financial_level = row[4] or "none"

            if role in {"CLIENTE", "CLIENT"} and str(owner_id) != user_id:
                status_code = 403
                return jsonify({"error": "Document access denied"}), 403
            if role != "ADMIN":
                if doc_hotel_id is None:
                    status_code = 403
                    return jsonify({"error": "Hotel access denied"}), 403
                if str(doc_hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                    status_code = 403
                    return jsonify({"error": "Hotel access denied"}), 403

            if financial_level and str(financial_level).lower() != "none" and role not in {"DIRECCION", "ADMIN"}:
                text_val = ""

            if text_val.strip():
                results = [{"doc_id": doc_id_int, "text": text_val, "filename": filename_val, "score": 1.0}]
        else:
            if rag:
                results = rag.search(
                    query,
                    k=int(tuning["rag_k"]),
                    db_manager=db,
                    owner_id=effective_owner_id,
                    hotel_ids=effective_hotel_ids,
                    hybrid=bool(tuning.get("rag_hybrid_search", True)),
                    rerank=bool(tuning.get("rag_rerank", True)),
                )

        results = _filter_financial_results(results, role=role, db=db)

        context_str = _build_context(
            results,
            chars_per_chunk=int(tuning["context_chars_per_chunk"]),
            max_total=int(tuning["max_total_context_chars"]),
        )
        history_context = _build_recent_history_context(
            db,
            session_id,
            user_id=user_id,
            max_messages=int(tuning.get("history_messages", 6)),
            max_chars=int(tuning.get("max_history_chars", 1400)),
            drop_last_user=True,
        )

        from web_app.services import get_orchestrator, get_prompt_manager
        orchestrator = get_orchestrator()
        system_prompt = get_prompt_manager().get_prompt(current_user.role) or get_prompt_manager().get_prompt("v1", key="CLIENTE")

        if context_str:
            blocks = []
            if history_context:
                blocks.append(f"Conversacion reciente:\n{history_context}")
            blocks.append(f"Contexto encontrado:\n{context_str}")
            blocks.append(f"Usuario: {query}")
            instruction = "\n\n".join(blocks)
            profile = tuning["llm_profile_with_context"] or "general_chat"
        else:
            if history_context:
                instruction = f"Conversacion reciente:\n{history_context}\n\nUsuario: {query}"
            else:
                instruction = query
            profile = tuning["llm_profile_no_context"] or "general_chat"

        ui_results = _slim_results_for_ui(results, snippet_chars=200)

        def _generate():
            answer_parts = []
            yield json.dumps(
                {"type": "meta", "results": ui_results, "tool_output": tool_output, "orchestration": orchestration},
                ensure_ascii=False,
            ) + "\n"

            try:
                for chunk in orchestrator.llm.chat_stream(
                    user_prompt=instruction,
                    system_prompt=system_prompt,
                    profile=profile,
                ):
                    if not chunk:
                        continue
                    chunk_str = str(chunk)
                    answer_parts.append(chunk_str)
                    yield json.dumps({"type": "delta", "content": chunk_str}, ensure_ascii=False) + "\n"
            except Exception as exc:
                get_logger().error("Chat stream failed: %s", exc, exc_info=True)

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = "Lo siento, no he podido generar una respuesta."

            try:
                db.insert_chat_message(session_id, "assistant", answer, user_id=user_id)
            except Exception:
                pass

            yield json.dumps(
                {"type": "final", "answer": answer, "results": ui_results, "tool_output": tool_output, "orchestration": orchestration},
                ensure_ascii=False,
            ) + "\n"

        resp = Response(stream_with_context(_generate()), mimetype="application/x-ndjson")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        release_in_finally = False
        return _attach_stream_teardown(
            resp,
            db=db,
            user_id=user_id,
            endpoint=endpoint,
            started_ts=started_ts,
            tuning=tuning,
            status_code=200,
        )
    finally:
        if release_in_finally:
            _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=status_code, started_ts=started_ts)
            _emit_chat_slo_alerts(db, tuning, endpoint=endpoint)
            _release_user_request(user_id)
@chat_bp.route("/api/chat/status/<task_id>", methods=["GET"])
@login_required
def api_chat_status(task_id):
    task_id = str(task_id or "").strip()
    if not task_id or len(task_id) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", task_id):
        return jsonify({"error": "Invalid task_id"}), 400

    db = get_db()
    task = db.get_chat_task(task_id, user_id=str(current_user.id))
    if not task:
        # Fail closed to avoid cross-user task-id probing.
        return jsonify({"error": "Task not found"}), 404

    if task.get("status") == "completed" and task.get("result") is not None:
        return jsonify({"status": "completed", "result": task["result"]})
    if task.get("status") == "failed":
        return jsonify({"status": "failed", "error": task.get("error") or "Task failed"})

    backend = str(task.get("backend") or "").lower()
    from modules.tasks import use_celery_backend

    if backend == "celery" or (not backend and use_celery_backend()):
        from celery.result import AsyncResult

        res = AsyncResult(task_id)
        if res.state == "SUCCESS":
            payload = res.result
            if isinstance(payload, dict):
                db.update_chat_task_status(task_id, "completed", result=payload)
                return jsonify({"status": "completed", "result": payload})
            db.update_chat_task_status(task_id, "completed", result={"answer": str(payload)})
            return jsonify({"status": "completed", "result": {"answer": str(payload)}})
        if res.state == "FAILURE":
            err = str(res.result)
            db.update_chat_task_status(task_id, "failed", error=err)
            return jsonify({"status": "failed", "error": err})
        return jsonify({"status": "processing"})

    # Default dev backend: Huey.
    try:
        from modules.tasks import huey

        payload = huey.result(task_id, blocking=False, preserve=True)
    except Exception as exc:
        get_logger().error(f"Huey status check failed for task {task_id}: {exc}")
        return jsonify({"status": "processing"})

    if payload is None:
        return jsonify({"status": "processing"})

    if isinstance(payload, dict) and payload.get("error") is True:
        err = str(payload.get("answer") or payload.get("error") or "Task failed")
        db.update_chat_task_status(task_id, "failed", error=err)
        return jsonify({"status": "failed", "error": err})

    if isinstance(payload, dict):
        db.update_chat_task_status(task_id, "completed", result=payload)
        return jsonify({"status": "completed", "result": payload})

    # Generic fallback for non-dict payloads.
    payload_dict = {"answer": str(payload)}
    db.update_chat_task_status(task_id, "completed", result=payload_dict)
    return jsonify({"status": "completed", "result": payload_dict})



@chat_bp.route("/api/chat/voice", methods=["POST"])
@login_required
def api_chat_voice():
    """Endpoint for voice queries."""
    started_ts = _now_ts()
    user_id = str(current_user.id)
    tuning = _get_chat_tuning()
    db = get_db()
    endpoint = "/api/chat/voice"

    admitted, reject_payload = _admit_user_request(user_id, tuning)
    if not admitted:
        _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=429, started_ts=started_ts)
        return jsonify(reject_payload), 429

    status_code = 200
    try:
        _maybe_run_chat_ttl_purge(db, tuning)

        if "audio" not in request.files:
            status_code = 400
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        default_session_id = f"user_{current_user.id}_default"
        session_id = (request.form.get("session_id") or default_session_id).strip()
        hotel_id = request.form.get("hotel_id")

        if not session_id or len(session_id) > 128:
            status_code = 400
            return jsonify({"error": "Invalid session_id"}), 400

        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"voice_{current_user.id}_{os.urandom(4).hex()}.wav"
        audio_file.save(str(audio_path))

        try:
            voice_mgr = get_voice_manager()
            if not voice_mgr or not voice_mgr.enabled:
                status_code = 503
                return jsonify({"error": "Servicio de voz no disponible localmente."}), 503

            query = voice_mgr.transcribe(str(audio_path))
            if not query or str(query).startswith("Error"):
                status_code = 500
                return jsonify({"error": query or "Could not transcribe audio"}), 500

            response = process_chat_query_sync(query, session_id, hotel_id)
            if isinstance(response, tuple) and len(response) >= 2:
                payload = response[0].get_json() if hasattr(response[0], "get_json") else {}
                payload = payload or {}
                payload["transcription"] = query
                status_code = int(response[1]) if response[1] is not None else 200
                return jsonify(payload), status_code

            status_code = int(getattr(response, "status_code", 200) or 200)
            res_data = response.get_json() if hasattr(response, "get_json") else {}
            res_data = res_data or {}
            res_data["transcription"] = query
            return jsonify(res_data), status_code
        finally:
            if audio_path.exists():
                os.remove(str(audio_path))
    finally:
        _record_chat_metric(db, user_id=user_id, endpoint=endpoint, status_code=status_code, started_ts=started_ts)
        _emit_chat_slo_alerts(db, tuning, endpoint=endpoint)
        _release_user_request(user_id)
def process_chat_query_sync(query: str, session_id: str, hotel_id: Optional[str], image_file=None, doc_id=None):
    """Helper to process a chat query through the AI Orchestrator or Vision Engine."""
    from web_app.services import get_orchestrator, get_db, get_rag_manager, get_prompt_manager, get_pipeline, load_configuration, get_logger
    from PIL import Image

    session_id = (session_id or f"user_{current_user.id}_default").strip()
    if not session_id or len(session_id) > 128:
        return jsonify({"error": "Invalid session_id"}), 400
    
    orchestrator = get_orchestrator()
    db = get_db()
    role = str(getattr(current_user, "role", "")).upper()
    tuning = _get_chat_tuning()

    # Tenant scoping for RAG: when hotel_id is not provided (the default UI case),
    # we search across the user's entire hotel_scope. For clients, also scope by owner_id.
    scope_list = []
    for h in (getattr(current_user, "hotel_scope", []) or []):
        try:
            scope_list.append(int(h))
        except Exception:
            continue
    if role != "ADMIN" and not scope_list:
        return jsonify({"error": "User has no hotel scope configured"}), 403

    requested_hotel_id = None
    if hotel_id:
        try:
            requested_hotel_id = int(hotel_id)
        except Exception:
            return jsonify({"error": "Invalid hotel_id"}), 400
        if role != "ADMIN" and requested_hotel_id not in set(scope_list):
            return jsonify({"error": "Hotel access denied"}), 403

    effective_hotel_ids = None if role == "ADMIN" else ([requested_hotel_id] if requested_hotel_id is not None else scope_list)
    effective_owner_id = int(current_user.id) if role in {"CLIENTE", "CLIENT"} else None
    
    # --- Vision Flow ---
    if image_file:
        tmp_path = None
        try:
            # Save upload to a temp path (avoid NamedTemporaryFile locking on Windows).
            suffix = Path(getattr(image_file, "filename", "") or "").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                suffix = ".jpg"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"autocr_chat_{current_user.id}_")
            os.close(fd)

            try:
                image_file.seek(0)
            except Exception:
                pass
            image_file.save(tmp_path)

            ql = (query or "").lower()
            is_search_intent = (not ql) or any(
                k in ql
                for k in (
                    "busca",
                    "encuentra",
                    "comprar",
                    "similar",
                    "parecido",
                    "find",
                    "search",
                    "buy",
                    "quiero",
                )
            )
            is_ocr_request = (not ql) or any(
                k in ql
                for k in (
                    "ocr",
                    "texto",
                    "lee",
                    "leer",
                    "transcribe",
                    "transcribir",
                    "extrae",
                    "extraer",
                    "que pone",
                    "quÃƒÂ© pone",
                )
            )

            # 1) Visual search (optional; only when it likely helps).
            visual_search_results = []
            if is_search_intent:
                try:
                    from web_app.services import get_product_manager

                    pm = get_product_manager()
                    if pm and getattr(pm, "vision_manager", None):
                        visual_search_results = pm.search_products(tmp_path, k=3)
                except Exception as e:
                    get_logger().error(f"Visual Search failed: {e}")

            # 2) OCR (always; it's the reliable fallback for image chat).
            pipeline = get_pipeline()
            ocr_text = ""
            ocr_conf = 0.0
            is_handwritten = False
            try:
                text_val, _lang, conf_val, hw = pipeline.ocr_manager.extract_text(tmp_path)
                ocr_text = (text_val or "").strip()
                ocr_conf = float(conf_val or 0.0)
                is_handwritten = bool(hw)
            except Exception as e:
                get_logger().error(f"Image OCR failed: {e}")

            # 3) VQA (only if PaddleVL is enabled AND the question is visual, not OCR/text-only).
            vqa_answer = ""
            vlm_engine = None
            try:
                vlm_engine = getattr(pipeline.ocr_manager, "extra_engines", {}).get("paddlevl")
            except Exception:
                vlm_engine = None

            if ql and vlm_engine and hasattr(vlm_engine, "chat") and not is_ocr_request:
                if any(
                    k in ql
                    for k in (
                        "describe",
                        "quÃƒÂ© ves",
                        "que ves",
                        "quÃƒÂ© hay",
                        "que hay",
                        "en la imagen",
                        "en la foto",
                        "color",
                        "material",
                        "estilo",
                    )
                ):
                    try:
                        with Image.open(tmp_path) as im:
                            im_rgb = im.convert("RGB")
                        vqa_answer = str(vlm_engine.chat(im_rgb, query)).strip()
                    except Exception as e:
                        get_logger().error(f"Image VQA failed: {e}")

            tool_output = "Image OCR"
            orchestration = {"action": "VISION_OCR"}

            if vqa_answer:
                answer = vqa_answer
                tool_output = "Image VQA"
                orchestration = {"action": "VISION_VQA"}
            elif not ql:
                answer = ocr_text or "No detectÃƒÂ© texto en la imagen."
            elif is_ocr_request:
                answer = ocr_text or "No detectÃƒÂ© texto en la imagen."
            else:
                # Question about the document/image: try to answer using text LLM with OCR context.
                if ocr_text:
                    system_prompt = (
                        get_prompt_manager().get_prompt(current_user.role)
                        or get_prompt_manager().get_prompt("v1", key="CLIENTE")
                    )
                    max_chars = int(tuning.get("max_total_context_chars", 6000))
                    ocr_context = ocr_text
                    if max_chars > 0 and len(ocr_context) > max_chars:
                        ocr_context = ocr_context[:max_chars] + "\n...[truncado]"

                    instruction = (
                        "Texto OCR de la imagen:\n"
                        f"(confianza={ocr_conf:.2f}, manuscrito={is_handwritten})\n"
                        f"{ocr_context}\n\n"
                        f"Pregunta: {query}"
                    )
                    llm_res = orchestrator.llm.chat(
                        user_prompt=instruction,
                        system_prompt=system_prompt,
                        profile=tuning.get("llm_profile_with_context") or "document_chat",
                    )
                    answer = (llm_res.get("analysis") or "").strip()
                    if not answer:
                        answer = ocr_text
                    tool_output = "Image OCR + LLM"
                    orchestration = {"action": "VISION_OCR_QA"}
                else:
                    answer = "No detectÃƒÂ© texto en la imagen."
                    tool_output = "Image OCR"
                    orchestration = {"action": "VISION_OCR_EMPTY"}

            # Merge product results into the final response (UI renders cards when tool_output mentions Visual Search).
            if visual_search_results:
                tool_output = f"{tool_output} + Visual Search"
                orchestration = {**orchestration, "visual_search": True}
                try:
                    product_list = "\n".join(
                        [
                            f"- {p.get('name', '')} ({p.get('price', '?')}Ã¢â€šÂ¬) [SKU: {p.get('sku', '')}]"
                            for p in (visual_search_results or [])
                        ]
                    ).strip()
                    if product_list:
                        answer = (answer or "").rstrip() + "\n\nProductos similares encontrados:\n" + product_list
                except Exception:
                    pass

            db.insert_chat_message(session_id, "user", f"[Imagen] {query}".strip(), user_id=str(current_user.id))
            db.insert_chat_message(session_id, "assistant", answer, user_id=str(current_user.id))

            return jsonify(
                {
                    "results": visual_search_results,
                    "answer": answer,
                    "tool_output": tool_output,
                    "orchestration": orchestration,
                }
            )

        except Exception as e:
            get_logger().error(f"Vision Flow Error: {e}", exc_info=True)
            return jsonify({"results": [], "answer": "Error interno en flujo de visiÃƒÂ³n."})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # --- Text / Orchestrator Flow ---
    user_context = {
        "role": current_user.role,
        "hotel_scope": current_user.hotel_scope,
        "current_hotel": hotel_id,
        "user_id": str(current_user.id),
    }

    # Fast-path: avoid the LLM router call for the common case (cuts one full LLM round-trip).
    if bool(tuning.get("fast_router", True)):
        try:
            results = []
            tool_output = None

            sku, qty = _extract_sku_and_qty(query)
            ql = (query or "").lower()

            if sku and any(k in ql for k in ("carrito", "add to cart", "aÃƒÂ±ade al carrito", "anade al carrito")):
                tool_output = get_tool_manager().execute_tool(
                    "add_to_cart", {"sku": sku, "quantity": qty or 1}, user_context=user_context
                )
                answer = tool_output
                route = {"action": "EXECUTE", "tool": "TOOL_CALL", "tool_name": "add_to_cart", "fast_router": True}

            elif sku and any(k in ql for k in ("inventario", "stock", "precio", "disponible", "availability", "check inventory")):
                tool_output = get_tool_manager().execute_tool(
                    "check_inventory", {"sku": sku}, user_context=user_context
                )
                answer = tool_output
                route = {"action": "EXECUTE", "tool": "TOOL_CALL", "tool_name": "check_inventory", "fast_router": True}

            elif _looks_like_product_query(query):
                advisor_res = orchestrator.handle_product_advice(query, user_context)
                results = advisor_res.get("results", [])
                answer = advisor_res.get("answer", "No encontrÃƒÂ© productos.")
                tool_output = "Product Search"
                route = {"action": "EXECUTE", "tool": "PRODUCT_ADVISOR", "fast_router": True}

            else:
                rag = get_rag_manager()
                if doc_id:
                    try:
                        doc_id_int = int(doc_id)
                    except Exception:
                        return jsonify({"error": "Invalid doc_id"}), 400

                    with db.get_connection() as conn:
                        cursor = db.get_cursor(conn)
                        cursor.execute(
                            f"""
                            SELECT d.owner_id, d.hotel_id, o.text, d.filename, d.financial_level
                            FROM documents d
                            LEFT JOIN ocr_texts o ON d.id = o.id_doc
                            WHERE d.id = {db.placeholder}
                            """,
                            (doc_id_int,),
                        )
                        row = cursor.fetchone()

                    if not row:
                        return jsonify({"results": [], "answer": "Documento no encontrado."})

                    owner_id = row[0]
                    doc_hotel_id = row[1]
                    text_val = row[2] or ""
                    filename_val = row[3] or ""
                    financial_level = row[4] or "none"

                    if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
                        return jsonify({"error": "Document access denied"}), 403
                    if role != "ADMIN":
                        if doc_hotel_id is None:
                            return jsonify({"error": "Hotel access denied"}), 403
                        if str(doc_hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                            return jsonify({"error": "Hotel access denied"}), 403

                    if financial_level and str(financial_level).lower() != "none" and role not in {"DIRECCION", "ADMIN"}:
                        answer = "Acceso financiero restringido para este documento."
                        results = []
                        route = {"action": "EXECUTE", "tool": "DENIED_FINANCIAL", "fast_router": True}
                    else:
                        if text_val.strip():
                            results = [{"doc_id": doc_id_int, "text": text_val, "filename": filename_val, "score": 1.0}]
                        else:
                            return jsonify({"results": [], "answer": "Documento no encontrado o sin texto."})
                else:
                    if rag:
                        results = rag.search(
                            query,
                            k=int(tuning.get("rag_k", 3)),
                            db_manager=db,
                            owner_id=effective_owner_id,
                            hotel_ids=effective_hotel_ids,
                            hybrid=bool(tuning.get("rag_hybrid_search", True)),
                            rerank=bool(tuning.get("rag_rerank", True)),
                        )

                if not locals().get("answer"):
                    results = _filter_financial_results(results, role=role, db=db)
                    history_context = _build_recent_history_context(
                        db,
                        session_id,
                        user_id=str(current_user.id),
                        max_messages=int(tuning.get("history_messages", 6)),
                        max_chars=int(tuning.get("max_history_chars", 1400)),
                    )

                    context_str = _build_context(
                        results,
                        chars_per_chunk=int(tuning.get("context_chars_per_chunk", 800)),
                        max_total=int(tuning.get("max_total_context_chars", 6000)),
                    )

                    system_prompt = get_prompt_manager().get_prompt(current_user.role) or get_prompt_manager().get_prompt("v1", key="CLIENTE")

                    if context_str:
                        blocks = []
                        if history_context:
                            blocks.append(f"Conversacion reciente:\n{history_context}")
                        blocks.append(f"Contexto encontrado:\n{context_str}")
                        blocks.append(f"Usuario: {query}")
                        instruction = "\n\n".join(blocks)
                        profile = tuning.get("llm_profile_with_context") or "general_chat"
                        route = {"action": "EXECUTE", "tool": "RAG_TEXT", "fast_router": True}
                    else:
                        if history_context:
                            instruction = f"Conversacion reciente:\n{history_context}\n\nUsuario: {query}"
                        else:
                            instruction = query
                        profile = tuning.get("llm_profile_no_context") or "general_chat"
                        route = {"action": "EXECUTE", "tool": "CHAT_GENERAL", "fast_router": True}

                    llm_res = orchestrator.llm.chat(
                        user_prompt=instruction,
                        system_prompt=system_prompt,
                        profile=profile,
                    )
                    answer = llm_res.get("analysis", "Lo siento, no he podido generar una respuesta.")

            db.insert_chat_message(session_id, "user", query, user_id=str(current_user.id))
            db.insert_chat_message(session_id, "assistant", answer, user_id=str(current_user.id))

            return jsonify(
                {
                    "results": _slim_results_for_ui(results, snippet_chars=200),
                    "answer": answer,
                    "tool_output": tool_output,
                    "orchestration": route,
                }
            )

        except requests.exceptions.ConnectionError:
            return jsonify(
                {
                    "results": [],
                    "answer": "Ã¢Å¡Â Ã¯Â¸Â No detecto LM Studio ejecutÃƒÂ¡ndose. Por favor inicia el servidor local en el puerto 1234.",
                }
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            get_logger().error(f"LLM Exception (fast-path): {e}\n{tb}")
            return jsonify({"results": [], "answer": f"OcurriÃƒÂ³ un error inesperado: {str(e)}"})
    
    route = orchestrator.route_request(query, user_context)
    if route["action"] == "DENIED":
        return jsonify({"results": [], "answer": route["message"]})

    try:
        target_tool = route["tool"]
        results = []
        tool_output = None
        
        # Note: RAG scoping is enforced via effective_hotel_ids/effective_owner_id (fail closed).

        if target_tool == "TOOL_CALL" and route.get("tool_name"):
            res_tool = orchestrator.execute_tool(
                route["tool_name"], route["params"], user_context=user_context
            )
            tool_output = res_tool.get("output", "")
            answer = f"AcciÃƒÂ³n ejecutada: {route['tool_name']}. \n\nResultado: {tool_output}"
        
        elif target_tool == "PRODUCT_ADVISOR":
            # [NEW] Product Advisor Flow
            advisor_res = orchestrator.handle_product_advice(query, user_context)
            results = advisor_res.get("results", [])
            answer = advisor_res.get("answer", "No encontrÃƒÂ© productos.")
            tool_output = "Product Search"

        else:
            rag = get_rag_manager()
            if target_tool in ["RAG_TEXT", "RAG_FINANCIAL", "CHAT_GENERAL"]:
                # [NEW] Contextual Chat
                if doc_id:
                     try:
                        try:
                            doc_id_int = int(doc_id)
                        except Exception:
                            return jsonify({"error": "Invalid doc_id"}), 400

                        # Fetch document + enforce access controls (owner + hotel scope).
                        with db.get_connection() as conn:
                            cursor = db.get_cursor(conn)
                            cursor.execute(
                                f"""
                                SELECT d.owner_id, d.hotel_id, o.text, d.filename
                                FROM documents d
                                LEFT JOIN ocr_texts o ON d.id = o.id_doc
                                WHERE d.id = {db.placeholder}
                                """,
                                (doc_id_int,),
                            )
                            row = cursor.fetchone()

                        if not row:
                            return jsonify({"results": [], "answer": "Documento no encontrado."})

                        owner_id = row[0]
                        doc_hotel_id = row[1]
                        text_val = row[2] or ""
                        filename_val = row[3] or ""

                        role = str(getattr(current_user, "role", "")).upper()
                        if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
                            return jsonify({"error": "Document access denied"}), 403
                        if role != "ADMIN":
                            # Fail closed if the document has no hotel_id (legacy/unscoped record).
                            if doc_hotel_id is None:
                                return jsonify({"error": "Hotel access denied"}), 403
                            if str(doc_hotel_id) not in [str(h) for h in current_user.hotel_scope]:
                                return jsonify({"error": "Hotel access denied"}), 403

                        if text_val.strip():
                            results = [{
                                "doc_id": doc_id_int,
                                "text": text_val,
                                "filename": filename_val,
                                "score": 1.0
                            }]
                        else:
                            return jsonify({"results": [], "answer": "Documento no encontrado o sin texto."})
                     except Exception as e:
                         get_logger().error(f"Failed to fetch doc context: {e}")
                else:
                    # Normal RAG
                    if rag:
                        results = rag.search(
                            query,
                            k=5,
                            db_manager=db,
                            owner_id=effective_owner_id,
                            hotel_ids=effective_hotel_ids,
                            hybrid=bool(tuning.get("rag_hybrid_search", True)),
                            rerank=bool(tuning.get("rag_rerank", True)),
                        )
                
            history_context = _build_recent_history_context(
                db,
                session_id,
                user_id=str(current_user.id),
                max_messages=int(tuning.get("history_messages", 6)),
                max_chars=int(tuning.get("max_history_chars", 1400)),
            )
            context_str = _build_context(
                results,
                chars_per_chunk=int(tuning.get("context_chars_per_chunk", 800)),
                max_total=int(tuning.get("max_total_context_chars", 6000)),
            )

            system_prompt = get_prompt_manager().get_prompt(current_user.role)
            if not system_prompt:
                system_prompt = get_prompt_manager().get_prompt("v1", key="CLIENTE")

            if context_str:
                blocks = []
                if history_context:
                    blocks.append(f"Conversacion reciente:\n{history_context}")
                blocks.append(f"Contexto encontrado:\n{context_str}")
                blocks.append(f"Usuario: {query}")
                instruction = "\n\n".join(blocks)
                profile = tuning.get("llm_profile_with_context") or "general_chat"
            else:
                if history_context:
                    instruction = f"Conversacion reciente:\n{history_context}\n\nUsuario: {query}"
                else:
                    instruction = query
                profile = tuning.get("llm_profile_no_context") or "general_chat"
            
            # Use the high-level chat() method to benefit from cleanup and profiles
            llm_res = orchestrator.llm.chat(
                user_prompt=instruction,
                system_prompt=system_prompt,
                profile=profile,
            )
            answer = llm_res.get("analysis", "Lo siento, no he podido generar una respuesta.")
        
        db.insert_chat_message(session_id, "user", query, user_id=str(current_user.id))
        db.insert_chat_message(session_id, "assistant", answer, user_id=str(current_user.id))

        return jsonify({
            "results": results,
            "answer": answer,
            "tool_output": tool_output,
            "orchestration": route
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "results": [], 
            "answer": "Ã¢Å¡Â Ã¯Â¸Â No detecto LM Studio ejecutÃƒÂ¡ndose. Por favor inicia el servidor local en el puerto 1234."
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        get_logger().error(f"LLM Exception: {e}\n{tb}")
        return jsonify({"results": [], "answer": f"OcurriÃƒÂ³ un error inesperado: {str(e)}"})


@chat_bp.route("/api/chat/history", methods=["GET"])
@login_required
def api_get_chat_history():
    """Get chat history for a session."""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    session_id = str(session_id).strip()
    if not session_id or len(session_id) > 128:
        return jsonify({"error": "Invalid session_id"}), 400
    
    db = get_db()
    history = db.get_chat_history(session_id, limit=200, user_id=str(current_user.id))
    return jsonify({"history": history})


@chat_bp.route("/api/chat/health", methods=["GET"])
@login_required
def api_chat_health():
    """Operational chat health snapshot (latency/error/queue/alerts)."""
    tuning = _get_chat_tuning()
    db = get_db()
    window_minutes = max(1, int(tuning.get("metrics_window_minutes", 15) or 15))

    summary_all = db.get_chat_metrics_summary(window_minutes=window_minutes)
    summary_stream = db.get_chat_metrics_summary(window_minutes=window_minutes, endpoint="/api/chat/stream")
    summary_sync = db.get_chat_metrics_summary(window_minutes=window_minutes, endpoint="/api/chat")
    summary_voice = db.get_chat_metrics_summary(window_minutes=window_minutes, endpoint="/api/chat/voice")

    pending_tasks = db.count_chat_tasks(
        statuses=("processing",),
        recent_seconds=max(60, int(tuning.get("async_recent_window_seconds", 900) or 900)),
    )

    alerts = db.get_recent_chat_alerts(limit=20)
    return jsonify(
        {
            "window_minutes": window_minutes,
            "pending_tasks": pending_tasks,
            "summaries": {
                "all": summary_all,
                "sync": summary_sync,
                "stream": summary_stream,
                "voice": summary_voice,
            },
            "alerts": alerts,
        }
    )

@chat_bp.route("/api/status/llm")
@login_required
def api_status_llm():
    """Check connectivity to the configured LLM provider."""
    tuning = _get_chat_tuning()
    ttl = float(tuning.get("status_cache_seconds", 5.0) or 0.0)
    now_ts = _now_ts()

    if ttl > 0:
        with _LLM_STATUS_CACHE_LOCK:
            cached = _LLM_STATUS_CACHE.get("payload")
            cached_ts = float(_LLM_STATUS_CACHE.get("ts") or 0.0)
            if cached and (now_ts - cached_ts) < ttl:
                return jsonify(cached)

    full_config = load_configuration()
    llm_conf = full_config.get("llm", {})
    chat_conf = llm_conf.get("routing", {}).get("general_chat", {})
    
    # Start with Chat config
    base_url = chat_conf.get("base_url", "").rstrip("/")
    if not base_url:
        # Fallback to pipeline
        base_url = llm_conf.get("base_url", "http://host.docker.internal:1234/v1").rstrip("/")

    if not llm_conf.get("enabled", False) and not chat_conf:
        payload = {"status": "disabled"}
        if ttl > 0:
            with _LLM_STATUS_CACHE_LOCK:
                _LLM_STATUS_CACHE["ts"] = now_ts
                _LLM_STATUS_CACHE["payload"] = payload
        return jsonify(payload)

    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        if resp.status_code == 200:
            payload = {"status": "online", "provider": "LM Studio / Local"}
        else:
            payload = {"status": "error", "code": resp.status_code}
    except Exception as e:
        payload = {"status": "offline", "error": str(e)}

    if ttl > 0:
        with _LLM_STATUS_CACHE_LOCK:
            _LLM_STATUS_CACHE["ts"] = now_ts
            _LLM_STATUS_CACHE["payload"] = payload
    return jsonify(payload)


@chat_bp.route("/api/rag/rebuild", methods=["POST"])
@login_required
@require_role(["DIRECCION", "ADMIN"])
def api_rag_rebuild():
    """Trigger RAG index rebuild (incremental by default)."""
    db = get_db()
    rag_manager = get_rag_manager()
    if not rag_manager:
        return jsonify({"error": "RAG system not initialized"}), 500

    tuning = _get_chat_tuning()
    body = request.get_json(silent=True) if request.is_json else {}
    body = body or {}
    force_full = str(body.get("full", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    incremental_mode = bool(tuning.get("rag_incremental_rebuild", True)) and not force_full

    def run_rebuild(incremental_flag: bool):
        try:
            rag_manager.rebuild(db, incremental=incremental_flag)
        except Exception as exc:
            get_logger().error("RAG Rebuild failed: %s", exc)

    threading.Thread(target=run_rebuild, args=(incremental_mode,), daemon=True).start()

    mode = "incremental" if incremental_mode else "full"
    return jsonify({"message": f"Proceso de reindexado {mode} iniciado en segundo plano.", "mode": mode})


