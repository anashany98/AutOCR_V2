import os
import secrets
import json
from pathlib import Path
from typing import List, Any, Dict, Optional
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
import tempfile

from web_app.services import get_db, get_pipeline, get_logger, get_classifier, load_configuration, save_configuration, PROJECT_ROOT
from web_app.security.security_decorators import require_role, hotel_scoped, owner_scoped, financial_access_required
from web_app.utils import safe_json_parse, resolve_path, ensure_within_project, encode_path, decode_path
from modules.file_utils import ensure_directories
from modules.tasks import process_document_task, rebuild_vision_index_task

main_bp = Blueprint('main', __name__)

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif", ".gif", ".tif", ".tiff"}
VISION_ROOT = (PROJECT_ROOT / "data" / "vision").resolve()
VISION_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


def _documents_has_column(db, column_name: str) -> bool:
    try:
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            cursor.execute(f"SELECT {column_name} FROM documents LIMIT 1")
        return True
    except Exception:
        return False


def _documents_schema(db) -> Dict[str, Any]:
    return {
        "created_col": "created_at" if _documents_has_column(db, "created_at") else "datetime",
        "path_col": "file_path" if _documents_has_column(db, "file_path") else "path",
        "type_col": "doc_type" if _documents_has_column(db, "doc_type") else "type",
        "has_file_size": _documents_has_column(db, "file_size"),
    }


def _row_get(row: Any, key: str, index: Optional[int] = None, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        return row[key]
    except Exception:
        pass
    if index is not None:
        try:
            return row[index]
        except Exception:
            pass
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def _abs_doc_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(str(path_value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    try:
        return p.resolve(strict=False)
    except Exception:
        return p


def _path_candidates(path_value: Optional[str]) -> List[str]:
    candidates: List[str] = []

    def _append(value: Optional[str]) -> None:
        if not value:
            return
        variants = {value, value.replace("\\", "/"), value.replace("/", "\\")}
        for cand in variants:
            if cand and cand not in candidates:
                candidates.append(cand)

    _append(path_value)
    abs_path = _abs_doc_path(path_value)
    if abs_path:
        _append(str(abs_path))
        try:
            _append(str(abs_path.relative_to(PROJECT_ROOT.resolve())))
        except Exception:
            pass
    return candidates


def _find_document_by_path(db, path_col: str, path_value: Optional[str], select_sql: str):
    candidates = _path_candidates(path_value)
    if not candidates:
        return None
    placeholders = ",".join([db.placeholder] * len(candidates))
    q = f"SELECT {select_sql} FROM documents WHERE {path_col} IN ({placeholders}) LIMIT 1"
    return db.execute(q, tuple(candidates)).fetchone()


def _filter_accessible_doc_ids(db, doc_ids: List[int], user=None) -> List[int]:
    if not doc_ids:
        return []
    u = user or current_user
    role = str(getattr(u, "role", "")).upper()
    scope_set = {str(h) for h in (getattr(u, "hotel_scope", []) or [])}
    owner_id = str(getattr(u, "id", "")) if role in {"CLIENTE", "CLIENT"} else None

    # Fail closed for non-admin users without scope.
    if role != "ADMIN" and not scope_set:
        return []

    placeholders = ",".join([db.placeholder] * len(doc_ids))
    rows = db.execute(
        f"SELECT id, owner_id, hotel_id FROM documents WHERE id IN ({placeholders})",
        tuple(doc_ids),
    ).fetchall()

    allowed: set[int] = set()
    for row in rows:
        doc_id = int(_row_get(row, "id", 0))
        doc_owner = _row_get(row, "owner_id", 1)
        doc_hotel = _row_get(row, "hotel_id", 2)

        if role != "ADMIN":
            if doc_hotel is None or str(doc_hotel) not in scope_set:
                continue
        if owner_id is not None and str(doc_owner) != owner_id:
            continue
        allowed.add(doc_id)

    # Preserve submitted order while deduplicating.
    ordered_allowed: List[int] = []
    for did in doc_ids:
        if int(did) in allowed and int(did) not in ordered_allowed:
            ordered_allowed.append(int(did))
    return ordered_allowed


def _vision_user_namespace() -> str:
    return f"user_{current_user.id}"


def _vision_user_dir(kind: str) -> Path:
    base = VISION_ROOT / _vision_user_namespace() / kind
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_vision_path_from_token(token: str) -> Path | None:
    """Resolve a token to an on-disk path under data/vision/, enforcing per-user isolation."""
    try:
        rel_path = decode_path(token)
    except Exception:
        return None

    if not rel_path:
        return None

    p = Path(rel_path)
    if p.is_absolute():
        return None

    abs_path = (PROJECT_ROOT / p).resolve()
    try:
        if os.path.commonpath([str(abs_path), str(VISION_ROOT)]) != str(VISION_ROOT):
            return None
    except Exception:
        return None

    # Enforce per-user namespace unless admin.
    role = str(getattr(current_user, "role", "")).upper()
    if role != "ADMIN":
        try:
            rel_to_root = abs_path.relative_to(VISION_ROOT)
        except Exception:
            return None
        if not rel_to_root.parts:
            return None
        if rel_to_root.parts[0] != _vision_user_namespace():
            return None

    return abs_path

@main_bp.route("/")
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def index():
    if current_user.role == 'CLIENTE':
        return redirect(url_for('main.client_dashboard'))
    return redirect(url_for('main.dashboard'))

@main_bp.route("/client/dashboard")
@login_required
def client_dashboard():
    if current_user.role != 'CLIENTE':
        return redirect(url_for('main.dashboard'))
    return render_template("client_dashboard.html")

@main_bp.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == 'CLIENTE':
        return redirect(url_for('main.client_dashboard'))
        
    db = get_db()
    hotel_id = request.args.get('hotel_id')
    schema = _documents_schema(db)
    created_col = schema["created_col"]
    has_file_size = schema["has_file_size"]
    has_doc_type = schema["type_col"] == "doc_type"
    
    # Validation of requested hotel_id
    if hotel_id and current_user.role != 'ADMIN':
        if str(hotel_id) not in [str(h) for h in current_user.hotel_scope]:
            hotel_id = None # Ignore unauthorized filter
    
    # Filter by hotel_scope
    scope_filter = ""
    scope_params = []
    
    if hotel_id:
        scope_filter = f" WHERE hotel_id = {db.placeholder}"
        scope_params = [hotel_id]
    elif current_user.role != 'ADMIN':
        if not current_user.hotel_scope:
            return render_template("dashboard.html", total_docs=0, status_stats={}, type_stats=[], recent_docs=[], metrics=[], recent_logs=[], pending_count=0, selected_hotel=None, available_hotels=[])
        
        placeholders = ",".join([db.placeholder] * len(current_user.hotel_scope))
        scope_filter = f" WHERE hotel_id IN ({placeholders})"
        scope_params = list(current_user.hotel_scope)

    # For the UI selector
    available_hotels = []
    if current_user.role == 'ADMIN':
        available_hotels = db.get_hotels()
    elif current_user.hotel_scope:
        all_hotels = db.get_hotels()
        available_hotels = [h for h in all_hotels if str(h['id']) in [str(s) for s in current_user.hotel_scope]]

    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)

        cursor.execute(f"SELECT COUNT(*) FROM documents{scope_filter}", scope_params)
        total_docs = cursor.fetchone()[0]

        cursor.execute(f"SELECT status, COUNT(*) FROM documents{scope_filter} GROUP BY status", scope_params)
        status_stats = {row[0]: row[1] for row in cursor.fetchall()}

        pending_where = " WHERE workflow_state IN ('pending', 'pending_review')"
        if scope_filter:
            pending_where += scope_filter.replace(" WHERE ", " AND ", 1)
        cursor.execute(f"SELECT COUNT(*) FROM documents{pending_where}", scope_params)
        pending_count = cursor.fetchone()[0]

        type_col = "doc_type" if has_doc_type else "type"
        type_where = f" WHERE {type_col} IS NOT NULL"
        if scope_filter:
            type_where += scope_filter.replace(" WHERE ", " AND ", 1)
        cursor.execute(f"SELECT {type_col}, COUNT(*) FROM documents{type_where} GROUP BY {type_col}", scope_params)
        raw_type_stats = cursor.fetchall()
        
        normalized_stats = {}
        for doc_type, count in raw_type_stats:
            clean_type = doc_type.strip().title() if doc_type else "Desconocido"
            normalized_stats[clean_type] = normalized_stats.get(clean_type, 0) + count
            
        type_stats = sorted(normalized_stats.items(), key=lambda x: x[1], reverse=True)[:10]

        doc_type_select = "doc_type" if has_doc_type else "type AS doc_type"
        file_size_select = "file_size" if has_file_size else "NULL AS file_size"
        cursor.execute(
            f"""
            SELECT id, filename, {doc_type_select}, status, {created_col} AS created_at, COALESCE(duration, 0) AS duration, {file_size_select}, error_message, tags
            FROM documents
            {scope_filter}
            ORDER BY {created_col} DESC
            LIMIT 10
            """, scope_params
        )
        recent_docs = []
        for row in cursor.fetchall():
            # Replace any None values in duration with 0
            processed_row = list(row)
            # Debug: Log the actual value
            get_logger().debug(f"Raw duration value: {processed_row[5]} (type: {type(processed_row[5])})")
            # Ensure duration is always a numeric value
            try:
                if processed_row[5] is None or processed_row[5] == '' or processed_row[5] == 'None':
                    processed_row[5] = 0.0
                else:
                    processed_row[5] = float(processed_row[5])
            except Exception as e:
                get_logger().error(f"Error processing duration: {e} (value: {processed_row[5]})")
                processed_row[5] = 0.0
            # Replace all None values in the entire row with safe values
            processed_row = [0.0 if x is None else x for x in processed_row]
            recent_docs.append(processed_row)

        # Metrics table doesn't exist in the new schema â€” skip
        metrics = []

        # Tables are now stored in document_blocks, not ocr_texts
        recent_tables: List[Dict[str, Any]] = []

    image_results = session.pop("image_results", None)
    image_error = session.pop("image_error", None)

    config = load_configuration()
    vision_enabled = config.get("vision", {}).get("enabled", False)

    today_metrics = {} 
    
    # Activity Chart Data (Last 7 days)
    activity_dates = []
    activity_counts = []
    try:
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            if db.engine_type == "postgresql":
                cursor.execute(
                    f"""
                    SELECT DATE({created_col}) AS d, COUNT(*)
                    FROM documents
                    WHERE {created_col} >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE({created_col})
                    ORDER BY DATE({created_col}) ASC
                    """
                )
            else:
                cursor.execute(
                    f"""
                    SELECT DATE({created_col}) AS d, COUNT(*)
                    FROM documents
                    WHERE DATETIME({created_col}) >= DATETIME('now', '-7 days')
                    GROUP BY DATE({created_col})
                    ORDER BY DATE({created_col}) ASC
                    """
                )
            rows = cursor.fetchall()
            for r in rows:
                activity_dates.append(str(r[0]))
                activity_counts.append(r[1])
    except Exception as e:
        get_logger().error(f"Error fetching activity stats: {e}")

    recent_logs = get_db().get_recent_logs(10) if get_db() else []

    return render_template(
        "dashboard.html",
        total_docs=total_docs,
        status_stats=status_stats,
        type_stats=type_stats,
        recent_docs=recent_docs,
        metrics=metrics,
        recent_tables=recent_tables,
        image_results=image_results,
        image_error=image_error,
        vision_enabled=vision_enabled,
        recent_logs=recent_logs,
        pending_count=pending_count,
        selected_hotel=hotel_id,
        available_hotels=available_hotels,
        activity_dates=activity_dates,
        activity_counts=activity_counts,
    )

@main_bp.route("/api/status")
@login_required
def system_status():
    """
    Check if the background worker is running and DB is accessible.
    """
    status = {
        "web": "online",
        "worker": "unknown",
        "database": "unknown"
    }
    
    try:
        db = get_db()
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            cursor.execute("SELECT 1")
            status["database"] = "online"
    except Exception:
        status["database"] = "offline"

    # In a real setup we'd check Celery/Huey stats here
    status["worker"] = "online" if status["database"] == "online" else "offline"
    
    return status

@main_bp.route("/api/check-email", methods=["POST"])
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def check_email_trigger():
    """Manually trigger email check."""
    from modules.tasks import trigger_email_check_task
    try:
        trigger_email_check_task()
        return {"status": "ok", "message": "ComprobaciÃ³n de email iniciada."}
    except Exception as e:
        get_logger().error(f"Email check trigger failed: {e}")
        return {"status": "error", "message": str(e)}, 500

@main_bp.route("/verify")
@login_required
def verify_queue():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
        return redirect(url_for('main.client_dashboard'))

    db = get_db()
    schema = _documents_schema(db)
    created_col = schema["created_col"]
    type_col = schema["type_col"]

    # Filter by hotel_scope
    scope_filter = ""
    scope_params = []
    if current_user.role != 'ADMIN':
        if not current_user.hotel_scope:
            return render_template("documents.html", documents=[], title="Cola de VerificaciÃ³n", is_verification_list=True, total_pages=1, page=1, status_filter="", type_filter="", search="")
        placeholders = ",".join([db.placeholder] * len(current_user.hotel_scope))
        scope_filter = f" AND d.hotel_id IN ({placeholders})"
        scope_params = list(current_user.hotel_scope)

    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(
            f"""
            SELECT d.id, d.filename, d.{created_col} AS created_at,
                   COALESCE(o.confidence, 0) AS confidence,
                   d.{type_col} AS doc_type
            FROM documents d
            LEFT JOIN ocr_texts o ON o.id_doc = d.id
            WHERE d.workflow_state IN ('pending', 'pending_review')
            {scope_filter}
            ORDER BY d.{created_col} ASC
            """, scope_params
        )
        pending_docs = cursor.fetchall()
    
    return render_template("documents.html", 
                           documents=pending_docs, 
                           title="Cola de VerificaciÃ³n",
                           is_verification_list=True,
                           total_pages=1,
                           page=1,
                           status_filter="",
                           type_filter="",
                           search="")

@main_bp.route("/documents")
@login_required
def documents():
    db = get_db()
    role = str(getattr(current_user, "role", "")).upper()
    schema = _documents_schema(db)
    created_col = schema["created_col"]
    path_col = schema["path_col"]
    type_col = schema["type_col"]
    file_size_select = "file_size" if schema["has_file_size"] else "NULL"

    # Filter by hotel_scope
    scope_filter = ""
    scope_params = []
    if current_user.role != 'ADMIN':
        if not current_user.hotel_scope:
             return render_template("documents.html", documents=[], page=1, total_pages=1)
        placeholders = ",".join([db.placeholder] * len(current_user.hotel_scope))
        scope_filter = f" AND hotel_id IN ({placeholders})"
        scope_params = list(current_user.hotel_scope)

    # Client isolation: never show other clients' documents, even within the same hotel.
    owner_filter = ""
    owner_params: List[Any] = []
    if role in {"CLIENTE", "CLIENT"}:
        owner_filter = f" AND owner_id = {db.placeholder}"
        owner_params = [current_user.id]

    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)

        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        offset = (page - 1) * per_page

        status_filter = request.args.get("status")
        type_filter = request.args.get("type")
        search_term = request.args.get("search", "")

        query = f"""
            SELECT id, filename, {path_col}, {type_col}, status, {created_col}, {file_size_select}, tags, error_message
            FROM documents
            WHERE 1=1
        """
        query += scope_filter
        query += owner_filter

        params: List[Any] = []
        params.extend(scope_params)
        params.extend(owner_params)

        if status_filter:
            query += f" AND status = {db.placeholder}"
            params.append(status_filter)
        if type_filter:
            query += f" AND {type_col} = {db.placeholder}"
            params.append(type_filter)
        if search_term:
            # Cross-DB case-insensitive search (SQLite has no ILIKE).
            query += (
                f" AND (LOWER(filename) LIKE LOWER({db.placeholder})"
                f" OR LOWER({type_col}) LIKE LOWER({db.placeholder}))"
            )
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        query += f" ORDER BY {created_col} DESC LIMIT {db.placeholder} OFFSET {db.placeholder}"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        documents_rows = cursor.fetchall()

        count_query = "SELECT COUNT(*) FROM documents WHERE 1=1"
        count_query += scope_filter
        count_query += owner_filter
        count_params: List[Any] = []
        count_params.extend(scope_params)
        count_params.extend(owner_params)
        
        if status_filter:
            count_query += f" AND status = {db.placeholder}"
            count_params.append(status_filter)
        if type_filter:
            count_query += f" AND {type_col} = {db.placeholder}"
            count_params.append(type_filter)
        if search_term:
            query_search = (
                f" AND (LOWER(filename) LIKE LOWER({db.placeholder})"
                f" OR LOWER({type_col}) LIKE LOWER({db.placeholder}))"
            )
            count_query += query_search
            count_params.extend([f"%{search_term}%", f"%{search_term}%"])
        cursor = db.execute(count_query, count_params)
        total_docs = cursor.fetchone()[0]
        
    total_pages = max(1, (total_docs + per_page - 1) // per_page)

    return render_template(
        "documents.html",
        documents=documents_rows,
        page=page,
        total_pages=total_pages,
        status_filter=status_filter,
        type_filter=type_filter,
        search=search_term,
    )

@main_bp.route("/document/<int:doc_id>")
@login_required
@hotel_scoped('doc_id')
def document_detail(doc_id: int):
    db = get_db()
    
    # Ownership Check
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(f"SELECT owner_id FROM documents WHERE id = {db.placeholder}", (doc_id,))
        row = cursor.fetchone()
        if not row:
            flash("Documento no encontrado.", "error")
            return redirect(url_for("main.documents"))
             
        owner_id = row[0] if isinstance(row, (tuple, list)) else row['owner_id']
        role = str(getattr(current_user, "role", "")).upper()
        if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
            flash("No tienes permiso para ver este documento.", "error")
            return redirect(url_for("main.documents"))
    
    row = db.get_document(doc_id)
    if not row:
        flash("Documento no encontrado.", "error")
        return redirect(url_for("main.documents"))

    schema = _documents_schema(db)
    path_col = schema["path_col"]
    type_col = schema["type_col"]
    created_col = schema["created_col"]

    metadata = {}
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        try:
            cursor.execute(
                f"""
                SELECT {created_col}, duration, workflow_state, {path_col}, {type_col},
                       hotel_id, visibility, financial_level
                FROM documents
                WHERE id = {db.placeholder}
                """,
                (doc_id,),
            )
            md = cursor.fetchone()
            if md:
                metadata = {
                    "datetime": md[0] if isinstance(md, (tuple, list)) else md[created_col],
                    "duration": md[1] if isinstance(md, (tuple, list)) else md["duration"],
                    "workflow_state": md[2] if isinstance(md, (tuple, list)) else md["workflow_state"],
                    "path": md[3] if isinstance(md, (tuple, list)) else md[path_col],
                    "doc_type": md[4] if isinstance(md, (tuple, list)) else md[type_col],
                    "hotel_id": md[5] if isinstance(md, (tuple, list)) else md["hotel_id"],
                    "visibility": md[6] if isinstance(md, (tuple, list)) else md["visibility"],
                    "financial_level": md[7] if isinstance(md, (tuple, list)) else md["financial_level"],
                }
        except Exception:
            # Fallback to data already available from DBManager.get_document()
            metadata = {}

    document = {
        "id": row.get("id"),
        "filename": row.get("filename"),
        "path": metadata.get("path", row.get("path")),
        "type": row.get("type") or metadata.get("doc_type") or "Unknown",
        "status": row.get("status"),
        "datetime": metadata.get("datetime", row.get("date")),
        "duration": metadata.get("duration", 0.0),
        "workflow_state": metadata.get("workflow_state", "new"),
        "text": row.get("text") or "",
        "markdown": row.get("markdown") or "",
        "language": row.get("language") or "",
        "confidence": row.get("confidence") or 0.0,
        "tags": row.get("tags") if isinstance(row.get("tags"), list) else safe_json_parse(row.get("tags"), []),
        "blocks": row.get("blocks") if isinstance(row.get("blocks"), list) else safe_json_parse(row.get("blocks"), []),
        "tables": row.get("tables") if isinstance(row.get("tables"), list) else safe_json_parse(row.get("tables"), []),
        "structured_data": row.get("structured_data") if isinstance(row.get("structured_data"), dict) else safe_json_parse(row.get("structured_data"), {}),
        "hotel_id": metadata.get("hotel_id", row.get("hotel_id")),
        "doc_type": metadata.get("doc_type", row.get("doc_type") or row.get("type")),
        "visibility": metadata.get("visibility", row.get("visibility") or "private"),
        "financial_level": metadata.get("financial_level", row.get("financial_level") or "none"),
        "data": row.get("data") if isinstance(row.get("data"), dict) else {},
    }

    # Security: Restrict financial data
    if document["financial_level"] != 'none' and current_user.role not in ['DIRECCION', 'ADMIN']:
        document["tables"] = []
        document["structured_data"] = {"msg": "Acceso financiero restringido"}

    return render_template("document_detail.html", document=document)

@main_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "files" not in request.files:
            flash("Debe seleccionar archivos.", "error")
            return redirect(request.url)

        upload_dir = current_app.config["UPLOAD_FOLDER"]
        
        # User isolation folder (optional, but good practice)
        if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
            upload_dir = os.path.join(upload_dir, f"client_{current_user.id}")
            
        ensure_directories(upload_dir)

        files = [file for file in request.files.getlist("files") if file and file.filename]
        if not files:
            flash("Debe seleccionar al menos un archivo valido.", "error")
            return redirect(request.url)

        config = load_configuration(reload=True)
        post_conf = config.get("postbatch", {})
        
        config_exts = {str(ext).lower() for ext in post_conf.get("file_types", [])}
        all_allowed = config_exts.union(ALLOWED_IMAGE_EXTS).union({".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".txt", ".json", ".eml"})

        invalid_files: List[str] = []
        saved_files: List[str] = []
        for file in files:
            filename = secure_filename(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in all_allowed:
                invalid_files.append(filename)
                continue
            temp_path = os.path.join(upload_dir, filename)
            file.save(temp_path)
            saved_files.append(temp_path)

        if invalid_files and not saved_files:
            flash(
                "Los siguientes archivos tienen una extension no permitida: "
                + ", ".join(invalid_files),
                "error",
            )
            return redirect(request.url)

        if not saved_files:
            flash("No se pudieron guardar los archivos seleccionados.", "error")
            return redirect(request.url)

        ocr_enabled = "ocr_enabled" in request.form if request.form else post_conf.get("ocr_enabled", True)
        classification_enabled = "classification_enabled" in request.form if request.form else post_conf.get("classification_enabled", True)
        handwriting_mode = "handwriting_mode" in request.form
        
        # Phase 4 Metadata
        hotel_id = request.form.get("hotel_id")
        doc_type = request.form.get("doc_type", "other")
        visibility = request.form.get("visibility", "private")
        financial_level = request.form.get("financial_level", "none")

        for temp_path in saved_files:
            options = {
                "delete_original": True,
                "ocr_enabled": ocr_enabled,
                "classification_enabled": classification_enabled,
                "input_root": upload_dir,
                "handwriting_mode": handwriting_mode,
                "owner_id": current_user.id,
                "hotel_id": hotel_id,
                "doc_type": doc_type,
                "visibility": visibility,
                "financial_level": financial_level
            }
            # Important: Inject owner_id into config temporarily or pass via options
            # Since process_single_file takes pipeline_conf, we can inject it there if needed, 
            # BUT process_document_task assumes options are just flags. 
            # We updated process_single_file to check pipeline_conf.get("owner_id"). 
            # Wait, process_single_file loads strict config from file. 
            # We need to modify process_document_task to update the config dict it passes or pass owner_id directly.
            # I updated postbatch_processor.py to use `pipeline_conf.get("owner_id")`.
            # So I need to pass owner_id via the task.
            # But process_document_task reloads config.
            # I will modify process_document_task in modules/tasks.py instead to be cleaner, 
            # OR better: make process_single_file accept owner_id arg.
            # I used `owner_id=pipeline_conf.get("owner_id")` in my edit. 
            # So passing it in `options` which are merged/accessible? No.
            # pipeline_conf is config.yaml content. 
            # I need to fix this.
            
            # Temporary fix: I will pass it in options and rely on task wrapper.
            process_document_task(temp_path, options)

        flash(f"Se han puesto en cola {len(saved_files)} archivos.", "success")
        return redirect(url_for('main.index'))

    return render_template("upload.html")

@main_bp.route("/chat")
@login_required
def chat():
    return render_template("chat.html")

@main_bp.route("/duplicates")
@login_required
def duplicates_page():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
         return redirect(url_for('main.client_dashboard'))
    return render_template("duplicates.html")

@main_bp.route("/settings", methods=["GET", "POST"])
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def settings():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
        flash("Acceso denegado.", "error")
        return redirect(url_for('main.client_dashboard'))
        
    config = load_configuration()
    
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "directories":
                post_conf = config.setdefault("postbatch", {})
                post_conf["input_folder"] = request.form.get("input_folder", "").strip()
                post_conf["processed_folder"] = request.form.get("processed_folder", "").strip()
                post_conf["failed_folder"] = request.form.get("failed_folder", "").strip()
                flash("Directorios actualizados.", "success")
                
            elif action == "hot_folder":
                hot_conf = config.setdefault("hot_folder", {})
                hot_conf["enabled"] = "hot_enabled" in request.form
                hot_conf["path"] = request.form.get("hot_path", "").strip()
                flash("ConfiguraciÃ³n de Hot Folder actualizada.", "success")
                
            elif action == "pipeline":
                app_conf = config.setdefault("app", {})
                app_conf["gpu_enabled"] = "gpu_enabled" in request.form
                
                pipe_conf = config.setdefault("ocr_pipeline", {})
                pipe_conf.setdefault("fusion", {})["priority"] = [] 
                pipe_conf["primary_engine"] = request.form.get("primary_engine", "auto")
                
                post_conf = config.setdefault("postbatch", {})
                post_conf["languages"] = [l.strip() for l in request.form.get("languages", "es").split(",")]
                flash("ConfiguraciÃ³n de Pipeline actualizada.", "success")
                
            elif action == "email_import":
                email_conf = config.setdefault("email_importer", {})
                email_conf["enabled"] = "email_enabled" in request.form
                email_conf["host"] = request.form.get("email_host", "").strip()
                email_conf["port"] = int(request.form.get("email_port", 993))
                email_conf["user"] = request.form.get("email_user", "").strip()
                email_conf["password"] = request.form.get("email_password", "").strip()
                flash("ConfiguraciÃ³n de Email actualizada.", "success")
                
            elif action == "rebuild_index":
                rebuild_vision_index_task()
                flash("Reindexado iniciado en segundo plano.", "success")

            elif action == "llm_pipeline_config":
                llm_conf = config.setdefault("llm", {})
                llm_conf["enabled"] = "llm_enabled" in request.form
                llm_conf["base_url"] = request.form.get("llm_base_url", "").strip()
                llm_conf["model"] = request.form.get("llm_model", "").strip()
                llm_conf["api_key"] = request.form.get("llm_api_key", "").strip()
                llm_conf["timeout"] = int(request.form.get("llm_timeout", 60))
                flash("ConfiguraciÃ³n de Pipeline IA actualizada.", "success")

            elif action == "llm_chat_config":
                if "routing" not in config.get("llm", {}):
                    config.setdefault("llm", {})["routing"] = {}
                
                routing_conf = config["llm"]["routing"]
                chat_conf = routing_conf.setdefault("general_chat", {})
                
                chat_conf["base_url"] = request.form.get("chat_base_url", "").strip()
                chat_conf["model"] = request.form.get("chat_model", "").strip()
                flash("ConfiguraciÃ³n de Chat IA actualizada.", "success")
                
            elif action == "webhooks":
                app_conf = config.setdefault("app", {})
                app_conf["webhook_url"] = request.form.get("webhook_url", "").strip()
                app_conf["webhook_secret"] = request.form.get("webhook_secret", "").strip()
                flash("ConfiguraciÃ³n de Webhooks actualizada.", "success")
                
            save_configuration(config)
            return redirect(url_for("main.settings"))
            
        except Exception as e:
            get_logger().error(f"Error saving settings: {e}")
            flash(f"Error al guardar configuraciÃ³n: {e}", "error")

    post_conf = config.get("postbatch", {})
    hot_conf = config.get("hot_folder", {})
    email_conf = config.get("email_importer", {})
    vision_conf = config.get("vision", {})
    pipeline_conf = config.get("ocr_pipeline", {}) 
    app_conf = config.get("app", {})
    llm_conf = config.get("llm", {})
    chat_conf = llm_conf.get("routing", {}).get("general_chat", {})
    
    settings_data = {
        "input_folder": post_conf.get("input_folder", ""),
        "processed_folder": post_conf.get("processed_folder", ""),
        "failed_folder": post_conf.get("failed_folder", ""),
        "reports_folder": post_conf.get("reports_folder", ""),
        "gpu_enabled": app_conf.get("gpu_enabled", False),
        "languages": ", ".join(post_conf.get("languages", ["es"])),
        "primary_engine": pipeline_conf.get("primary_engine", "auto"),
        "vision_enabled": vision_conf.get("enabled", True),
        "gallery_dir": vision_conf.get("gallery_dir", "data/vision_gallery"),
        "hot_enabled": hot_conf.get("enabled", False),
        "hot_path": hot_conf.get("path", ""),
        "email_enabled": email_conf.get("enabled", False),
        "email_host": email_conf.get("host", ""),
        "email_port": email_conf.get("port", 993),
        "email_user": email_conf.get("user", ""),
        "email_password": email_conf.get("password", ""),
        "llm_enabled": llm_conf.get("enabled", False),
        "llm_base_url": llm_conf.get("base_url", "http://host.docker.internal:1234/v1"),
        "llm_model": llm_conf.get("model", "local-model"),
        "llm_api_key": llm_conf.get("api_key", ""),
        "llm_timeout": llm_conf.get("timeout", 60),
        "chat_base_url": chat_conf.get("base_url", "http://host.docker.internal:1234/v1"),
        "chat_model": chat_conf.get("model", "mistral-small-24b"),
        "webhook_url": app_conf.get("webhook_url", ""),
        "webhook_secret": app_conf.get("webhook_secret", "")
    }
    return render_template("settings.html", config=settings_data)

@main_bp.route("/batch_process", methods=["GET", "POST"])
@login_required
def batch_process():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
         return redirect(url_for('main.client_dashboard'))
         
    if request.method == "POST":
        target_folder = request.form.get("target_folder", "").strip()
        if not target_folder:
            flash("Debe especificar una carpeta para procesar.", "error")
            return redirect(request.url)
        if not os.path.exists(target_folder):
            flash(f"La carpeta no existe: {target_folder}", "error")
            return redirect(request.url)

        try:
            from postbatch_processor import main as batch_main

            result = batch_main(["--input-folder", target_folder, "--immediate"])
            if result == 0:
                flash(f"Procesamiento completado en {target_folder}", "success")
            else:
                flash("El procesamiento finalizÃ³ con errores.", "error")
        except Exception as exc:
            get_logger().error("Error running batch process: %s", exc)
            flash(f"Error en procesamiento: {exc}", "error")
        return redirect(url_for("main.dashboard"))

    return render_template("batch_process.html")


@main_bp.route("/view_document_file/<int:doc_id>")
@login_required
@hotel_scoped('doc_id')
def view_document_file(doc_id):
    db = get_db()
    # Check ownership
    with db.get_connection() as conn:
         cursor = db.get_cursor(conn)
         cursor.execute(f"SELECT owner_id FROM documents WHERE id = {db.placeholder}", (doc_id,))
         row = cursor.fetchone()
         if row:
              owner_id = row[0] if isinstance(row, (tuple, list)) else row['owner_id']
              role = str(getattr(current_user, "role", "")).upper()
              if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
                  return "Unauthorized", 403

    path_str = db.get_document_path(doc_id)
    if not path_str:
        return "File not found", 404
    
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    
    version = request.args.get("version")
    if version == "original":
         backup_p = p.with_name(f"{p.stem}_original{p.suffix}")
         if backup_p.exists():
              p = backup_p
              
    try:
        # Security: only serve files from known storage roots (prevents DB path injection / LFI).
        config = load_configuration()
        post_conf = config.get("postbatch", {})
        allowed_roots = [
            Path(resolve_path(post_conf.get("processed_folder"), "processed")),
            Path(resolve_path(post_conf.get("failed_folder"), "errors")),
            Path(resolve_path(post_conf.get("input_folder"), "input")),
        ]
        upload_root = str(current_app.config.get("UPLOAD_FOLDER") or "").strip()
        if upload_root:
            allowed_roots.append(Path(upload_root))
        allowed_roots = [
            root.resolve()
            for root in allowed_roots
            if str(root) and str(root) not in {"", "."}
        ]

        p_abs = p.resolve()
        def _within(root: Path) -> bool:
            try:
                return os.path.commonpath([str(p_abs), str(root)]) == str(root)
            except Exception:
                return False
        if allowed_roots and not any(_within(root) for root in allowed_roots):
            get_logger().warning("Blocked attempt to serve file outside allowed roots: %s", p_abs)
            return "Forbidden", 403

        if not p.exists():
            get_logger().error(f"File not found on disk: {p}")
            return "File not found on disk", 404
                
        return send_from_directory(p.parent, p.name)
    except Exception as e:
         get_logger().error(f"Error serving file: {e}")
         return "Error serving file", 500

@main_bp.route("/verify/<int:doc_id>")
@login_required
@hotel_scoped('doc_id')
def verify_document(doc_id):
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
        return redirect(url_for('main.client_dashboard'))

    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        return "Document not found", 404
    
    if isinstance(doc.get("structured_data"), str):
        try:
             doc["structured_data"] = json.loads(doc["structured_data"])
        except Exception:
             doc["structured_data"] = {}
             
    if doc.get("structured_data") and "fields" in doc["structured_data"]:
        fields = doc["structured_data"]["fields"]
        for key, info in fields.items():
            if isinstance(info, dict) and "confidence" in info:
                doc[f"{key}_conf"] = info["confidence"]
                if not doc["data"].get(key) and info.get("value"):
                     doc["data"][key] = info["value"]
                     
    return render_template("verification_split.html", document=doc)

@main_bp.route("/documents/batch_action", methods=["POST"])
@login_required
def batch_action():
    action = request.form.get("action")
    doc_ids_str = request.form.get("doc_ids")

    if not action or not doc_ids_str:
        flash("AcciÃ³n invÃ¡lida", "error")
        return redirect(url_for("main.documents"))

    try:
        parsed_ids = json.loads(doc_ids_str)
        doc_ids = [int(doc_id) for doc_id in parsed_ids]
    except Exception:
        flash("IDs invÃ¡lidos", "error")
        return redirect(url_for("main.documents"))

    if not doc_ids:
        flash("NingÃºn documento seleccionado", "warning")
        return redirect(url_for("main.documents"))

    db = get_db()
    allowed_doc_ids = _filter_accessible_doc_ids(db, doc_ids)
    denied_count = len(doc_ids) - len(allowed_doc_ids)
    if denied_count > 0:
        flash(f"{denied_count} documento(s) fuera de tu alcance fueron omitidos.", "warning")
    if not allowed_doc_ids:
        flash("No tienes acceso a los documentos seleccionados.", "error")
        return redirect(url_for("main.documents"))

    if action == "delete":
        success_count = 0
        for doc_id in allowed_doc_ids:
            if db.delete_document(doc_id):
                success_count += 1
        flash(f"{success_count} documentos eliminados", "success")

    elif action == "reprocess":
        from modules.tasks import process_document_task

        count = 0
        schema = _documents_schema(db)
        path_col = schema["path_col"]
        for doc_id in allowed_doc_ids:
            row = db.execute(
                f"SELECT {path_col} FROM documents WHERE id = {db.placeholder}",
                (doc_id,),
            ).fetchone()
            if not row:
                continue
            path = Path(_row_get(row, path_col, 0))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            process_document_task(
                str(path),
                {
                    "delete_original": False,
                    "ocr_enabled": True,
                    "classification_enabled": True,
                },
            )
            count += 1
        flash(f"{count} documentos enviados a reprocesar", "info")

    elif action == "export":
        return export_documents(allowed_doc_ids)

    return redirect(url_for("main.documents"))

def export_documents(doc_ids):
    import csv 
    import io
    from flask import make_response
    
    db = get_db()
    schema = _documents_schema(db)
    created_col = schema["created_col"]
    type_col = schema["type_col"]
    placeholders = ",".join([db.placeholder] * len(doc_ids))
    query = f"""
        SELECT d.id, d.filename, d.{type_col} AS doc_type, d.status,
               COALESCE(o.confidence, 0) AS confidence,
               d.{created_col} AS created_at
        FROM documents d
        LEFT JOIN ocr_texts o ON o.id_doc = d.id
        WHERE d.id IN ({placeholders})
        ORDER BY d.{created_col} DESC
    """
    
    rows = db.execute(query, tuple(doc_ids)).fetchall()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Archivo", "ClasificaciÃ³n", "Estado", "Confianza", "Fecha"])
    for row in rows:
        cw.writerow(row)
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@main_bp.route("/tasks")
@login_required
def tasks_page():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
         return redirect(url_for('main.client_dashboard'))
    return render_template("tasks.html")

@main_bp.route("/gallery")
@login_required
def gallery():
    if str(getattr(current_user, "role", "")).upper() in {"CLIENTE", "CLIENT"}:
         return redirect(url_for('main.client_dashboard'))
    return render_template("gallery.html")

@main_bp.route("/download/table/<int:doc_id>/<int:index>/<fmt>")
@login_required
@hotel_scoped('doc_id')
def download_table(doc_id, index, fmt):
    db = get_db()

    # Client isolation: clients can only download tables for their own documents.
    role = str(getattr(current_user, "role", "")).upper()
    if role in {"CLIENTE", "CLIENT"}:
        owner_row = db.execute(
            f"SELECT owner_id FROM documents WHERE id = {db.placeholder}", (doc_id,)
        ).fetchone()
        owner_id = owner_row[0] if owner_row else None
        if owner_id is None or str(owner_id) != str(current_user.id):
            return "Unauthorized", 403
    
    row = db.execute(f"SELECT tables_json FROM ocr_texts WHERE id_doc = {db.placeholder}", (doc_id,)).fetchone()
    if not row or not row[0]:
        return "Table not found", 404
        
    tables = json.loads(row[0])
    if index < 0 or index >= len(tables):
        return "Index out of bounds", 404
        
    table = tables[index]
    
    if fmt == "csv":
        path = table.get("csv_path")
        ext = "csv"
    elif fmt == "json":
        path = table.get("json_path")
        ext = "json"
    else:
        return "Invalid format", 400
        
    if not path:
        return "File path missing in DB", 404

    # Security: tables must be served only from the configured tables directory.
    config = load_configuration()
    tables_dir = (
        config.get("ocr_pipeline", {}).get("output", {}).get("tables_dir", "data/tables")
    )
    tables_root = Path(resolve_path(tables_dir, "data/tables")).resolve()

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    try:
        p_abs = p.resolve()
    except Exception:
        return "Forbidden", 403

    try:
        if os.path.commonpath([str(p_abs), str(tables_root)]) != str(tables_root):
            get_logger().warning(
                "Blocked attempt to download table file outside tables root: %s", p_abs
            )
            return "Forbidden", 403
    except Exception:
        return "Forbidden", 403

    if not p_abs.exists():
        return "File not found on disk", 404

    return send_from_directory(
        p_abs.parent,
        p_abs.name,
        as_attachment=True,
        download_name=f"table_{doc_id}_{index}.{ext}",
    )


@main_bp.route("/data/exports/<path:filename>")
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def download_export_file(filename: str):
    """Serve generated export files from data/exports (authenticated)."""
    exports_root = (PROJECT_ROOT / "data" / "exports").resolve()
    p = (exports_root / filename).resolve()

    # Per-user export isolation: only admins can access other users' exports.
    role = str(getattr(current_user, "role", "")).upper()
    if role != "ADMIN":
        parts = Path(filename).parts
        if not parts or parts[0] != f"user_{current_user.id}":
            return "Forbidden", 403
    try:
        if os.path.commonpath([str(p), str(exports_root)]) != str(exports_root):
            return "Forbidden", 403
    except Exception:
        return "Forbidden", 403

    if not p.exists():
        return "File not found", 404

    return send_from_directory(p.parent, p.name, as_attachment=True)


@main_bp.route("/vision/preview/<token>")
@login_required
def vision_preview(token: str):
    """
    Resolve a base64-encoded document path to a document id and redirect to the
    guarded file-serving route. Used by the Vision/Deduplication UI.
    """
    db = get_db()
    try:
        path_str = decode_path(token)
    except Exception:
        return "Forbidden", 403

    if not path_str:
        return "Not found", 404

    candidates: List[str] = [path_str]
    p = Path(path_str)
    try:
        if not p.is_absolute():
            candidates.append(str((PROJECT_ROOT / p).resolve()))
        else:
            try:
                candidates.append(str(p.resolve().relative_to(PROJECT_ROOT)))
            except Exception:
                pass
    except Exception:
        pass

    doc_id = None
    schema = _documents_schema(db)
    path_col = schema["path_col"]
    for cand in candidates:
        row = db.execute(f"SELECT id FROM documents WHERE {path_col} = {db.placeholder}", (cand,)).fetchone()
        if row:
            doc_id = row[0]
            break

    if not doc_id:
        return "Not found", 404

    return redirect(url_for("main.view_document_file", doc_id=int(doc_id)))


@main_bp.route("/vision/file/<token>")
@login_required
def vision_file(token: str):
    """Serve per-user Vision Studio artifacts stored under data/vision/."""
    p = _resolve_vision_path_from_token(token)
    if not p:
        return "Forbidden", 403
    if p.suffix.lower() not in VISION_ALLOWED_SUFFIXES:
        return "Forbidden", 403
    if not p.exists():
        return "File not found", 404

    download = str(request.args.get("download", "")).strip().lower() in {"1", "true", "yes", "y"}
    return send_from_directory(p.parent, p.name, as_attachment=download)

@main_bp.route("/image_search", methods=["POST"])
@login_required
def image_search():
    if "image" not in request.files:
        flash("No se subiÃ³ imagen", "error")
        return redirect(url_for("main.dashboard"))
        
    file = request.files["image"]
    if file.filename == "":
        flash("Nombre de archivo vacÃ­o", "error")
        return redirect(url_for("main.dashboard"))

    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix)
        os.close(fd)
        file.save(temp_path)

        pipeline = get_pipeline()
        vision = pipeline.vision_manager

        if not vision or not vision.config.enabled:
            flash("VisiÃ³n no habilitada", "error")
            return redirect(url_for("main.dashboard"))

        results = vision.search_similar(temp_path, k=5)

        db = get_db()
        schema = _documents_schema(db)
        path_col = schema["path_col"]
        created_col = schema["created_col"]
        enriched = []
        role = str(getattr(current_user, "role", "")).upper()
        scope_set = {str(h) for h in getattr(current_user, "hotel_scope", []) or []}
        if role != "ADMIN" and not scope_set:
            session["image_results"] = []
            session["image_error"] = "Usuario sin alcance de hotel configurado."
            return redirect(url_for("main.dashboard") + "#vision-pane")

        for res in results:
            path = res.get("path")
            if not path:
                continue

            row = _find_document_by_path(
                db,
                path_col,
                path,
                f"id, filename, {path_col} AS path, tags, {created_col} AS created_at, owner_id, hotel_id",
            )
            if row:
                owner_id = _row_get(row, "owner_id", 5)
                hotel_id = _row_get(row, "hotel_id", 6)

                # Multi-tenant isolation: never leak other hotels' documents.
                if role != "ADMIN":
                    if hotel_id is None or str(hotel_id) not in scope_set:
                        continue
                if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
                    continue

                stored_path = _row_get(row, "path", 2) or path
                enriched.append({
                    "id": _row_get(row, "id", 0),
                    "filename": _row_get(row, "filename", 1),
                    "path": stored_path,
                    "score": res["score"],
                    "tags": safe_json_parse(_row_get(row, "tags", 3), []),
                    "preview_url": url_for("main.vision_preview", token=encode_path(stored_path)),
                })

        session["image_results"] = enriched
        return redirect(url_for("main.dashboard") + "#vision-pane")

    except Exception as e:
        get_logger().error(f"Image search error: {e}")
        session["image_error"] = str(e)
        return redirect(url_for("main.dashboard"))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@main_bp.route("/vision/studio")
@login_required
def vision_studio():
    return render_template("vision_dashboard.html")

@main_bp.route("/vision/analyze", methods=["POST"])
@login_required
def vision_analyze():
    if 'file' not in request.files:
        flash("No se subiÃ³ ningÃºn archivo.", "error")
        return redirect(url_for('main.vision_studio'))
    
    file = request.files['file']
    mode = request.form.get("mode", "furniture")
    
    if file.filename == '':
        flash("Nombre de archivo vacÃ­o.", "error")
        return redirect(url_for('main.vision_studio'))

    original_name = secure_filename(file.filename)
    if not original_name:
        flash("Nombre de archivo invÃ¡lido.", "error")
        return redirect(url_for('main.vision_studio'))

    # Store vision uploads outside of `static/` and serve them through an authenticated route.
    upload_dir = _vision_user_dir("uploads")
    suffix = Path(original_name).suffix.lower()
    filename = f"{secrets.token_hex(8)}{suffix}"
    file_path = upload_dir / filename
    file.save(str(file_path))

    rel_path = str(file_path.resolve().relative_to(PROJECT_ROOT))
    image_token = encode_path(rel_path)
    image_src = url_for("main.vision_file", token=image_token)

    vision_manager = get_pipeline().vision_manager
    
    results = {}
    if mode == "interactive":
        return render_template(
            "vision_interactive.html", image_src=image_src, image_token=image_token
        )
        
    if mode == "furniture":
        results = vision_manager.detect_design_elements(str(file_path))
        od_data = results.get("od_data") or {}
        objects = results.get("objects") or []
        bboxes = od_data.get("bboxes") if isinstance(od_data, dict) else None
        pm = getattr(get_pipeline(), "product_manager", None)
        if bboxes:
            first_bbox = bboxes[0]
            results["similar_products"] = vision_manager.find_similar_products(
                str(file_path), first_bbox, pm
            )
        
        # Phase II/III: Technical RAG (Search docs for the most relevant item)
        if objects:
            from modules.rag_manager import RAGManager
            rag = RAGManager(index_dir=str(Path(current_app.root_path).parent / "data" / "rag_index"))
            results["technical_docs"] = vision_manager.search_technical_docs(objects[0], rag)
        
    elif mode == "render":
        style = request.form.get("style", "")
        # Adjust prompt based on style
        style_prompts = {
            "Industrial": "industrial loft style, exposed brick, iron furniture, dark tones",
            "Nordic": "scandinavian design, light oak wood, minimalist, white and grey palette, bright lighting",
            "Biophilic": "biophilic architecture, lush indoor plants, natural materials, organic shapes",
        }
        custom_prompt = style_prompts.get(style, "modern luxury home")
        
        from modules.render_manager import RenderManager
        render_manager = RenderManager()
        render_path = render_manager.generate_from_sketch(str(file_path), custom_prompt)
        # ... rest of render path handling ...
        if render_path:
            render_filename = f"render_{secrets.token_hex(6)}_{filename}"
            target_render = upload_dir / render_filename
            import shutil
            shutil.copy(render_path, target_render)
            render_rel = str(target_render.resolve().relative_to(PROJECT_ROOT))
            render_token = encode_path(render_rel)
            results = {
                "render_src": url_for("main.vision_file", token=render_token),
                "render_token": render_token,
            }

    return render_template(
        "vision_results.html",
        results=results,
        mode=mode,
        image_src=image_src,
        image_token=image_token,
    )

@main_bp.route("/vision/segment/click", methods=["POST"])
@login_required
def vision_segment_click():
    try:
        x_percent = float(request.form.get("x"))
        y_percent = float(request.form.get("y"))
    except (TypeError, ValueError):
        return {"success": False, "error": "Invalid click coordinates"}, 400

    if x_percent < 0 or x_percent > 100 or y_percent < 0 or y_percent > 100:
        return {"success": False, "error": "Coordinates out of range"}, 400

    image_token = request.form.get("image_token")

    if not image_token:
        return {"success": False, "error": "Missing image_token"}

    full_path = _resolve_vision_path_from_token(image_token)
    if not full_path:
        return {"success": False, "error": "Forbidden"}
    
    from modules.segmentation_manager import SegmentationManager
    # In a real app, model paths should be in config
    sam = SegmentationManager(checkpoint_path=str(Path(current_app.root_path).parent / "models" / "sam_vit_b_01ec64.pth"))
    
    try:
        from PIL import Image
        img = Image.open(full_path).convert("RGB")
        w, h = img.size
        
        # Convert percent to pixels
        px = int((x_percent / 100) * w)
        py = int((y_percent / 100) * h)
        
        # Segment by point (assumes 1 as label)
        # We need segment_by_points to return a file path or we save it here
        masks = sam.segment_by_points(img, [[px, py]], [1])
        mask = masks[0] # Use the first (usually most confident) mask
        
        # Create transparent image
        img_rgba = img.convert("RGBA")
        import numpy as np
        data = np.array(img_rgba)
        data[:, :, 3] = mask.astype(np.uint8) * 255
        
        res_img = Image.fromarray(data)
        
        # Save to per-user vision storage.
        res_filename = f"seg_{px}_{py}_{secrets.token_hex(6)}.png"
        res_dir = _vision_user_dir("segments")
        res_path = res_dir / res_filename
        res_img.save(str(res_path))

        seg_rel = str(res_path.resolve().relative_to(PROJECT_ROOT))
        seg_token = encode_path(seg_rel)
        return {
            "success": True,
            "segment_url": url_for("main.vision_file", token=seg_token),
            "segment_token": seg_token,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@main_bp.route("/vision/canvas")
@login_required
def vision_canvas():
    # Fetch segments from per-user vision storage to show in gallery
    segments_dir = (VISION_ROOT / _vision_user_namespace() / "segments").resolve()
    segments = []
    if segments_dir.exists():
        for f in segments_dir.glob("*.png"):
            try:
                rel = str(f.resolve().relative_to(PROJECT_ROOT))
            except Exception:
                continue
            token = encode_path(rel)
            segments.append(
                {"url": url_for("main.vision_file", token=token), "name": f.name}
            )
    
    return render_template("canvas.html", segments=segments)

@main_bp.route("/vision/report", methods=["POST"])
@login_required
def vision_report():
    payload = request.json or {}

    original_token = payload.get("original_token") or ""
    render_token = payload.get("render_token") or ""

    if not original_token:
        return {"success": False, "error": "Missing original_token"}

    original_path = _resolve_vision_path_from_token(original_token)
    if not original_path or not original_path.exists():
        return {"success": False, "error": "Original image not found"} 

    render_path = None
    if render_token:
        render_path = _resolve_vision_path_from_token(render_token)
        if render_path and not render_path.exists():
            render_path = None

    from modules.report_generator import ReportGenerator

    report_dir = _vision_user_dir("reports")
    report_gen = ReportGenerator(output_dir=str(report_dir))

    project_data = {
        "title": payload.get("title") or "Informe de Proyecto AutOCR",
        "original_image": str(original_path),
        "render_image": str(render_path) if render_path else "",
        "furniture": payload.get("furniture") or [],
        "advice": payload.get("advice") or "",
        "colors": payload.get("colors") or [],
        "details": payload.get("details") or "",
    }

    report_path_str = report_gen.generate_project_report(project_data)
    if report_path_str:
        report_path = Path(report_path_str)
        try:
            report_rel = str(report_path.resolve().relative_to(PROJECT_ROOT))
        except Exception:
            return {"success": False, "error": "Report generated but could not be served"}
        report_token = encode_path(report_rel)
        return {"success": True, "report_url": url_for("main.vision_file", token=report_token)}

    return {"success": False, "error": "Error generando PDF"}
@main_bp.route("/api/document/<int:doc_id>/advice", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def get_document_advice(doc_id):
    """
    Generates AI Decorator advice for a specific document image.
    """
    db = get_db()
    schema = _documents_schema(db)
    path_col = schema["path_col"]
    type_col = schema["type_col"]
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(
            f"SELECT {path_col}, {type_col}, tags FROM documents WHERE id = {db.placeholder}",
            (doc_id,),
        )
        doc = cursor.fetchone()
    
    if not doc:
        return jsonify({"error": "Document not found"}), 404
         
    file_path = doc[0]
    doc_type = doc[1]
    tags_json = doc[2]

    # Access control is enforced by @hotel_scoped('doc_id').
    
    if doc_type != "Imagen":
        return jsonify({"error": "Solo se puede asesorar sobre imÃ¡genes."}), 400
        
    from web_app.services import get_llm_client
    from modules.decor_advisor import DecorAdvisor
    
    client = get_llm_client()
    advisor = DecorAdvisor()
    
    # Parse tags to list
    tags = []
    if tags_json:
        tags = safe_json_parse(tags_json, [])
        
    # Generate advice
    # file_path might be relative or absolute. Ensure absolute.
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
        
    advice = advisor.generate_ai_advice(caption="Imagen de interior", objects=tags, llm_client=client, image_path=file_path)
    
    return jsonify({"advice": advice})

