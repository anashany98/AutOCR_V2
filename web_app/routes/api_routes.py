import os
import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from flask import Blueprint, jsonify, request, url_for, send_from_directory, send_file, current_app
from PIL import Image

from pydantic import ValidationError

from web_app.services import get_db, get_pipeline, get_vision_manager, get_logger, PROJECT_ROOT
from web_app.security.security_decorators import require_role, hotel_scoped, owner_scoped, financial_access_required
from flask_login import login_required, current_user
from web_app.utils import safe_json_parse, ensure_within_project, encode_path

from modules.schemas import DocumentUpdateSchema
from modules.moodboard import MoodboardGenerator
from modules.deduplicator import Deduplicator
from modules.learning import ModelTrainer
from modules.image_utils import enhance_image
from modules.tasks import huey

api_bp = Blueprint('api', __name__)


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif", ".gif", ".tif", ".tiff"}


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
    path_obj = Path(str(path_value))
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    try:
        return path_obj.resolve(strict=False)
    except Exception:
        return path_obj


def _path_candidates(path_value: Optional[str]) -> list[str]:
    candidates: list[str] = []

    def _add_variants(value: Optional[str]) -> None:
        if not value:
            return
        variants = {value, value.replace("\\", "/"), value.replace("/", "\\")}
        for v in variants:
            if v and v not in candidates:
                candidates.append(v)

    _add_variants(str(path_value) if path_value else None)
    abs_path = _abs_doc_path(path_value)
    if abs_path:
        abs_str = str(abs_path)
        _add_variants(abs_str)
        try:
            rel_str = str(abs_path.relative_to(PROJECT_ROOT.resolve()))
            _add_variants(rel_str)
        except Exception:
            pass
    return candidates


def _find_document_by_path(db, path_col: str, path_value: Optional[str], select_sql: str):
    candidates = _path_candidates(path_value)
    if not candidates:
        return None

    placeholders = ",".join([db.placeholder] * len(candidates))
    query = f"SELECT {select_sql} FROM documents WHERE {path_col} IN ({placeholders}) LIMIT 1"
    return db.execute(query, tuple(candidates)).fetchone()


@api_bp.route("/api/document/<int:doc_id>/update", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_update_document(doc_id: int):
    """Update document data with Pydantic validation."""
    try:
        data = request.json
        validated = DocumentUpdateSchema(**data)
        
        db = get_db()
        doc_schema = _documents_schema(db)
        type_col = doc_schema["type_col"]
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            
            # --- Update 'documents' table ---
            doc_updates = []
            doc_params = []
            
            if validated.filename:
                doc_updates.append(f"filename = {db.placeholder}")
                doc_params.append(validated.filename)
            if validated.type:
                doc_updates.append(f"{type_col} = {db.placeholder}")
                doc_params.append(validated.type)
            if validated.status:
                doc_updates.append(f"status = {db.placeholder}")
                doc_params.append(validated.status)
            if validated.tags is not None:
                doc_updates.append(f"tags = {db.placeholder}")
                doc_params.append(json.dumps(validated.tags))
                
            if doc_updates:
                doc_params.append(doc_id)
                cursor.execute(
                    f"UPDATE documents SET {', '.join(doc_updates)} WHERE id = {db.placeholder}",
                    tuple(doc_params)
                )

            # --- Update 'ocr_texts' table ---
            ocr_updates = []
            ocr_params = []
            
            if validated.text is not None:
                ocr_updates.append(f"text = {db.placeholder}")
                ocr_params.append(validated.text)
            if validated.markdown is not None:
                ocr_updates.append(f"markdown_text = {db.placeholder}")
                ocr_params.append(validated.markdown)
            
            if validated.date or validated.total is not None or validated.supplier:
                cursor.execute(f"SELECT structured_data FROM ocr_texts WHERE id_doc = {db.placeholder}", (doc_id,))
                row = cursor.fetchone()
                current_data = safe_json_parse(row[0], {}) if row else {}
                
                if validated.date:
                    current_data["date"] = validated.date
                if validated.total is not None:
                    current_data["total"] = validated.total
                if validated.supplier:
                    current_data["supplier"] = validated.supplier
                if validated.corrections is not None:
                    current_data["corrections"] = validated.corrections
                
                ocr_updates.append(f"structured_data = {db.placeholder}")
                ocr_params.append(json.dumps(current_data))
            
            if ocr_updates:
                ocr_params.append(doc_id)
                cursor.execute(
                    f"UPDATE ocr_texts SET {', '.join(ocr_updates)} WHERE id_doc = {db.placeholder}",
                    tuple(ocr_params)
                )
            
            conn.commit()
            
        return jsonify({"status": "success", "message": "Document updated successfully"})

    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 400
    except Exception as e:
        get_logger().error(f"Update error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/create_moodboard", methods=["POST"])
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def api_create_moodboard():
    data = request.json
    doc_ids = data.get("ids", [])
    title = data.get("title", "Mi Moodboard")

    if not doc_ids:
        return jsonify({"error": "No documents selected"}), 400

    db = get_db()
    schema = _documents_schema(db)
    path_col = schema["path_col"]
    
    # Enforce access controls for every selected document (multi-tenant isolation).
    try:
        doc_ids = [int(x) for x in doc_ids]
    except Exception:
        return jsonify({"error": "Invalid document IDs"}), 400

    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        ph = db.placeholder
        placeholders = ",".join(ph for _ in doc_ids)
        query = f"SELECT id, {path_col} AS path, owner_id, hotel_id FROM documents WHERE id IN ({placeholders})"
        cursor.execute(query, tuple(doc_ids))
        rows = cursor.fetchall()

    # Filter to documents accessible by the current user.
    role = str(getattr(current_user, "role", "")).upper()
    scope_set = {str(h) for h in (getattr(current_user, "hotel_scope", []) or [])}
    allowed_paths = []
    for row in rows:
        path_val = _row_get(row, "path", 1)
        owner_val = _row_get(row, "owner_id", 2)
        hotel_val = _row_get(row, "hotel_id", 3)

        if role != "ADMIN":
            if hotel_val is None or str(hotel_val) not in scope_set:
                continue
        if role in {"CLIENTE", "CLIENT"} and str(owner_val) != str(current_user.id):
            continue

        # Normalise path to absolute for the generator.
        abs_path = _abs_doc_path(path_val)
        if not abs_path or not abs_path.exists():
            continue
        allowed_paths.append(str(abs_path))

    if len(allowed_paths) != len(doc_ids):
        return jsonify({"error": "Access denied for one or more documents"}), 403
    
    if not allowed_paths:
        return jsonify({"error": "No valid images found for selected IDs"}), 404

    generator = MoodboardGenerator(output_dir=PROJECT_ROOT / "data" / "moodboards")
    try:
        output_path = generator.create(allowed_paths, title)
        if not output_path:
             return jsonify({"error": "Failed to generate moodboard (invalid images?)"}), 500
             
        filename = Path(output_path).name
        url = url_for("api.serve_moodboard", filename=filename) 
        return jsonify({"url": url})
        
    except Exception as e:
        get_logger().error(f"Moodboard generation error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/moodboard_file/<filename>")
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def serve_moodboard(filename):
    return send_from_directory(PROJECT_ROOT / "data" / "moodboards", filename)

@api_bp.route("/api/duplicates/scan")
@login_required
@require_role(['GESTOR', 'DIRECCION', 'ADMIN'])
def api_scan_duplicates():
    vision_manager = get_vision_manager()
    if not vision_manager:
        return jsonify([])
    
    try:
        deduper = Deduplicator(vision_manager)
        visual_dupes = deduper.find_duplicates()
        
        results = []
        
        def format_doc(meta):
            try:
                p = ensure_within_project(Path(meta["path"]))
                token = encode_path(str(p))
                # Assuming main.vision_preview handles /vision/preview/<token>
                url = f"/vision/preview/{token}" 
            except Exception:
                url = "/static/img/placeholder.png"
                
            return {
                "id": meta.get("doc_id"),
                "filename": meta.get("filename"),
                "path": meta.get("path"),
                "preview_url": url,
                "date": meta.get("date", "")
            }

        for group in visual_dupes:
            primary = format_doc(group["primary"])
            dupes_list = [format_doc(d) for d in group["duplicates"]]
            results.append({
                "primary": primary,
                "duplicates": dupes_list
            })
            
        return jsonify(results)
    except Exception as e:
        get_logger().error(f"Duplicate scan failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/document/<int:doc_id>/verify", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_verify_document(doc_id):
    db = get_db()
    if db.update_document_state(doc_id, "verified"):
        get_logger().info(f"Document {doc_id} manually verified.")
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update state"}), 500

@api_bp.route("/api/document/<int:doc_id>/fields", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_update_fields(doc_id):
    db = get_db()
    data = request.json
    
    if not data or 'fields' not in data:
        return jsonify({"error": "Missing fields data"}), 400
    
    try:
        cursor = db.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404
        
        structured_data = json.loads(row[0]) if row[0] else {}
        structured_data['fields'] = data['fields']
        
        db.execute(
            f"UPDATE ocr_texts SET structured_data = {db.placeholder} WHERE id_doc = {db.placeholder}",
            (json.dumps(structured_data), doc_id),
            commit=True
        )
        
        get_logger().info(f"Updated fields for document {doc_id}")
        return jsonify({"success": True})
    except Exception as e:
        get_logger().error(f"Failed to update fields: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/providers/search")
@login_required
@require_role(["GESTOR", "DIRECCION", "ADMIN"])
def api_search_providers():
    query = request.args.get('q', '').lower()
    try:
        providers_path = PROJECT_ROOT / "data" / "providers.json"
        if providers_path.exists():
            with open(providers_path, 'r', encoding='utf-8') as f:
                providers = json.load(f)
            
            results = []
            for provider_id, provider_data in providers.items():
                canonical = provider_data.get('canonical_name', '')
                aliases = provider_data.get('aliases', [])
                
                if query in canonical.lower() or any(query in alias.lower() for alias in aliases):
                    results.append({
                        'id': provider_id,
                        'name': canonical,
                        'vat': provider_data.get('vat_number', ''),
                        'category': provider_data.get('category', '')
                    })
            return jsonify(results[:10])
        return jsonify([])
    except Exception as e:
        get_logger().error(f"Provider search failed: {e}")
        return jsonify([])

@api_bp.route("/api/document/<int:doc_id>/dismiss-anomaly", methods=["POST"])
@login_required
@hotel_scoped("doc_id")
@owner_scoped("doc_id")
def api_dismiss_anomaly(doc_id):
    db = get_db()

    data = request.json or {}
    anomaly = data.get("anomaly")
    if not anomaly:
        return jsonify({"error": "Missing anomaly code"}), 400

    try:
        row = db.execute(
            f"SELECT structured_data FROM ocr_texts WHERE id_doc = {db.placeholder}",
            (doc_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404

        structured_json = row[0] if isinstance(row, (tuple, list)) else row["structured_data"]
        structured_data = safe_json_parse(structured_json, {})
        if not isinstance(structured_data, dict):
            structured_data = {}

        anomalies = structured_data.get("anomalies", [])

        if isinstance(anomalies, list) and anomaly in anomalies:
            structured_data["anomalies"] = [a for a in anomalies if a != anomaly]

            db.execute(
                f"UPDATE ocr_texts SET structured_data = {db.placeholder} WHERE id_doc = {db.placeholder}",
                (json.dumps(structured_data), doc_id),
                commit=True,
            )

        return jsonify({"success": True})
    except Exception as e:
        get_logger().error(f"Dismiss anomaly failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/document/<int:doc_id>", methods=["DELETE"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_delete_document(doc_id):
    db = get_db()

    doc = db.get_document(doc_id)
    if not doc:
        return jsonify({"error": "Not found"}), 404

    path_str = doc.get("path")
    
    try:
        # 1. Database Cleanup
        deleted = db.delete_document(doc_id)
        if not deleted:
            return jsonify({"error": "Delete failed"}), 500

        # 2. File Cleanup
        try:
            abs_path = _abs_doc_path(path_str)
            if abs_path and abs_path.exists():
                os.remove(abs_path)
        except OSError as e:
            get_logger().error(f"Failed to delete file {path_str}: {e}")
            
        return jsonify({"success": True})
        
    except Exception as e:
        get_logger().error(f"Delete failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/train", methods=["POST"])
@login_required
@require_role(["DIRECCION", "ADMIN"])
def api_train_model():
    db = get_db()
    model_path = PROJECT_ROOT / "data" / "models" / "classifier.pkl"
    trainer = ModelTrainer(db, str(model_path))
    
    success, msg = trainer.train()
    if success:
        import web_app.services as services
        # Naive invalidation
        pass
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500

@api_bp.route("/api/document/<int:doc_id>/enhance", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_enhance_document(doc_id):
    try:
        data = request.json
        contrast = float(data.get("contrast", 1.0))
        brightness = float(data.get("brightness", 1.0))
        sharpness = float(data.get("sharpness", 1.0))
        apply_clahe = bool(data.get("clahe", False))
        
        db = get_db()
        path_str = db.get_document_path(doc_id)
        if not path_str:
            return jsonify({"error": "Document not found"}), 404
            
        original_path = Path(path_str)
        if not original_path.is_absolute():
            original_path = PROJECT_ROOT / original_path
            
        if not original_path.exists():
             return jsonify({"error": "File not found on disk"}), 404

        try:
            # Backup original if not exists
            backup_path = original_path.with_name(f"{original_path.stem}_original{original_path.suffix}")
            if not backup_path.exists():
                import shutil
                shutil.copy2(original_path, backup_path)
                
            with Image.open(backup_path) as img: # Use backup as source for enhancement to always be consistent?
                # Or use current? If we want to chain enhancements, use original_path.
                # But safer to always enhance from original raw. 
                # Let's use original_path (which might be already enhanced if we didn't backup earlier).
                # Actually, if we want "Restore", we need the TRUE original.
                # If we use backup_path as source, we are safe.
                # Let's open backup_path if exists, enabling non-destructive edits.
                if not backup_path.exists(): # Should exist now
                     with Image.open(original_path) as img:
                        enhanced = enhance_image(img, contrast, brightness, sharpness, apply_clahe)
                else:
                     with Image.open(backup_path) as img:
                        enhanced = enhance_image(img, contrast, brightness, sharpness, apply_clahe)
                
                enhanced.save(original_path)
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {e}"}), 500

        pipeline = get_pipeline()
        db.update_document_status(doc_id, "processing")
        
        def reprocess():
            try:
                pipeline.process_single_file(original_path, doc_id=doc_id)
            except Exception as e:
                get_logger().error(f"Re-processing enhanced doc {doc_id} failed: {e}")
                db.update_document_status(doc_id, "error")

        threading.Thread(target=reprocess).start()
        
        return jsonify({
            "status": "success", 
            "message": "Imagen mejorada. Re-procesando OCR..."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/shutdown", methods=["POST"])
@login_required
@require_role(["ADMIN"])
def shutdown():
    # Never expose server shutdown in production deployments.
    if not current_app.debug:
        return jsonify({"error": "Not found"}), 404

    shutdown_func = request.environ.get("werkzeug.server.shutdown")
    if shutdown_func:
        shutdown_func()
        return "Server shutting down..."
    return jsonify({"error": "Shutdown not supported"}), 400

@api_bp.route("/api/visual_search")
@login_required
def api_visual_search():
    query = request.args.get("q", "").strip()
    k = int(request.args.get("k", 50))

    vision_manager = get_vision_manager()

    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    role = str(getattr(current_user, "role", "")).upper()
    scope_list = list(getattr(current_user, "hotel_scope", []) or [])
    scope_set = {str(h) for h in scope_list}
    owner_filter = current_user.id if role in {"CLIENTE", "CLIENT"} else None

    # Fail closed: unscoped users must not see global results.
    if role != "ADMIN" and not scope_list:
        return jsonify([])

    try:
        db = get_db()
        schema = _documents_schema(db)
        path_col = schema["path_col"]
        created_col = schema["created_col"]
        if query:
            raw_results = vision_manager.search_by_text(query, k=k)
        else:
            get_logger().info("Empty query, fetching recent docs from DB")
            q = (
                f"SELECT id, filename, {path_col} AS path, {created_col} AS created_at, "
                "tags, owner_id, hotel_id FROM documents WHERE 1=1"
            )
            params = []
            if role != "ADMIN":
                placeholders = ",".join([db.placeholder] * len(scope_list))
                q += f" AND hotel_id IN ({placeholders})"
                params.extend(scope_list)
            if owner_filter is not None:
                q += f" AND owner_id = {db.placeholder}"
                params.append(owner_filter)
            q += f" ORDER BY id DESC LIMIT {db.placeholder}"
            params.append(k)
            rows = db.execute(q, tuple(params)).fetchall()

            clean_results = []
            for r in rows:
                row_path = _row_get(r, "path")
                if row_path and Path(row_path).suffix.lower() in ALLOWED_IMAGE_EXTS:
                    clean_results.append({
                        "id": _row_get(r, "id"),
                        "filename": _row_get(r, "filename"),
                        "path": row_path,
                        "score": 0.0,
                        "date": _row_get(r, "created_at"),
                        "tags": safe_json_parse(_row_get(r, "tags"), []),
                    })
            return jsonify(clean_results)

        clean_results = []
        for res in raw_results:
            path = res.get("path")
            if not path or Path(path).suffix.lower() not in ALLOWED_IMAGE_EXTS:
                continue

            row = _find_document_by_path(
                db,
                path_col,
                path,
                f"id, filename, {path_col} AS path, {created_col} AS created_at, tags, owner_id, hotel_id",
            )

            if row:
                owner_id = _row_get(row, "owner_id")
                hotel_id = _row_get(row, "hotel_id")

                if role != "ADMIN":
                    if hotel_id is None or str(hotel_id) not in scope_set:
                        continue
                if owner_filter is not None and str(owner_id) != str(owner_filter):
                    continue

                tags = safe_json_parse(_row_get(row, "tags"), [])
                has_color_tags = any(t.startswith("color:") for t in tags)
                row_path = _row_get(row, "path")
                abs_row_path = _abs_doc_path(row_path)
                if (
                    not has_color_tags
                    and vision_manager
                    and abs_row_path
                    and abs_row_path.exists()
                ):
                     try:
                         colors = vision_manager.analyze_colors(str(abs_row_path), num_colors=4)
                         for c in colors:
                             tags.append(f"color:{c['hex']}")
                     except Exception:
                         pass

                clean_results.append({
                    "id": _row_get(row, "id"),
                    "filename": _row_get(row, "filename"),
                    "path": row_path or path,
                    "score": res["score"],
                    "date": _row_get(row, "created_at"),
                    "tags": tags
                })
        return jsonify(clean_results)

    except Exception as e:
        get_logger().error(f"Gallery API failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/documents/search")
@login_required
def api_documents_search():
    """API endpoint for documents search with filters (AJAX)."""
    db = get_db()
    role = str(getattr(current_user, "role", "")).upper()
    
    # Get filter parameters
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    offset = (page - 1) * per_page
    status_filter = request.args.get("status", "")
    type_filter = request.args.get("type", "")
    search_term = request.args.get("search", "")
    
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
            return jsonify({"documents": [], "total": 0, "page": 1, "total_pages": 1})
        placeholders = ",".join([db.placeholder] * len(current_user.hotel_scope))
        scope_filter = f" AND hotel_id IN ({placeholders})"
        scope_params = list(current_user.hotel_scope)
    
    # Client isolation
    owner_filter = ""
    owner_params = []
    if role in {"CLIENTE", "CLIENT"}:
        owner_filter = f" AND owner_id = {db.placeholder}"
        owner_params = [current_user.id]
    
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        
        query = f"""
            SELECT id, filename, {path_col}, {type_col}, status, {created_col}, {file_size_select}, tags, error_message
            FROM documents
            WHERE 1=1
        """
        query += scope_filter
        query += owner_filter
        
        params = []
        params.extend(scope_params)
        params.extend(owner_params)
        
        if status_filter:
            query += f" AND status = {db.placeholder}"
            params.append(status_filter)
        if type_filter:
            query += f" AND {type_col} = {db.placeholder}"
            params.append(type_filter)
        if search_term:
            query += (
                f" AND (LOWER(filename) LIKE LOWER({db.placeholder})"
                f" OR LOWER({type_col}) LIKE LOWER({db.placeholder}))"
            )
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM documents WHERE 1=1" + scope_filter + owner_filter
        count_params = list(params[:len(scope_params) + len(owner_params)])
        
        if status_filter:
            count_query += f" AND status = {db.placeholder}"
            count_params.append(status_filter)
        if type_filter:
            count_query += f" AND {type_col} = {db.placeholder}"
            count_params.append(type_filter)
        if search_term:
            count_query += f" AND (LOWER(filename) LIKE LOWER({db.placeholder}) OR LOWER({type_col}) LIKE LOWER({db.placeholder}))"
            count_params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        cursor.execute(count_query, count_params)
        total_docs = cursor.fetchone()[0]
        
        query += f" ORDER BY {created_col} DESC LIMIT {db.placeholder} OFFSET {db.placeholder}"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        documents_rows = cursor.fetchall()
    
    total_pages = max(1, (total_docs + per_page - 1) // per_page)
    
    # Convert to dict
    documents = []
    for row in documents_rows:
        doc = {
            "id": row[0],
            "filename": row[1],
            "path": row[2],
            "doc_type": row[3],
            "status": row[4],
            "created_at": str(row[5]) if row[5] else None,
            "file_size": row[6],
            "tags": row[7] if isinstance(row[7], list) else [],
            "error_message": row[8]
        }
        documents.append(doc)
    
    return jsonify({
        "documents": documents,
        "total": total_docs,
        "page": page,
        "total_pages": total_pages
    })


@api_bp.route("/api/search")
@login_required
def api_search():
    """API endpoint for full-text search."""
    query = request.args.get("q", "")

    role = str(getattr(current_user, "role", "")).upper()
    scope_list = list(getattr(current_user, "hotel_scope", []) or [])
    owner_filter = current_user.id if role in {"CLIENTE", "CLIENT"} else None

    # Fail closed for unscoped users.
    if role != "ADMIN" and not scope_list:
        return jsonify([])

    if not query:
        db = get_db()
        schema = _documents_schema(db)
        path_col = schema["path_col"]
        created_col = schema["created_col"]
        q = f"SELECT id, filename, {path_col} AS path, {created_col} AS created_at, tags FROM documents WHERE 1=1"
        params = []
        if role != "ADMIN":
            placeholders = ",".join([db.placeholder] * len(scope_list))
            q += f" AND hotel_id IN ({placeholders})"
            params.extend(scope_list)
        if owner_filter is not None:
            q += f" AND owner_id = {db.placeholder}"
            params.append(owner_filter)
        q += " ORDER BY id DESC LIMIT 20"
        rows = db.execute(q, tuple(params)).fetchall()
        clean_results = []
        for r in rows:
            clean_results.append({
                "id": _row_get(r, "id"),
                "filename": _row_get(r, "filename"),
                "path": _row_get(r, "path"),
                "date": _row_get(r, "created_at"),
                "tags": safe_json_parse(_row_get(r, "tags"), []),
            })
        return jsonify(clean_results)

    local_db = get_db()
    results = local_db.search_documents(
        query, hotel_ids=None if role == "ADMIN" else scope_list, owner_id=owner_filter
    )
    return jsonify([
        {"id": r[0], "filename": r[1], "snippet": r[2]}
        for r in results
    ])

@api_bp.route("/api/search/similar/<int:doc_id>")
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_search_similar(doc_id):
    db = get_db()
    vision_manager = get_vision_manager()
    
    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    doc = db.get_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    source_path = _abs_doc_path(doc.get("path"))
    if not source_path or not source_path.exists():
        return jsonify([])

    try:
        results = vision_manager.search_similar(str(source_path), k=10)
        formatted_results = []
        role = str(getattr(current_user, "role", "")).upper()
        scope_set = {str(h) for h in getattr(current_user, "hotel_scope", []) or []}
        schema = _documents_schema(db)
        path_col = schema["path_col"]

        for item in results:
            item_path = item.get("path")
            item_abs = _abs_doc_path(item_path)
            if item_abs and os.path.abspath(str(item_abs)) == os.path.abspath(str(source_path)):
                continue

            db_doc = _find_document_by_path(
                db,
                path_col,
                item_path,
                f"id, filename, {path_col} AS path, owner_id, hotel_id",
            )

            if not db_doc:
                continue

            owner_id = _row_get(db_doc, "owner_id")
            hotel_id = _row_get(db_doc, "hotel_id")

            # Multi-tenant isolation: never leak other hotels' documents.
            if role != "ADMIN":
                if hotel_id is None or str(hotel_id) not in scope_set:
                    continue
            if role in {"CLIENTE", "CLIENT"} and str(owner_id) != str(current_user.id):
                continue

            formatted_results.append(
                {
                    "id": _row_get(db_doc, "id"),
                    "filename": _row_get(db_doc, "filename"),
                    "path": _row_get(db_doc, "path") or (str(item_abs) if item_abs else item_path),
                    "score": round(float(item["score"]), 4),
                }
            )
        return jsonify(formatted_results)
    except Exception as e:
        get_logger().error(f"Similar search failed: {e}")
        return jsonify([])

@api_bp.route("/api/tasks")
@login_required
@require_role(["GESTOR", "DIRECCION", "ADMIN"])
def api_tasks():
    try:
        role = str(getattr(current_user, "role", "")).upper()
        scope_list = list(getattr(current_user, "hotel_scope", []) or [])

        # Huey is global; don't leak cross-tenant task payloads to non-admin users.
        pending_tasks = huey.pending() if role == "ADMIN" else []
        scheduled_tasks = huey.scheduled() if role == "ADMIN" else []

        # Format for UI
        tasks_list = []

        # 1. Get DB "Processing" documents (The real "active" tasks from user perspective)
        db = get_db()
        if role != "ADMIN" and not scope_list:
            processing_docs = []
        else:
            schema = _documents_schema(db)
            type_col = schema["type_col"]
            q = f"SELECT id, filename, {type_col} AS doc_type FROM documents WHERE status = 'processing'"
            params = []
            if role != "ADMIN":
                placeholders = ",".join([db.placeholder] * len(scope_list))
                q += f" AND hotel_id IN ({placeholders})"
                params.extend(scope_list)
            processing_docs = db.execute(q, tuple(params)).fetchall()
        
        for doc in processing_docs:
            tasks_list.append({
                "id": f"DOC-{_row_get(doc, 'id', 0)}",
                "name": f"Procesando: {_row_get(doc, 'filename', 1)}",
                "args": f"Tipo: {_row_get(doc, 'doc_type', 2) or 'Detectando...'}",
                "status": "active", # Custom status
                "progress": 50 # Mock progress or fetch if available
            })

        # 2. Get Huey Pending (Waiting in queue)
        for task in pending_tasks:
            tasks_list.append({
                "id": task.id if hasattr(task, 'id') else str(task),
                "name": task.name if hasattr(task, 'name') else "Tarea en cola",
                "args": str(task.args) if hasattr(task, 'args') else "",
                "status": "pending",
                "progress": 0
            })

        return jsonify({
            "pending_count": len(pending_tasks) if role == "ADMIN" else 0,
            "scheduled_count": len(scheduled_tasks) if role == "ADMIN" else 0,
            "pending_details": tasks_list[:50]
        })
    except Exception as e:
        get_logger().error(f"Task monitor error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/templates", methods=["GET"])
@login_required
@require_role(["GESTOR", "DIRECCION", "ADMIN"])
def api_list_templates():
    db = get_db()
    templates = db.get_templates()
    return jsonify(templates)

@api_bp.route("/api/templates/create", methods=["POST"])
@login_required
@require_role(["GESTOR", "DIRECCION", "ADMIN"])
def api_create_template():
    data = request.json
    name = data.get("name")
    description = data.get("description", "")
    zones = data.get("zones", []) # JSON object/array
    
    if not name:
        return jsonify({"error": "Template name required"}), 400
        
    db = get_db()
    try:
        t_id = db.insert_template(name, description, json.dumps(zones))
        return jsonify({"success": True, "id": t_id})
    except Exception as e:
        get_logger().error(f"Create template failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/templates/<int:t_id>", methods=["DELETE"])
@login_required
@require_role(["GESTOR", "DIRECCION", "ADMIN"])
def api_delete_template(t_id):
    db = get_db()
    if db.delete_template(t_id):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to delete"}), 500

@api_bp.route("/api/document/<int:doc_id>/export/dxf", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_export_dxf(doc_id):
    """
    Export a document (image) to DXF format for CAD.
    """
    db = get_db()
    document = db.get_document(doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404
    
    file_path = _abs_doc_path(document.get("path"))
    if not file_path or not file_path.exists():
        return jsonify({"error": "File parsing failed: File not found on disk"}), 404

    # Output path
    output_filename = f"{os.path.splitext(document.get('filename'))[0]}.dxf"
    # We save to a temp location or processed folder? 
    # Let's save next to original for simplicity or in temp.
    # Actually, we want to return it as a download.
    # We'll create a temp file.
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    
    try:
        from modules.vectorization_manager import VectorizationManager
        vm = VectorizationManager(logger=get_logger())
        success = vm.raster_to_dxf(str(file_path), temp_path)
        
        if success:
            response = send_file(
                temp_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype="application/dxf"
            )
            # Clean up temp file after the response is fully sent (Windows-safe).
            response.call_on_close(lambda: os.path.exists(temp_path) and os.remove(temp_path))
            return response
        else:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            return jsonify({"error": "Vectorization failed (could not detect edges or empty result)"}), 500
            
    except Exception as e:
        get_logger().error(f"DXF Export failed: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/metrics/history")
@login_required
@require_role(["DIRECCION", "ADMIN"])
def api_metrics_history():
    try:
        limit = int(request.args.get("limit", 20))
        db = get_db()
        cursor = db.execute(
            """
            SELECT datetime, ok_docs, failed_docs, avg_time, reliability_pct
            FROM metrics
            ORDER BY datetime DESC
            LIMIT ?
            """, (limit,)
        )
        rows = cursor.fetchall()
        
        history = []
        for r in rows:
            history.append({
                "date": r[0],
                "ok": r[1],
                "failed": r[2],
                "avg_time": round(r[3], 2),
                "reliability": round(r[4], 1)
            })
            
        return jsonify(history)
    except Exception as e:
        get_logger().error(f"Metrics history failed: {e}")
        return jsonify([])

@api_bp.route("/api/document/<int:doc_id>/generate_proposal", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_generate_proposal(doc_id):
    """
    Generates a PDF proposal (dossier) for the document.
    """
    try:
        from modules.proposal_manager import ProposalManager
        
        db = get_db()
        doc = db.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404
            
        # Get structured data for detected items
        cursor = db.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        structured_data = json.loads(row[0]) if row and row[0] else {}
        
        items = []
        
        # 1. Add the main image itself
        main_image_path = _abs_doc_path(doc.get("path"))
        if main_image_path and main_image_path.exists():
            items.append({
                'label': f"Vista General ({doc.get('type') or doc.get('doc_type') or 'Unknown'})",
                'image_path': str(main_image_path),
                'price': 'Consultar'
            })
        
        # 2. Add crops if available (assuming structured_data['crops'] contains paths)
        if 'crops' in structured_data:
            for crop in structured_data['crops']:
                crop_path = _abs_doc_path(crop.get('path', ''))
                if crop_path and crop_path.exists():
                    items.append({
                        'label': crop.get('label', 'Elemento'),
                        'image_path': str(crop_path),
                        'price': 'Consultar'
                    })

        if not items:
            return jsonify({"error": "No valid images found for proposal generation"}), 404
        
        # Output setup
        filename = f"Propuesta_{doc.get('filename')}_{doc_id}.pdf"
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        
        pm = ProposalManager(logger=get_logger())
        success = pm.generate_proposal(doc.get('filename'), items, temp_path)
        
        if success and os.path.exists(temp_path):
            response = send_file(
                temp_path,
                as_attachment=True,
                download_name=filename,
                mimetype="application/pdf"
            )
            response.call_on_close(lambda: os.path.exists(temp_path) and os.remove(temp_path))
            return response
        else:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            return jsonify({"error": "Failed to generate PDF"}), 500

    except Exception as e:
        get_logger().error(f"Proposal generation failed: {e}")
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/document/<int:doc_id>/generate_order", methods=["POST"])
@login_required
@hotel_scoped('doc_id')
@owner_scoped("doc_id")
def api_generate_order(doc_id):
    """
    Export detected items to an Excel order list ('Del Moodboard al Pedido').
    """
    try:
        import pandas as pd
        import tempfile
        
        db = get_db()
        doc = db.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404
            
        cursor = db.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        structured_data = json.loads(row[0]) if row and row[0] else {}
        
        # Extract items from structured data or tags
        items = []
        
        # 1. From 'crops' (if Visual Intelligence ran)
        if 'crops' in structured_data:
            for crop in structured_data['crops']:
                items.append({
                    "Referencia": f"AUTO-{crop.get('label', 'Item')[:3].upper()}-{doc_id}",
                    "Descripción": crop.get('label', 'Elemento Decorativo'),
                    "Cantidad": 1,
                    "Proveedor": "A determinar",
                    "Precio Est.": "Consultar",
                    "Estado": "Pendiente",
                    "Origen": "Detección Visual"
                })
                
        # 2. From Tags (if no crops but tags exist)
        elif doc.get('tags'):
            for tag in doc.get('tags'):
                if ":" not in tag: # Avoid system tags like color:
                    items.append({
                        "Referencia": "ETIQUETA",
                        "Descripción": tag,
                        "Cantidad": 1,
                        "Proveedor": "A determinar",
                        "Precio Est.": "Consultar",
                        "Estado": "Pendiente",
                        "Origen": "Etiquetado Auto"
                    })
        
        if not items:
             items.append({"Descripción": "No se detectaron elementos específicos para listar."})

        df = pd.DataFrame(items)
        
        # Create Excel
        filename = f"Pedido_{doc.get('filename')}_{doc_id}.xlsx"
        fd, temp_path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        
        df.to_excel(temp_path, index=False)
        
        response = send_file(
            temp_path,
            as_attachment=True,
            download_name=filename,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.call_on_close(lambda: os.path.exists(temp_path) and os.remove(temp_path))
        return response

    except Exception as e:
        get_logger().error(f"Order generation failed: {e}")
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/document/<int:doc_id>/interpret_blueprint", methods=["POST"])
@login_required
@hotel_scoped("doc_id")
@owner_scoped("doc_id")
def api_interpret_blueprint(doc_id):
    """
    Analyze OCR text to extracting blueprint metadata (Scale, Rooms, Areas).
    """
    try:
        from modules.blueprint_interpreter import BlueprintInterpreter
        from web_app.services import get_llm_client

        db = get_db()

        doc = db.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # We need the OCR text for analysis.
        row = db.execute(
            f"SELECT text FROM ocr_texts WHERE id_doc = {db.placeholder}",
            (doc_id,),
        ).fetchone()
        raw_text = ""
        if row:
            raw_text = row[0] if isinstance(row, (tuple, list)) else (row["text"] or "")

        # Get image path for optional multimodal analysis.
        image_path_obj = _abs_doc_path(doc.get("path") if isinstance(doc, dict) else None)
        image_path = str(image_path_obj) if image_path_obj and image_path_obj.exists() else None

        interpreter = BlueprintInterpreter()
        llm = get_llm_client()
        
        # Pass LLM client + Image Path for Vision (Sketch) Analysis
        metadata = interpreter.infer_metadata(
            text=raw_text, 
            llm_client=llm, 
            image_path=image_path
        )
        
        # Save this metadata back to DB (optional, in 'structured_data' or tags)
        return jsonify(metadata)

    except Exception as e:
        get_logger().error(f"Blueprint interpretation failed: {e}")
        return jsonify({"error": str(e)}), 500
