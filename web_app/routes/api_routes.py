import os
import json
import threading
import tempfile
from pathlib import Path
from flask import Blueprint, jsonify, request, url_for, send_from_directory
from PIL import Image

from pydantic import ValidationError

from web_app.services import get_db, get_pipeline, get_logger, PROJECT_ROOT
from web_app.utils import safe_json_parse, ensure_within_project, encode_path

from modules.schemas import DocumentUpdateSchema
from modules.moodboard import MoodboardGenerator
from modules.deduplicator import Deduplicator
from modules.learning import ModelTrainer
from modules.learning import ModelTrainer
from modules.image_utils import enhance_image
from modules.tasks import huey

api_bp = Blueprint('api', __name__)

@api_bp.route("/api/document/<int:doc_id>/update", methods=["POST"])
def api_update_document(doc_id: int):
    """Update document data with Pydantic validation."""
    try:
        data = request.json
        validated = DocumentUpdateSchema(**data)
        
        db = get_db()
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            
            # --- Update 'documents' table ---
            doc_updates = []
            doc_params = []
            
            if validated.filename:
                doc_updates.append(f"filename = {db.placeholder}")
                doc_params.append(validated.filename)
            if validated.type:
                doc_updates.append(f"type = {db.placeholder}")
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
def api_create_moodboard():
    data = request.json
    doc_ids = data.get("ids", [])
    title = data.get("title", "Mi Moodboard")

    if not doc_ids:
        return jsonify({"error": "No documents selected"}), 400

    db = get_db()
    
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        ph = db.placeholder
        placeholders = ",".join(ph for _ in doc_ids)
        query = f"SELECT path FROM documents WHERE id IN ({placeholders})"
        cursor.execute(query, tuple(doc_ids))
        rows = cursor.fetchall()
    
    paths = [(row["path"] if isinstance(row, dict) else row[0]) for row in rows]
    
    if not paths:
        return jsonify({"error": "No valid images found for selected IDs"}), 404

    generator = MoodboardGenerator(output_dir=PROJECT_ROOT / "data" / "moodboards")
    try:
        output_path = generator.create(paths, title)
        if not output_path:
             return jsonify({"error": "Failed to generate moodboard (invalid images?)"}), 500
             
        filename = Path(output_path).name
        url = url_for("api.serve_moodboard", filename=filename) 
        return jsonify({"url": url})
        
    except Exception as e:
        get_logger().error(f"Moodboard generation error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/moodboard_file/<filename>")
def serve_moodboard(filename):
    return send_from_directory(PROJECT_ROOT / "data" / "moodboards", filename)

@api_bp.route("/api/duplicates/scan")
def api_scan_duplicates():
    pipeline = get_pipeline()
    if not pipeline.vision_manager:
        return jsonify([])
    
    try:
        deduper = Deduplicator(pipeline.vision_manager)
        visual_dupes = deduper.find_duplicates()
        
        results = []
        
        def format_doc(meta):
            try:
                p = ensure_within_project(Path(meta["path"]))
                token = encode_path(str(p))
                # Assuming main.vision_preview handles /vision/preview/<token>
                url = f"/vision/preview/{token}" 
            except:
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
def api_verify_document(doc_id):
    db = get_db()
    if db.update_document_state(doc_id, "verified"):
        get_logger().info(f"Document {doc_id} manually verified.")
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update state"}), 500

@api_bp.route("/api/document/<int:doc_id>/fields", methods=["POST"])
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
def api_dismiss_anomaly(doc_id):
    db = get_db()
    data = request.json
    
    if not data or 'anomaly' not in data:
        return jsonify({"error": "Missing anomaly code"}), 400
    
    try:
        cursor = db.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404
        
        structured_data = json.loads(row[0]) if row[0] else {}
        anomalies = structured_data.get('anomalies', [])
        
        if data['anomaly'] in anomalies:
            anomalies.remove(data['anomaly'])
            structured_data['anomalies'] = anomalies
            
            db.execute(
                "UPDATE ocr_texts SET structured_data = ? WHERE id_doc = ?",
                (json.dumps(structured_data), doc_id)
            )
            db.conn.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/document/<int:doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    db = get_db()
    
    row = db.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
        
    path_str = (row["path"] if isinstance(row, dict) else row[0])
    
    try:
        if hasattr(db, "delete_document"):
             db.delete_document(doc_id)
        else:
             db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
             db.execute("DELETE FROM ocr_texts WHERE id_doc = ?", (doc_id,))
             db.conn.commit()

        try:
            if os.path.exists(path_str):
                os.remove(path_str)
        except OSError as e:
            get_logger().error(f"Failed to delete file {path_str}: {e}")
            
        return jsonify({"success": True})
        
    except Exception as e:
        get_logger().error(f"Delete failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/train", methods=["POST"])
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
def shutdown():
    shutdown_func = request.environ.get("werkzeug.server.shutdown")
    if shutdown_func:
        shutdown_func()
    return "Server shutting down..."

@api_bp.route("/api/visual_search")
def api_visual_search():
    query = request.args.get("q", "").strip()
    k = int(request.args.get("k", 50))
    
    pipeline = get_pipeline()
    vision_manager = pipeline.vision_manager
    results = []

    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    try:
        ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif", ".gif", ".tif", ".tiff"}
        if query:
            raw_results = vision_manager.search_by_text(query, k=k)
        else:
            get_logger().info("Empty query, fetching recent docs from DB")
            db = get_db()
            rows = db.execute(
                "SELECT id, filename, path, datetime, tags FROM documents ORDER BY id DESC LIMIT ?", 
                (k,)
            ).fetchall()
            
            clean_results = []
            for r in rows:
                if Path(r["path"]).suffix.lower() in ALLOWED_IMAGE_EXTS:
                    clean_results.append({
                        "id": r["id"],
                        "filename": r["filename"],
                        "path": r["path"],
                        "score": 0.0,
                        "date": r["datetime"],
                        "tags": json.loads(r["tags"]) if r["tags"] else []
                    })
            return jsonify(clean_results)

        db = get_db()
        clean_results = []
        for res in raw_results:
            path = res["path"]
            row = db.execute(
                "SELECT id, filename, datetime, tags FROM documents WHERE path = ?", 
                (path,)
            ).fetchone()
            
            if row:
                tags = json.loads(row["tags"]) if row["tags"] else []
                has_color_tags = any(t.startswith("color:") for t in tags)
                if not has_color_tags and vision_manager and os.path.exists(path):
                     try:
                         colors = vision_manager.analyze_colors(path, num_colors=4)
                         for c in colors:
                             tags.append(f"color:{c['hex']}")
                     except Exception:
                         pass

                clean_results.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "path": path,
                    "score": res["score"],
                    "date": row["datetime"],
                    "tags": tags
                })
        return jsonify(clean_results)

    except Exception as e:
        get_logger().error(f"Gallery API failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/search")
def api_search():
    """API endpoint for full-text search."""
    query = request.args.get("q", "")
    if not query:
        db = get_db()
        rows = db.execute("SELECT id, filename, path, datetime, tags FROM documents ORDER BY id DESC LIMIT 20").fetchall()
        clean_results = []
        for r in rows:
            clean_results.append({
                "id": r["id"],
                "filename": r["filename"],
                "path": r["path"],
                "date": r["datetime"],
                "tags": json.loads(r["tags"]) if r["tags"] else []
            })
        return jsonify(clean_results)
    
    local_db = get_db()
    results = local_db.search_documents(query)
    return jsonify([
        {"id": r[0], "filename": r[1], "snippet": r[2]} 
        for r in results
    ])

@api_bp.route("/api/search/similar/<int:doc_id>")
def api_search_similar(doc_id):
    db = get_db()
    pipeline = get_pipeline()
    vision_manager = pipeline.vision_manager
    
    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    doc = db.get_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    file_path = doc.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify([])

    try:
        results = vision_manager.search_similar(file_path, k=10)
        formatted_results = []
        for item in results:
            if os.path.abspath(item["path"]) == os.path.abspath(file_path):
                continue

            path_obj = Path(item["path"])
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, filename FROM documents WHERE path = %s", (str(path_obj),))
                db_doc = cursor.fetchone()
                
            if db_doc:
                formatted_results.append({
                    "id": db_doc[0],
                    "filename": db_doc[1],
                    "path": str(path_obj),
                    "score": round(float(item["score"]), 4)
                })
        return jsonify(formatted_results)
    except Exception as e:
        get_logger().error(f"Similar search failed: {e}")
        return jsonify([])

@api_bp.route("/api/tasks")
def api_tasks():
    try:
        # Huey specific: pending() returns list of tasks in queue
        pending_tasks = huey.pending() 
        scheduled_tasks = huey.scheduled()
        
        # Format for UI
        tasks_list = []
        
        # 1. Get DB "Processing" documents (The real "active" tasks from user perspective)
        db = get_db()
        processing_docs = db.execute("SELECT id, filename, type FROM documents WHERE status = 'processing'").fetchall()
        
        for doc in processing_docs:
            tasks_list.append({
                "id": f"DOC-{doc[0]}",
                "name": f"Procesando: {doc[1]}",
                "args": f"Tipo: {doc[2] or 'Detectando...'}",
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
            "pending_count": len(pending_tasks),
            "scheduled_count": len(scheduled_tasks),
            "pending_details": tasks_list[:50] 
        })
    except Exception as e:
        get_logger().error(f"Task monitor error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/templates", methods=["GET"])
def api_list_templates():
    db = get_db()
    templates = db.get_templates()
    return jsonify(templates)

@api_bp.route("/api/templates/create", methods=["POST"])
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
def api_delete_template(t_id):
    db = get_db()
    if db.delete_template(t_id):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to delete"}), 500
