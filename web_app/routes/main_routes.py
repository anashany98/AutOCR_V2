import os
import json
from pathlib import Path
from typing import List, Any, Dict
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, send_from_directory
from werkzeug.utils import secure_filename
import tempfile

from web_app.services import get_db, get_pipeline, get_logger, get_classifier, load_configuration, save_configuration, PROJECT_ROOT
from web_app.utils import safe_json_parse, resolve_path, ensure_within_project, encode_path, decode_path
from modules.file_utils import ensure_directories
from modules.tasks import process_document_task

main_bp = Blueprint('main', __name__)

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif", ".gif", ".tif", ".tiff"}

@main_bp.route("/")
def index():
    return redirect(url_for('main.dashboard'))

@main_bp.route("/dashboard")
def dashboard():
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
        status_stats = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) FROM documents WHERE workflow_state = 'pending'")
        pending_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT type, COUNT(*) FROM documents
            WHERE type IS NOT NULL
            GROUP BY type
            """
        )
        raw_type_stats = cursor.fetchall()
        
        # Normalize types in Python (e.g., "invoice" -> "Invoice")
        normalized_stats = {}
        for doc_type, count in raw_type_stats:
            clean_type = doc_type.strip().title() if doc_type else "Desconocido"
            normalized_stats[clean_type] = normalized_stats.get(clean_type, 0) + count
            
        # Sort and limit to top 10
        type_stats = sorted(normalized_stats.items(), key=lambda x: x[1], reverse=True)[:10]

        cursor.execute(
            """
            SELECT id, filename, type, status, datetime, duration, error_message
            FROM documents
            ORDER BY datetime DESC
            LIMIT 10
            """
        )
        recent_docs = cursor.fetchall()

        cursor.execute(
            """
            SELECT datetime, ok_docs, failed_docs, avg_time, reliability_pct
            FROM metrics
            ORDER BY datetime DESC
            LIMIT 5
            """
        )
        metrics = cursor.fetchall()

        cursor.execute(
            """
            SELECT d.id, d.filename, d.datetime, o.tables_json
            FROM documents d
            JOIN ocr_texts o ON d.id = o.id_doc
            WHERE o.tables_json IS NOT NULL
            ORDER BY d.datetime DESC
            LIMIT 10
            """
        )
        tables_rows = cursor.fetchall()
        recent_tables: List[Dict[str, Any]] = []
        for row in tables_rows:
            try:
                tables_data = json.loads(row[3]) if row[3] else []
            except json.JSONDecodeError:
                tables_data = []
            for index, table in enumerate(tables_data):
                recent_tables.append(
                    {
                        "doc_id": row[0],
                        "filename": row[1],
                        "datetime": row[2],
                        "index": index,
                        "csv_path": table.get("csv_path"),
                        "json_path": table.get("json_path"),
                        "structure": table.get("structure", {}),
                    }
                )

    image_results = session.pop("image_results", None)
    image_error = session.pop("image_error", None)

    pipeline = get_pipeline()
    vision_enabled = pipeline.vision_manager is not None and pipeline.vision_manager.config.enabled

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
    )

@main_bp.route("/verify")
def verify_queue():
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(
            """
            SELECT d.id, d.filename, d.datetime, o.confidence, d.type
            FROM documents d
            LEFT JOIN ocr_texts o ON d.id = o.id_doc
            WHERE d.workflow_state = 'pending'
            ORDER BY d.datetime ASC
            """
        )
        pending_docs = cursor.fetchall()
    
    return render_template("documents.html", 
                           documents=pending_docs, 
                           title="Cola de Verificación",
                           is_verification_list=True,
                           total_pages=1,
                           page=1,
                           status_filter="",
                           type_filter="",
                           search="")

@main_bp.route("/documents")
def documents():
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)

        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        offset = (page - 1) * per_page

        status_filter = request.args.get("status")
        type_filter = request.args.get("type")
        search_term = request.args.get("search", "")

        query = """
            SELECT id, filename, path, type, status, datetime, duration, tags, error_message
            FROM documents
            WHERE 1=1
        """
        params: List[Any] = []

        if status_filter:
            query += f" AND status = {db.placeholder}"
            params.append(status_filter)
        if type_filter:
            query += f" AND type = {db.placeholder}"
            params.append(type_filter)
        if search_term:
            query += f" AND (filename ILIKE {db.placeholder} OR type ILIKE {db.placeholder})"
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        query += f" ORDER BY datetime DESC LIMIT {db.placeholder} OFFSET {db.placeholder}"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        documents_rows = cursor.fetchall()
        
        count_query = "SELECT COUNT(*) FROM documents WHERE 1=1"
        count_params: List[Any] = []
        if status_filter:
            count_query += f" AND status = {db.placeholder}"
            count_params.append(status_filter)
        if type_filter:
            count_query += f" AND type = {db.placeholder}"
            count_params.append(type_filter)
        if search_term:
            query_search = f" AND (filename LIKE {db.placeholder} OR type LIKE {db.placeholder})"
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
def document_detail(doc_id: int):
    db = get_db()
    
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(
            f"""
            SELECT d.id, d.filename, d.path, d.type, d.status, d.datetime, d.duration,
                   d.tags, d.workflow_state, o.text, o.markdown_text, o.language, o.confidence,
                   o.blocks_json, o.tables_json, o.structured_data
            FROM documents d
            LEFT JOIN ocr_texts o ON d.id = o.id_doc
            WHERE d.id = {db.placeholder}
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
    if not row:
        flash("Documento no encontrado.", "error")
        return redirect(url_for("main.documents"))

    document = {
        "id": row[0],
        "filename": row[1],
        "path": row[2],
        "type": row[3],
        "status": row[4],
        "datetime": row[5],
        "duration": row[6],
        "workflow_state": row[8],
        "text": row[9],
        "markdown": row[10],
        "language": row[11],
        "confidence": row[12],
    }

    document["tags"] = safe_json_parse(row[7], [])
    document["blocks"] = safe_json_parse(row[13], [])
    document["tables"] = safe_json_parse(row[14], [])
    document["structured_data"] = safe_json_parse(row[15], None)

    return render_template("document_detail.html", document=document)

@main_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "files" not in request.files:
            flash("Debe seleccionar archivos.", "error")
            return redirect(request.url)

        upload_dir = current_app.config["UPLOAD_FOLDER"]
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

        for temp_path in saved_files:
            options = {
                "delete_original": True,
                "ocr_enabled": ocr_enabled,
                "classification_enabled": classification_enabled,
                "input_root": upload_dir,
                "handwriting_mode": handwriting_mode
            }
            process_document_task(temp_path, options)

        flash(f"Se han puesto en cola {len(saved_files)} archivos para procesamiento en segundo plano.", "success")
        return redirect(url_for('main.dashboard'))

    return render_template("upload.html")

@main_bp.route("/chat")
def chat():
    return render_template("chat.html")

@main_bp.route("/duplicates")
def duplicates_page():
    return render_template("duplicates.html")

@main_bp.route("/settings", methods=["GET", "POST"])
def settings():
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
                flash("Configuración de Hot Folder actualizada.", "success")
                
            elif action == "pipeline":
                app_conf = config.setdefault("app", {})
                app_conf["gpu_enabled"] = "gpu_enabled" in request.form
                
                pipe_conf = config.setdefault("ocr_pipeline", {})
                pipe_conf.setdefault("fusion", {})["priority"] = [] # logic for engine selection could be complex
                # Simplified for UI mapping
                pipe_conf["primary_engine"] = request.form.get("primary_engine", "auto")
                
                post_conf = config.setdefault("postbatch", {})
                post_conf["languages"] = [l.strip() for l in request.form.get("languages", "es").split(",")]
                flash("Configuración de Pipeline actualizada.", "success")
                
            elif action == "email_import":
                email_conf = config.setdefault("email_importer", {})
                email_conf["enabled"] = "email_enabled" in request.form
                email_conf["host"] = request.form.get("email_host", "").strip()
                email_conf["port"] = int(request.form.get("email_port", 993))
                email_conf["user"] = request.form.get("email_user", "").strip()
                email_conf["password"] = request.form.get("email_password", "").strip()
                flash("Configuración de Email actualizada.", "success")
                
            elif action == "rebuild_index":
                # Logic to trigger rebuild via background task
                flash("Reindexado solicitado (no implementado en UI todavía).", "info")

            elif action == "llm_pipeline_config":
                llm_conf = config.setdefault("llm", {})
                llm_conf["enabled"] = "llm_enabled" in request.form
                llm_conf["base_url"] = request.form.get("llm_base_url", "").strip()
                llm_conf["model"] = request.form.get("llm_model", "").strip()
                llm_conf["api_key"] = request.form.get("llm_api_key", "").strip()
                llm_conf["timeout"] = int(request.form.get("llm_timeout", 60))
                flash("Configuración de Pipeline IA actualizada.", "success")

            elif action == "llm_chat_config":
                # Ensure routing section exists using nested setdefaults safely
                if "routing" not in config.get("llm", {}):
                    config.setdefault("llm", {})["routing"] = {}
                
                # Check for existing structure or initialize
                routing_conf = config["llm"]["routing"]
                chat_conf = routing_conf.setdefault("general_chat", {})
                
                chat_conf["base_url"] = request.form.get("chat_base_url", "").strip()
                chat_conf["model"] = request.form.get("chat_model", "").strip()
                flash("Configuración de Chat IA actualizada.", "success")
                
            save_configuration(config)
            return redirect(url_for("main.settings"))
            
        except Exception as e:
            get_logger().error(f"Error saving settings: {e}")
            flash(f"Error al guardar configuración: {e}", "error")

    post_conf = config.get("postbatch", {})
    hot_conf = config.get("hot_folder", {})
    email_conf = config.get("email_importer", {})
    vision_conf = config.get("vision", {})
    pipeline_conf = config.get("ocr_pipeline", {}) # Fixed key name
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
        
        # LLM Pipeline Settings
        "llm_enabled": llm_conf.get("enabled", False),
        "llm_base_url": llm_conf.get("base_url", "http://host.docker.internal:1234/v1"),
        "llm_model": llm_conf.get("model", "local-model"),
        "llm_api_key": llm_conf.get("api_key", ""),
        "llm_timeout": llm_conf.get("timeout", 60),

        # LLM Chat Settings
        "chat_base_url": chat_conf.get("base_url", "http://host.docker.internal:1234/v1"),
        "chat_model": chat_conf.get("model", "mistral-small-24b")
    }
    return render_template("settings.html", config=settings_data)

@main_bp.route("/batch_process", methods=["GET", "POST"])
def batch_process():
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
                flash("El procesamiento finalizó con errores.", "error")
        except Exception as exc:
            get_logger().error("Error running batch process: %s", exc)
            flash(f"Error en procesamiento: {exc}", "error")
        return redirect(url_for("main.dashboard"))

    return render_template("batch_process.html")


@main_bp.route("/view_document_file/<int:doc_id>")
def view_document_file(doc_id):
    db = get_db()
    path_str = db.get_document_path(doc_id)
    if not path_str:
        return "File not found", 404
    
    # Resolve to absolute path
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    
    # Handle version request
    version = request.args.get("version")
    if version == "original":
         backup_p = p.with_name(f"{p.stem}_original{p.suffix}")
         if backup_p.exists():
             p = backup_p
             
    try:
        if not p.exists():
            get_logger().error(f"File not found on disk: {p}")
            return "File not found on disk", 404
                
        return send_from_directory(p.parent, p.name)
    except Exception as e:
         get_logger().error(f"Error serving file: {e}")
         return "Error serving file", 500

@main_bp.route("/verify/<int:doc_id>")
def verify_document(doc_id):
    """Split-screen verification UI."""
    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        return "Document not found", 404
    
    if isinstance(doc.get("structured_data"), str):
        try:
             doc["structured_data"] = json.loads(doc["structured_data"])
        except:
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
def batch_action():
    action = request.form.get("action")
    doc_ids_str = request.form.get("doc_ids")
    
    if not action or not doc_ids_str:
        flash("Acción inválida", "error")
        return redirect(url_for("main.documents"))
        
    try:
        doc_ids = json.loads(doc_ids_str)
        doc_ids = [int(id) for id in doc_ids]
    except:
        flash("IDs inválidos", "error")
        return redirect(url_for("main.documents"))
        
    if not doc_ids:
        flash("Ningún documento seleccionado", "warning")
        return redirect(url_for("main.documents"))

    pipeline = get_pipeline()
    db = get_db()
    
    if action == "delete":
        success_count = 0
        for doc_id in doc_ids:
            # Use the robust delete_document method from DBManager
            if db.delete_document(doc_id):
                success_count += 1
        flash(f"{success_count} documentos eliminados", "success")
        
    elif action == "reprocess":
        from modules.tasks import huey, process_document_task
        count = 0
        for doc_id in doc_ids:
            row = db.execute(f"SELECT path FROM documents WHERE id = {db.placeholder}", (doc_id,)).fetchone()
            if row:
                path = PROJECT_ROOT / row[0]
                process_document_task(str(path), {
                    "delete_original": False,
                    "ocr_enabled": True,
                    "classification_enabled": True
                })
                count += 1
        flash(f"{count} documentos enviados a reprocesar", "info")
        
    elif action == "export":
        return export_documents(doc_ids)

    elif action == "auto_verify":
        success_count = 0
        skipped_count = 0
        for doc_id in doc_ids:
            row = db.execute(f"SELECT confidence FROM documents WHERE id = {db.placeholder}", (doc_id,)).fetchone()
            if row:
                conf = row[0]
                if conf is not None and conf >= 0.8:
                    db.update_document_status(doc_id, "OK", workflow_state="verified")
                    success_count += 1
                else:
                    skipped_count += 1
        
        msg = f"{success_count} documentos verificados automáticamente."
        if skipped_count > 0:
            msg += f" ({skipped_count} omitidos por baja confianza)"
        flash(msg, "success" if success_count > 0 else "warning")
        
    return redirect(url_for("main.documents"))

def export_documents(doc_ids):
    import csv 
    import io
    from flask import make_response
    
    db = get_db()
    placeholders = ",".join([db.placeholder] * len(doc_ids))
    query = f"SELECT d.id, d.filename, d.classification, d.status, d.confidence, d.datetime FROM documents d WHERE d.id IN ({placeholders})"
    
    rows = db.execute(query, tuple(doc_ids)).fetchall()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Archivo", "Clasificación", "Estado", "Confianza", "Fecha"])
    for row in rows:
        cw.writerow(row)
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@main_bp.route("/tasks")
def tasks_page():
    return render_template("tasks.html")

@main_bp.route("/gallery")
def gallery():
    return render_template("gallery.html")

@main_bp.route("/download/table/<int:doc_id>/<int:index>/<fmt>")
def download_table(doc_id, index, fmt):
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute("SELECT tables_json FROM ocr_texts WHERE id_doc = %s", (doc_id,)) # %s for psycopg2 or ? for sqlite. DBManager handles? 
        # DBManager usually handles placeholder.
        # But wait, db.execute uses the placeholder.
        # Here I am using cursor directly.
        pass
    
    # Better use db.execute helper if available or standard query
    # Using db.execute which returns a cursor.
    row = db.execute(f"SELECT tables_json FROM ocr_texts WHERE id_doc = {db.placeholder}", (doc_id,)).fetchone()
    
    if not row or not row[0]:
        return "Table not found", 404
        
    tables = json.loads(row[0])
    if index < 0 or index >= len(tables):
        return "Index out of bounds", 404
        
    table = tables[index]
    
    if fmt == "csv":
        path = table.get("csv_path")
        mime = "text/csv"
        ext = "csv"
    elif fmt == "json":
        path = table.get("json_path")
        mime = "application/json"
        ext = "json"
    else:
        return "Invalid format", 400
        
    if not path:
        return "File path missing in DB", 404
        
    abs_path = PROJECT_ROOT / path
    if not abs_path.exists():
        return "File not found on disk", 404
        
    return send_from_directory(abs_path.parent, abs_path.name, as_attachment=True, download_name=f"table_{doc_id}_{index}.{ext}")

@main_bp.route("/image_search", methods=["POST"])
def image_search():
    if "image" not in request.files:
        flash("No se subió imagen", "error")
        return redirect(url_for("main.dashboard"))
        
    file = request.files["image"]
    if file.filename == "":
        flash("Nombre de archivo vacío", "error")
        return redirect(url_for("main.dashboard"))

    try:
        # Use temp file
        fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix)
        os.close(fd)
        file.save(temp_path)
        
        pipeline = get_pipeline()
        vision = pipeline.vision_manager
        
        if not vision or not vision.config.enabled:
            os.remove(temp_path)
            flash("Visión no habilitada", "error")
            return redirect(url_for("main.dashboard"))
            
        results = vision.search_similar(temp_path, k=5)
        os.remove(temp_path)
        
        # Enrich results
        db = get_db()
        enriched = []
        for res in results:
            path = res["path"]
            row = db.execute(f"SELECT id, filename, tags, datetime FROM documents WHERE path = {db.placeholder}", (path,)).fetchone()
            if row:
                enriched.append({
                    "id": row[0],
                    "filename": row[1],
                    "path": path,
                    "score": res["score"],
                    "tags": json.loads(row[2]) if row[2] else [],
                    "preview_url": f"/vision/preview/{encode_path(path)}" 
                })
        
        session["image_results"] = enriched
        return redirect(url_for("main.dashboard") + "#vision-pane")
        
    except Exception as e:
        get_logger().error(f"Image search error: {e}")
        session["image_error"] = str(e)
        return redirect(url_for("main.dashboard"))
