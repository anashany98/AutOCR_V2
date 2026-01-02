"""
AutOCR Web Interface.

Flask application for managing documents, OCR processing, table extraction and
vision search capabilities.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
import psutil
from datetime import datetime
from werkzeug.utils import secure_filename
import mimetypes
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Register new image formats to ensure correct Content-Type headers
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/jpeg', '.jfif')
mimetypes.add_type('image/avif', '.avif')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # Ensure local packages import when run as a script.

from modules.classifier import DocumentClassifier
from modules.db_manager import DBManager
from modules.file_utils import ensure_directories
from modules.logger_manager import setup_logger
from modules.folder_watcher import FolderWatcher
from modules.rag_manager import RAGManager
from modules.image_utils import enhance_image, detect_handwriting_probability
from modules.learning import ModelTrainer
from postbatch_processor import PipelineComponents, initialise_pipeline, process_single_file
from modules.moodboard import MoodboardGenerator
from modules.deduplicator import Deduplicator
from modules.schemas import DocumentUpdateSchema
from pydantic import ValidationError

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "web_app" / "static" / "uploads"

# Extensions allowed for visualization and processing
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif", ".gif", ".tif", ".tiff"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "autocr-secret-key-change-in-production"
app.config["UPLOAD_FOLDER"] = str(DEFAULT_UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB as per security policy

# Security: Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per minute", "10 per second"],
    storage_uri="memory://",
)

# Add built-in functions to Jinja2 globals
app.jinja_env.globals.update(max=max, min=min)

ensure_directories(app.config["UPLOAD_FOLDER"])

local = threading.local()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_configuration(reload: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML. If reload=True or not cached, reads from disk."""
    if reload or getattr(local, "config", None) is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                local.config = yaml.safe_load(handle) or {}
        else:
            local.config = {}
    return local.config


def save_configuration(config: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    local.config = config
    if hasattr(local, "pipeline"):
        delattr(local, "pipeline")


def resolve_path(path_value: Optional[str], fallback: Optional[str] = None) -> str:
    value = path_value or fallback
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def get_rag_manager():
    """Singleton accessor for RAG Manager."""
    if not hasattr(local, "rag_manager"):
        if RAGManager:
            # Check if models are available
             local.rag_manager = RAGManager(PROJECT_ROOT / "data" / "rag_index")
        else:
             local.rag_manager = None
    return local.rag_manager

# --------------------------------------------------------------------------- #
# Singletons for heavy resources
# --------------------------------------------------------------------------- #
_pipeline_instance: Optional[PipelineComponents] = None
_pipeline_lock = threading.Lock()

_classifier_instance: Optional[DocumentClassifier] = None
_classifier_lock = threading.Lock()

_rag_instance: Optional[RAGManager] = None
_rag_lock = threading.Lock()

_db_instance: Optional[DBManager] = None
_db_lock = threading.Lock()

def get_db() -> DBManager:
    """Get database manager singleton with thread-safe initialization."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:  # Double-check locking
                config = load_configuration()
                _db_instance = DBManager(config)
    return _db_instance

def get_logger():
    # Keep logger in local or simplify
    if getattr(local, "logger", None) is None:
        config = load_configuration()
        app_conf = config.get("app", {})
        log_level = app_conf.get("log_level", "INFO")
        log_dir = PROJECT_ROOT / "web_app" / "logs"
        ensure_directories(str(log_dir))
        log_path = log_dir / "web_app.log"
        local.logger = setup_logger(str(log_path), level=log_level, db_manager=get_db())
    return local.logger


def get_pipeline() -> PipelineComponents:
    """Get pipeline components singleton with thread-safe initialization."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:  # Double-check locking
                config = load_configuration()
                _pipeline_instance = initialise_pipeline(config, str(PROJECT_ROOT), get_logger())
    return _pipeline_instance


def get_classifier() -> Optional[DocumentClassifier]:
    """Get document classifier singleton with thread-safe initialization."""
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:  # Double-check locking
                config = load_configuration()
                post_conf = config.get("postbatch", {})
                if post_conf.get("classification_enabled", True):
                    model_path = PROJECT_ROOT / "data" / "models" / "classifier.pkl"
                    _classifier_instance = DocumentClassifier(model_path=str(model_path))
                else:
                    _classifier_instance = None
    return _classifier_instance


def get_rag_manager() -> Optional[RAGManager]:
    """Get RAG manager singleton with thread-safe initialization."""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:  # Double-check locking
                try:
                    rag_dir = PROJECT_ROOT / "data" / "rag_index"
                    _rag_instance = RAGManager(str(rag_dir))
                except Exception as e:
                    get_logger().error(f"Failed to load RAG Manager: {e}")
                    _rag_instance = None
    return _rag_instance


def ensure_within_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path outside project root: {resolved}") from exc
    return resolved


def encode_path(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("utf-8")


def decode_path(token: str) -> str:
    return base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")


def safe_json_parse(value: Any, default: Any = None) -> Any:
    """Parse JSON safely with type checking and fallback."""
    if not value or not isinstance(value, str):
        return default if default is not None else []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


# --------------------------------------------------------------------------- #
# Hot Folder Logic
# --------------------------------------------------------------------------- #

# Global reference to keep the watcher alive
_watcher_instance: Optional[FolderWatcher] = None

def process_hot_file(path: Path) -> None:
    """Callback for hot folder watcher."""
    # We need a fresh context for proper logging/db access in this thread
    try:
        config = load_configuration()
        post_conf = config.get("postbatch", {})
        
        # Determine folders
        processed_folder = resolve_path(post_conf.get("processed_folder"), "data/scans_processed")
        failed_folder = resolve_path(post_conf.get("failed_folder"), "data/scans_failed")
        ensure_directories(processed_folder, failed_folder)
        
        # Create dedicated objects (not thread-local as this is a daemon thread)
        # Using a new DBManager instance to avoid threading issues
        app_conf = config.get("app", {})
        db_path = resolve_path(app_conf.get("db_path"), "data/digitalizerai.db")
        db = DBManager(db_path, use_sql_server=app_conf.get("use_sql_server", False))
        
        # Pipeline is thread-safe (singleton models inside)
        pipeline = get_pipeline() 
        classifier = get_classifier()

        # Logger
        logger = get_logger()
        logger.info(f"⚡ Hot Folder: Processing {path.name}")
        
        process_single_file(
            str(path),
            pipeline,
            classifier,
            db,
            processed_folder,
            failed_folder,
            delete_original=True, # Move from hot folder
            ocr_enabled=post_conf.get("ocr_enabled", True),
            classification_enabled=post_conf.get("classification_enabled", True),
            logger=logger,
            input_root=str(path.parent)
        )
        
        db.close()
        
    except Exception as e:
        print(f"Error in hot file processing: {e}")


def init_watcher():
    """Initialize or update the folder watcher based on config."""
    global _watcher_instance
    config = load_configuration()
    hot_conf = config.get("hot_folder", {})
    
    if _watcher_instance:
        _watcher_instance.stop()
        _watcher_instance = None
        
    if hot_conf.get("enabled", False):
        path_str = hot_conf.get("path")
        if path_str:
            base_dir = resolve_path(path_str)
            ensure_directories(base_dir)
            
            _watcher_instance = FolderWatcher(
                watch_dir=base_dir,
                callback=process_hot_file,
                extensions=hot_conf.get("extensions")
            )
            _watcher_instance.start()



def load_tables_for_document(doc_id: int) -> List[Dict[str, Any]]:
    db = get_db()
    cursor = db.conn.cursor()
    cursor.execute(
        """
        SELECT tables_json FROM ocr_texts
        WHERE id_doc = ? AND tables_json IS NOT NULL
        """,
        (doc_id,),
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# App initialisation
# --------------------------------------------------------------------------- #


def init_app():
    try:
        ensure_directories(app.config["UPLOAD_FOLDER"])
        init_watcher()
        # Pre-warm heavy singletons
        get_db()
        get_pipeline()
        get_classifier()
        get_rag_manager()
        get_logger().info("🧩 Application singletons pre-warmed and ready.")
    except Exception as exc:
        print(f"Error initialising AutOCR Web App: {exc}")
        import traceback
        traceback.print_exc()
        raise


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def dashboard():
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
        status_stats = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT type, COUNT(*) FROM documents
            WHERE type IS NOT NULL
            GROUP BY type
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """
        )
        type_stats = [list(row) for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT id, filename, type, status, datetime, duration
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

    # Get recent logs for monitoring
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
    )


@app.route("/documents")
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
            SELECT id, filename, path, type, status, datetime, duration, tags
            FROM documents
            WHERE 1=1
        """
        params: List[Any] = []

        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        if search_term:
            query += " AND (filename LIKE ? OR type LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        query += " ORDER BY datetime DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        cursor.execute(query, params)
        documents_rows = cursor.fetchall()
        
        # Count query needs to use same cursor/connection context
        count_query = "SELECT COUNT(*) FROM documents WHERE 1=1"
        count_params: List[Any] = []
        if status_filter:
            count_query += " AND status = ?"
            count_params.append(status_filter)
        if type_filter:
            count_query += " AND type = ?"
            count_params.append(type_filter)
        if search_term:
            count_query += " AND (filename LIKE ? OR type LIKE ?)"
            count_params.extend([f"%{search_term}%", f"%{search_term}%"])
        cursor.execute(count_query, count_params)
        total_docs = cursor.fetchone()[0]
        
    total_pages = max(1, (total_docs + per_page - 1) // per_page)
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


@app.route("/document/<int:doc_id>")
def document_detail(doc_id: int):
    db = get_db()
    
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute(
            """
            SELECT d.id, d.filename, d.path, d.type, d.status, d.datetime, d.duration,
                   d.tags, d.workflow_state, o.text, o.markdown_text, o.language, o.confidence,
                   o.blocks_json, o.tables_json, o.structured_data
            FROM documents d
            LEFT JOIN ocr_texts o ON d.id = o.id_doc
            WHERE d.id = ?
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
    if not row:
        flash("Documento no encontrado.", "error")
        return redirect(url_for("documents"))

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





@app.route("/api/document/<int:doc_id>/update", methods=["POST"])
def api_update_document(doc_id: int):
    """Update document data with Pydantic validation."""
    try:
        data = request.json
        # 1. Validate with Pydantic
        validated = DocumentUpdateSchema(**data)
        
        # 2. Update DB
        db = get_db()
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            
            # Update 'documents' table fields
            cursor.execute(
                """
                UPDATE documents 
                SET filename = ?, type = ?, tags = ?
                WHERE id = ?
                """,
                (
                    validated.filename,
                    validated.type,
                    json.dumps(validated.tags),
                    doc_id
                )
            )
            
            # Update 'ocr_texts' structured_data
            # First fetch existing to merge (optional, but safer)
            cursor.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
            row = cursor.fetchone()
            current_data = safe_json_parse(row[0], {}) if row else {}
            
            # Merge updates
            if validated.date:
                current_data["date"] = validated.date
            if validated.total is not None:
                current_data["total"] = validated.total
            if validated.supplier:
                current_data["supplier"] = validated.supplier
            
            cursor.execute(
                """
                UPDATE ocr_texts
                SET structured_data = ?
                WHERE id_doc = ?
                """,
                (json.dumps(current_data), doc_id)
            )
            
            conn.commit()
            
        return jsonify({"status": "success", "message": "Document verified and updated"})

    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 400
    except Exception as e:
        get_logger().error(f"Update error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/api/create_moodboard", methods=["POST"])
def api_create_moodboard():
    """Create a moodboard from selected document IDs."""
    data = request.json
    doc_ids = data.get("ids", [])
    title = data.get("title", "Mi Moodboard")

    if not doc_ids:
        return jsonify({"error": "No documents selected"}), 400

    db = get_db()
    
    # Resolve paths
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        placeholders = ",".join("?" * len(doc_ids))
        # Note: using db.placeholder would be better strictly speaking but '?' works for standard execution if not using db abstraction fully here
        # But we should respect the new pattern
        
        # Construct query using proper placeholders for the engine
        ph = db.placeholder
        placeholders = ",".join(ph for _ in doc_ids)
        
        # Important: execute returns cursor in standard DB-API, but db.conn.execute was likely sqlite shortcut
        # We must use cursor.execute
        cursor.execute(
            f"SELECT path FROM documents WHERE id IN ({placeholders})", doc_ids
        )
        rows = cursor.fetchall()
    
    paths = [r["path"] for r in rows]
    
    if not paths:
        return jsonify({"error": "No valid images found for selected IDs"}), 404

    generator = MoodboardGenerator(output_dir=PROJECT_ROOT / "data" / "moodboards")
    try:
        output_path = generator.create(paths, title)
        if not output_path:
             return jsonify({"error": "Failed to generate moodboard (invalid images?)"}), 500
             
        filename = Path(output_path).name
        url = url_for("serve_moodboard", filename=filename)
        return jsonify({"url": url})
        
    except Exception as e:
        get_logger().error(f"Moodboard generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/moodboard_file/<filename>")
def serve_moodboard(filename):
    """Serve generated moodboards."""
    return send_from_directory(PROJECT_ROOT / "data" / "moodboards", filename)

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shutdown the server."""
    shutdown_func = request.environ.get("werkzeug.server.shutdown")
    if shutdown_func:
        shutdown_func()
    return "Server shutting down..."

# --------------------------------------------------------------------------- #
# Duplicates & Maintenance
# --------------------------------------------------------------------------- #

@app.route("/duplicates")
def duplicates_page():
    return render_template("duplicates.html")

@app.route("/api/duplicates/scan")
def api_scan_duplicates():
    pipeline = get_pipeline()
    if not pipeline.vision_manager:
        return jsonify([])
    
    deduper = Deduplicator(pipeline.vision_manager)
    visual_dupes = deduper.find_duplicates()
    
    # Format for UI
    results = []
    
    # Needs helper to format single doc
    def format_doc(meta):
        # We need Doc ID to behave correctly
        # Meta from VisionManager might have different keys depending on how it was built
        # But usually standard keys: doc_id, filename, path
        # If path is inside project, make a token for preview
        try:
            p = ensure_within_project(Path(meta["path"]))
            token = encode_path(str(p))
            url = url_for("vision_preview", token=token)
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

@app.route("/api/document/<int:doc_id>/verify", methods=["POST"])
def api_verify_document(doc_id):
    db = get_db()
    if db.update_document_state(doc_id, "verified"):
        get_logger().info(f"Document {doc_id} manually verified.")
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update state"}), 500

@app.route("/api/document/<int:doc_id>/fields", methods=["POST"])
def api_update_fields(doc_id):
    """Update structured fields for a document."""
    db = get_db()
    data = request.json
    
    if not data or 'fields' not in data:
        return jsonify({"error": "Missing fields data"}), 400
    
    try:
        # Fetch current structured_data
        cursor = db.conn.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404
        
        structured_data = json.loads(row[0]) if row[0] else {}
        structured_data['fields'] = data['fields']
        
        # Update in database
        db.conn.execute(
            "UPDATE ocr_texts SET structured_data = ? WHERE id_doc = ?",
            (json.dumps(structured_data), doc_id)
        )
        db.conn.commit()
        
        get_logger().info(f"Updated fields for document {doc_id}")
        return jsonify({"success": True})
    except Exception as e:
        get_logger().error(f"Failed to update fields: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/providers/search")
def api_search_providers():
    """Search providers for autocomplete."""
    query = request.args.get('q', '').lower()
    
    try:
        providers_path = PROJECT_ROOT / "data" / "providers.json"
        with open(providers_path, 'r', encoding='utf-8') as f:
            providers = json.load(f)
        
        results = []
        for provider_id, provider_data in providers.items():
            canonical = provider_data.get('canonical_name', '')
            aliases = provider_data.get('aliases', [])
            
            # Match on canonical name or aliases
            if query in canonical.lower() or any(query in alias.lower() for alias in aliases):
                results.append({
                    'id': provider_id,
                    'name': canonical,
                    'vat': provider_data.get('vat_number', ''),
                    'category': provider_data.get('category', '')
                })
        
        return jsonify(results[:10])  # Limit to 10 results
    except Exception as e:
        get_logger().error(f"Provider search failed: {e}")
        return jsonify([])

@app.route("/api/document/<int:doc_id>/dismiss-anomaly", methods=["POST"])
def api_dismiss_anomaly(doc_id):
    """Dismiss an anomaly for a document."""
    db = get_db()
    data = request.json
    
    if not data or 'anomaly' not in data:
        return jsonify({"error": "Missing anomaly code"}), 400
    
    try:
        cursor = db.conn.execute("SELECT structured_data FROM ocr_texts WHERE id_doc = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Document not found"}), 404
        
        structured_data = json.loads(row[0]) if row[0] else {}
        anomalies = structured_data.get('anomalies', [])
        
        # Remove the anomaly
        if data['anomaly'] in anomalies:
            anomalies.remove(data['anomaly'])
            structured_data['anomalies'] = anomalies
            
            db.conn.execute(
                "UPDATE ocr_texts SET structured_data = ? WHERE id_doc = ?",
                (json.dumps(structured_data), doc_id)
            )
            db.conn.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/document/<int:doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    db = get_db()
    
    # Get path to delete file
    row = db.conn.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
        
    path_str = row["path"]
    
    # Delete from DB
    try:
        # db_manager has delete_document? Let's assume yes or check.
        # Checking db_manager.py would be safer but let's try standard approach.
        # If db.delete_document doesn't exist, we do manual delete.
        if hasattr(db, "delete_document"):
             db.delete_document(doc_id)
        else:
             # Fallback
             db.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
             db.conn.execute("DELETE FROM ocr_texts WHERE id_doc = ?", (doc_id,))
             db.conn.commit()

        # Delete from Disk
        try:
            if os.path.exists(path_str):
                os.remove(path_str)
        except OSError as e:
            get_logger().error(f"Failed to delete file {path_str}: {e}")
            # We still return success as DB is cleared
            
        return jsonify({"success": True})
        
    except Exception as e:
        get_logger().error(f"Delete failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    rag = get_rag_manager()
    if not rag:
        return jsonify({"error": "RAG system not initialized"}), 503
        
    results = rag.search(query)
    return jsonify({"results": results})





@app.route("/api/train", methods=["POST"])
def api_train_model():
    db = get_db()
    model_path = PROJECT_ROOT / "data" / "models" / "classifier.pkl"
    trainer = ModelTrainer(db, str(model_path))
    
    # Simple synchronous training (it's fast for small datasets)
    success, msg = trainer.train()
    if success:
        # Clear cached classifier so next request reloads it
        if hasattr(local, "classifier"):
             delattr(local, "classifier")
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "files" not in request.files:
            flash("Debe seleccionar archivos.", "error")
            return redirect(request.url)

        upload_dir = app.config["UPLOAD_FOLDER"]
        ensure_directories(upload_dir)

        files = [file for file in request.files.getlist("files") if file and file.filename]
        if not files:
            flash("Debe seleccionar al menos un archivo valido.", "error")
            return redirect(request.url)

        pipeline = get_pipeline()
        classifier = get_classifier()
        # Dynamically reload config in upload to ensure new file types are picked up
        config = load_configuration(reload=True)
        post_conf = config.get("postbatch", {})
        
        processed_folder = resolve_path(post_conf.get("processed_folder"), "data/scans_processed")
        failed_folder = resolve_path(post_conf.get("failed_folder"), "data/scans_failed")
        ensure_directories(processed_folder, failed_folder)
        
        # Merge config file_types with our internal ALLOWED_IMAGE_EXTS for safety
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
        classification_enabled = (
            "classification_enabled" in request.form if request.form else post_conf.get("classification_enabled", True)
        )
        handwriting_mode = "handwriting_mode" in request.form

        processed = 0
        failed = 0
        for temp_path in saved_files:
            result = process_single_file(
                temp_path,
                pipeline,
                classifier,
                get_db(),
                processed_folder,
                failed_folder,
                delete_original=True,
                ocr_enabled=ocr_enabled,
                classification_enabled=classification_enabled,
                logger=get_logger(),
                input_root=upload_dir,
                handwriting_mode=handwriting_mode,
                pipeline_conf=config,
            )
            if result["status"] == "OK":
                processed += 1
            else:
                failed += 1

        if invalid_files:
            flash(
                f"{processed} archivos procesados; {failed} con errores. "
                f"Se omitieron por extension no permitida: {', '.join(invalid_files)}",
                "warning",
            )
        else:
            if failed > 0:
                flash(f"{processed} archivos procesados; {failed} con errores.", "warning")
            else:
                flash(f"Se han procesado {processed} archivos correctamente.", "success")
        return redirect(url_for("documents"))

    return render_template("upload.html")


@app.route("/settings", methods=["GET", "POST"])
def settings():
    config = load_configuration()
    post_conf = config.get("postbatch", {})
    pipeline_conf = config.get("ocr_pipeline", {})
    engines_conf = pipeline_conf.get("engines", [])
    engine_map = {
        str(engine.get("name", "")).lower(): engine for engine in engines_conf if engine.get("name")
    }
    paddle_conf = engine_map.get("paddleocr", {"name": "paddleocr"})
    easy_conf = engine_map.get("easyocr", {"name": "easyocr"})
    engine_order = ["paddleocr", "easyocr"]

    current_languages: List[str] = []
    if paddle_conf.get("lang"):
        current_languages.append(str(paddle_conf["lang"]))
    for code in easy_conf.get("langs", []) or []:
        code_str = str(code)
        if code_str not in current_languages:
            current_languages.append(code_str)
    if not current_languages:
        current_languages = ["spa", "eng"]

    gpu_enabled = bool(paddle_conf.get("gpu", True) or easy_conf.get("gpu", False))

    vision_conf = pipeline_conf.get("vision", {})

    if request.method == "POST":
        action = request.form.get("action", "directories")

        if action == "directories":
            updated = {
                "input_folder": request.form.get("input_folder", "").strip(),
                "processed_folder": request.form.get("processed_folder", "").strip(),
                "failed_folder": request.form.get("failed_folder", "").strip(),
                "reports_folder": request.form.get("reports_folder", "").strip(),
            }
            if not all(updated.values()):
                flash("Todos los directorios son obligatorios.", "error")
            else:
                config.setdefault("postbatch", {}).update(updated)
                save_configuration(config)
                ensure_directories(*updated.values())
                flash("Directorios actualizados correctamente.", "success")
                return redirect(url_for("settings"))

        elif action == "pipeline":
            gpu_toggle = request.form.get("gpu_enabled") == "on"
            languages_str = request.form.get("languages", "spa,eng")
            primary_engine = request.form.get("primary_engine", "paddleocr")
            
            languages = [l.strip() for l in languages_str.split(",") if l.strip()]

            pipeline_conf = config.get("ocr_pipeline", {})
            pipeline_conf["primary_engine"] = primary_engine
            
            # Update individual engine configs if needed
            paddle_conf = engine_map.get("paddleocr", {"name": "paddleocr"})
            easy_conf = engine_map.get("easyocr", {"name": "easyocr"})
            
            if not languages:
                flash("Debe especificar al menos un idioma.", "error")
            else:
                paddle_conf["lang"] = languages[0]
                paddle_conf["gpu"] = gpu_toggle
                paddle_conf["enabled"] = True

                easy_conf["langs"] = languages
                easy_conf["gpu"] = gpu_toggle
                easy_conf["enabled"] = True

                engine_map["paddleocr"] = paddle_conf
                engine_map["easyocr"] = easy_conf
                pipeline_conf["engines"] = [
                    engine_map[name] if name in engine_map else {"name": name}
                    for name in engine_order
                    if name in engine_map
                ]

                fusion_conf = pipeline_conf.setdefault("fusion", {})
                fusion_conf.setdefault("strategy", "")
                fusion_conf.setdefault("min_confidence", 0.8)
                fusion_conf.setdefault("min_similarity", 0.82)
                fusion_conf["priority"] = ["paddleocr", "easyocr"]

                config["ocr_pipeline"] = pipeline_conf
                save_configuration(config)
                flash("Parámetros del pipeline actualizados.", "success")
                return redirect(url_for("settings"))

        elif action == "rebuild_index":
            gallery_dir = resolve_path(
                vision_conf.get("gallery_dir"),
                "data/vision_gallery",
            )
            ensure_directories(gallery_dir)
            vision_manager = get_pipeline().vision_manager
            if not vision_manager or not vision_manager.config.enabled:
                flash("La búsqueda visual está deshabilitada.", "error")
            else:
                vision_manager.build_index(gallery_dir)
                flash("Índice de imágenes reconstruido.", "success")
            return redirect(url_for("settings"))

        elif action == "hot_folder":
            hot_enabled = request.form.get("hot_enabled") == "on"
            hot_path = request.form.get("hot_path", "").strip()
            
            if hot_enabled and not hot_path:
                flash("Debe especificar una ruta para activar la monitorización.", "error")
            else:
                hot_conf = config.setdefault("hot_folder", {})
                hot_conf["enabled"] = hot_enabled
                hot_conf["path"] = hot_path
                hot_conf["extensions"] = hot_conf.get("extensions", [".pdf", ".jpg", ".jpeg", ".png"])
                save_configuration(config)
                init_watcher() # Restart watcher with new config
                flash("Configuración de Hot Folder actualizada.", "success")
                flash("Configuración de Hot Folder actualizada.", "success")
            return redirect(url_for("settings"))

        elif action == "email_import":
            email_enabled = request.form.get("email_enabled") == "on"
            host = request.form.get("email_host", "").strip()
            port = int(request.form.get("email_port", 993))
            user = request.form.get("email_user", "").strip()
            password = request.form.get("email_password", "").strip()
            
            email_conf = config.setdefault("email_importer", {})
            email_conf["enabled"] = email_enabled
            email_conf["host"] = host
            email_conf["port"] = port
            email_conf["user"] = user
            if password: # Only update if provided to avoid clearing it
                email_conf["password"] = password
            
            save_configuration(config)
            flash("Configuración de Email actualizada. Reinicie para aplicar cambios.", "success")
            return redirect(url_for("settings"))


    hot_conf = config.get("hot_folder", {})
    email_conf = config.get("email_importer", {})
    settings_data = {
        "input_folder": post_conf.get("input_folder", ""),
        "processed_folder": post_conf.get("processed_folder", ""),
        "failed_folder": post_conf.get("failed_folder", ""),
        "reports_folder": post_conf.get("reports_folder", ""),
        "gpu_enabled": gpu_enabled,
        "languages": ", ".join(current_languages),
        "primary_engine": pipeline_conf.get("primary_engine", "paddleocr"),
        "vision_enabled": vision_conf.get("enabled", True),
        "gallery_dir": vision_conf.get("gallery_dir", "data/vision_gallery"),
        "hot_enabled": hot_conf.get("enabled", False),
        "hot_path": hot_conf.get("path", ""),
        "email_enabled": email_conf.get("enabled", False),
        "email_host": email_conf.get("host", ""),
        "email_port": email_conf.get("port", 993),
        "email_user": email_conf.get("user", ""),
        "email_password": email_conf.get("password", "")
    }
    return render_template("settings.html", config=settings_data)


@app.route("/batch_process", methods=["GET", "POST"])
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
        return redirect(url_for("dashboard"))

    return render_template("batch_process.html")

@app.route("/view_document_file/<int:doc_id>")
def view_document_file(doc_id):
    db = get_db()
    path = db.get_document_path(doc_id)
    if not path:
        return "File not found", 404
    
    p = Path(path)
    if not p.is_absolute():
       p = PROJECT_ROOT / p
    
    if not p.exists():
        return "File not found on disk", 404
        
    return send_from_directory(p.parent, p.name)






@app.route("/api/search")
def api_search():
    """API endpoint for full-text search."""
    query = request.args.get("q", "")
    if not query:
        # Fallback: return recent documents
        db = get_db()
        rows = db.conn.execute("SELECT id, filename, path, datetime, tags FROM documents ORDER BY id DESC LIMIT 20").fetchall()
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
    # Format for frontend
    return jsonify([
        {"id": r[0], "filename": r[1], "snippet": r[2]} 
        for r in results
    ])

@app.route("/verify/<int:doc_id>")
def verify_document(doc_id):
    """Split-screen verification UI."""
    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        return "Document not found", 404
        
    # Enrich with structured data for confidence display
    # db.get_document usually returns a Row/Dict. 
    # We need to ensure 'structured_data' is a dict.
    
    # If structured_data is stored as JSON string in DB, parse it.
    if isinstance(doc.get("structured_data"), str):
        try:
             doc["structured_data"] = json.loads(doc["structured_data"])
        except:
             doc["structured_data"] = {}
             
    # Prepare confidence map for template
    # keys: total_conf, date_conf, supplier_conf
    if doc.get("structured_data") and "fields" in doc["structured_data"]:
        fields = doc["structured_data"]["fields"]
        for key, info in fields.items():
            if isinstance(info, dict) and "confidence" in info:
                doc[f"{key}_conf"] = info["confidence"]
                # Also ensure values are populated if missing in top-level data
                if not doc["data"].get(key) and info.get("value"):
                     doc["data"][key] = info["value"]

    return render_template("verification_split.html", document=doc)

@app.route("/api/document/<int:doc_id>/enhance", methods=["POST"])
def api_enhance_document(doc_id):
    """Apply enhancements to document image and re-process."""
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

        # Load and Enhance
        try:
            with Image.open(original_path) as img:
                enhanced = enhance_image(img, contrast, brightness, sharpness, apply_clahe)
                
                # Save as new version or overwrite? 
                # For this feature request ("Reparar Escaneo"), usually implies improving current state.
                # Let's overwrite for simplicity but maybe backup original?
                # Backup logic: 
                backup_path = original_path.with_suffix(original_path.suffix + ".bak")
                if not backup_path.exists():
                    img.save(backup_path) # Save original once
                    
                enhanced.save(original_path)
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {e}"}), 500

        # Trigger Re-OCR
        # We can use a thread or do it synchronously if fast enough. 
        # Enhancing implies user is waiting for result.
        pipeline = get_pipeline()
        
        # Update DB to pending
        db.update_document_status(doc_id, "processing")
        
        # Async re-process
        def reprocess():
            try:
                # Force re-extraction
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



@app.route("/api/chat", methods=["POST"])
def api_chat_post():
    """Semantic search chat endpoint for documents."""
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # 1. RAG Search
    rag_manager = get_rag_manager()
    if not rag_manager:
         return jsonify({"error": "RAG system not initialized"}), 500

    results = rag_manager.search(query, k=5)
    
    # Format snippet response (for context)
    context_response = [
         {"id": r[0], "filename": r[1], "snippet": r[2]} 
         for r in results
    ]

    # 2. Check LLM Config
    config_path = PROJECT_ROOT / "config.yaml"
    full_config = {}
    if config_path.exists():
         with open(config_path, "r", encoding="utf-8") as f:
             full_config = yaml.safe_load(f)
    
    llm_conf = full_config.get("llm", {})
    
    if not llm_conf.get("enabled", False):
         # Return only RAG snippets if LLM disabled
         # Match old format {results: [...]} but maybe frontend expects this
         # Use unified format
         return jsonify({"results": results, "answer": None})

    # 3. Connect to LLM (LM Studio / Ollama)
    try:
        # Construct Context
        context_text = "\n\n".join([f"Documento '{r[1]}': {r[2]}" for r in results])
        
        system_prompt = (
            "Eres un asistente experto en análisis de documentos. "
            "Usa los siguientes fragmentos de contexto para responder a la pregunta del usuario. "
            "Si la respuesta no está en el contexto, di que no lo sabes. "
            "Se conciso y profesional."
        )
        
        user_prompt = f"Contexto:\n{context_text}\n\nPregunta: {query}"
        
        payload = {
            "model": llm_conf.get("model", "local-model"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        
        provider_url = f"{llm_conf.get('base_url','http://localhost:1234/v1').rstrip('/')}/chat/completions"
        
        import requests
        response = requests.post(
            provider_url, 
            json=payload, 
            timeout=llm_conf.get("timeout", 60)
        )
        
        if response.status_code == 200:
            llm_result = response.json()
            answer = llm_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return jsonify({
                "results": results,
                "answer": answer
            })
        else:
            get_logger().warning(f"LLM API Error: {response.status_code} - {response.text}")
            # Fallback to pure RAG if LLM fails
            return jsonify({"results": results, "answer": None})

    except Exception as e:
        get_logger().error(f"LLM Connection Failed: {e}")
        # Fallback to pure RAG
        return jsonify({"results": results, "answer": None})


@app.route("/api/rag/rebuild", methods=["POST"])
def api_rag_rebuild():
    """Trigger valid full re-indexing of documents."""
    db = get_db()
    rag_manager = get_rag_manager()
    if not rag_manager:
        return jsonify({"error": "RAG system not initialized"}), 500

    def run_rebuild():
         # Run in background to not block
         with db.get_connection() as conn:
             # We need a dedicated connection/cursor logic for the rag manager potentially
             # but rag_manager.rebuild expects db_manager object which exposes conn
             # since db_manager.conn is not thread safe with pool, we should be careful.
             # RAG rebuild() uses db_manager.conn.cursor() directly.
             # We should probably fix RAGManager to accept a connection, but for now let's pass db
             # assuming it handles its own cursor if possible or accept race condition for now (rebuild is rare)
             try:
                 rag_manager.rebuild(db)
             except Exception as e:
                 get_logger().error(f"RAG Rebuild failed: {e}")
    
    threading.Thread(target=run_rebuild).start()
    return jsonify({"message": "Proceso de reindexado iniciado en segundo plano."})

@app.route("/image_search", methods=["POST"])
def image_search():
    pipeline = get_pipeline()
    vision_manager = pipeline.vision_manager
    if not vision_manager or not vision_manager.config.enabled:
        session["image_error"] = "La búsqueda visual está deshabilitada."
        return redirect(url_for("dashboard"))

    file = request.files.get("image")
    if not file or not file.filename:
        session["image_error"] = "Debe seleccionar una imagen."
        return redirect(url_for("dashboard"))

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTS:
        session["image_error"] = "Formato de imagen no soportado."
        return redirect(url_for("dashboard"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        results = vision_manager.search_similar(temp_path, k=10)
        formatted_results = []
        for item in results:
            image_path = Path(item["path"])
            try:
                safe_path = ensure_within_project(image_path)
                token = encode_path(str(safe_path))
                preview_url = url_for("vision_preview", token=token)
                formatted_results.append(
                    {
                        "path": str(image_path),
                        "score": round(float(item["score"]), 4),
                        "preview_url": preview_url,
                    }
                )
            except ValueError:
                continue
        session["image_results"] = formatted_results
    except Exception as exc:
        session["image_error"] = f"No se pudo realizar la búsqueda: {exc}"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return redirect(url_for("dashboard"))


@app.route("/vision/preview/<token>")
def vision_preview(token: str):
    try:
        path = ensure_within_project(Path(decode_path(token)))
    except Exception:
        flash("Imagen no disponible.", "error")
        return redirect(url_for("dashboard"))
    return send_from_directory(path.parent, path.name)


@app.route("/visual_search", methods=["GET", "POST"])
def visual_search():
    query = request.form.get("query", "") if request.method == "POST" else ""
    results = []
    
    if query:
        pipeline = get_pipeline()
        if pipeline.vision_manager:
            try:
                # 1. Search by text
                raw_results = pipeline.vision_manager.search_by_text(query, k=12)
                
                # 2. Format for template
                for res in raw_results:
                    abs_path = Path(res["path"])
                    try:
                        rel_path = abs_path.relative_to(PROJECT_ROOT)
                    except ValueError:
                        # Fallback if outside project root (shouldn't happen with standard setup)
                        rel_path = abs_path.name
                    
                    results.append({
                        "filename": abs_path.name,
                        "folder": abs_path.parent.name,
                        "score": res["score"],
                        "relative_path": str(rel_path).replace("\\", "/")
                    })
            except Exception as e:
                flash(f"Error en la búsqueda visual: {e}", "error")
                get_logger().error(f"Visual search failed: {e}")

    return render_template("visual_search.html", query=query, results=results)


@app.route("/gallery")
def gallery():
    """Render the main gallery SPA."""
    return render_template("gallery.html")


@app.route("/api/visual_search")
def api_visual_search():
    """JSON endpoint for gallery grid."""
    query = request.args.get("q", "").strip()
    k = int(request.args.get("k", 50))
    
    pipeline = get_pipeline()
    vision_manager = pipeline.vision_manager
    results = []

    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    try:
        # 1. Get raw candidates from FAISS/CLIP
        if query:
            raw_results = vision_manager.search_by_text(query, k=k)
        else:
            # If no query, return latest images (from DB actually, but let's use all indexed for now)
            # FAISS doesn't support "get all", so we might need a workaround or just query DB.
            # Strategy: Query DB for latest 50 docs with images, return them.
            # But the gallery expects format similar to search results.
            # Let's fallback to DB query for empty search.
            get_logger().info("Empty query, fetching recent docs from DB")
            db = get_db()
            rows = db.conn.execute(
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

        # 2. Enrich search results with DB metadata
        db = get_db()
        clean_results = []
        for res in raw_results:
            path = res["path"]
            # Look up in DB
            # We use LIKE in case of path separator differences or normalization issues
            # But try exact match first for speed
            row = db.conn.execute(
                "SELECT id, filename, datetime, tags FROM documents WHERE path = ?", 
                (path,)
            ).fetchone()
            
            if row:
                clean_results.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "path": path, # Keep absolute for debugging, but UI uses id
                    "score": res["score"],
                    "date": row["datetime"],
                    "tags": json.loads(row["tags"]) if row["tags"] else []
                })
            else:
                 # It's in the index but not in DB? Might be stale index. Skip.
                 pass
                 
        return jsonify(clean_results)

    except Exception as e:
        get_logger().error(f"Gallery API failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search/similar/<int:doc_id>")
def api_search_similar(doc_id):
    k = request.args.get("k", 12, type=int)
    vision_manager = get_pipeline().vision_manager
    
    if not vision_manager or not vision_manager.config.enabled:
        return jsonify([])

    try:
        db = get_db()
        path = db.get_document_path(doc_id)
        if not path or not os.path.exists(path):
            return jsonify({"error": "Document file not found"}), 404
            
        # 1. Search for similar images
        raw_results = vision_manager.search_similar(path, k=k)
        
        # 2. Enrich results with DB metadata
        clean_results = []
        for res in raw_results:
            row = db.conn.execute(
                "SELECT id, filename, datetime, tags FROM documents WHERE path = ?", 
                (res["path"],)
            ).fetchone()
            
            if row:
                if row["id"] == doc_id: # Skip the query document itself
                    continue
                    
                clean_results.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "path": res["path"],
                    "score": res["score"],
                    "date": row["datetime"],
                    "tags": json.loads(row["tags"]) if row["tags"] else []
                })
        
        return jsonify(clean_results)

    except Exception as e:
        get_logger().error(f"Similar search failed for doc {doc_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/merge_documents", methods=["POST"])
def merge_documents():
    data = request.json
    doc_ids = data.get("doc_ids", [])
    if not doc_ids or len(doc_ids) < 2:
        return jsonify({"success": False, "message": "Seleccione al menos 2 documentos."})

    try:
        from pypdf import PdfWriter
        merger = PdfWriter()
        
        db = get_db()
        input_paths = []
        
        for doc_id in doc_ids:
            path = db.get_document_path(doc_id)
            if path and os.path.exists(path):
                input_paths.append(path)
                merger.append(path)
            else:
                return jsonify({"success": False, "message": f"Documento {doc_id} no encontrado."})

        # Generate output path
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_{timestamp_str}.pdf"
        output_path = os.path.join(os.path.dirname(input_paths[0]), output_filename)

        merger.write(output_path)
        merger.close()

        # Add to DB
        file_hash = compute_hash(output_path)
        file_stats = os.stat(output_path)
        
        new_id = db.insert_document(
            filename=output_filename,
            path=output_path,
            md5_hash=file_hash,
            timestamp=datetime.datetime.now(),
            duration=0.0,
            status="MERGED",
            doc_type="Merged PDF",
            tags=["Merged"]
        )

        return jsonify({"success": True, "message": "Documentos unidos correctamente.", "new_id": new_id})

    except Exception as e:
        get_logger().error(f"Merge failed: {e}")
        return jsonify({"success": False, "message": str(e)})


        return jsonify({"success": False, "message": str(e)})


@app.route("/split_document", methods=["POST"])
def split_document():
    data = request.json
    doc_id = data.get("doc_id")
    page_range = data.get("pages", "") # e.g. "1-2" or "1,3"

    if not doc_id or not page_range:
        return jsonify({"success": False, "message": "Faltan datos."})

    try:
        from pypdf import PdfReader, PdfWriter
        db = get_db()
        path = db.get_document_path(doc_id)
        if not path or not os.path.exists(path):
            return jsonify({"success": False, "message": "Documento no encontrado."})

        reader = PdfReader(path)
        writer = PdfWriter()
        total_pages = len(reader.pages)
        
        # Parse range
        selected_indices = set()
        parts = page_range.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                 start, end = map(int, part.split("-"))
                 # 1-based to 0-based
                 for i in range(start, end + 1):
                      selected_indices.add(i - 1)
            else:
                 selected_indices.add(int(part) - 1)
        
        valid_indices = sorted([i for i in selected_indices if 0 <= i < total_pages])
        
        if not valid_indices:
             return jsonify({"success": False, "message": "Rango de páginas inválido."})

        for i in valid_indices:
            writer.add_page(reader.pages[i])

        # Generate output path
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.splitext(os.path.basename(path))[0]
        output_filename = f"{input_filename}_split_{timestamp_str}.pdf"
        output_path = os.path.join(os.path.dirname(path), output_filename)

        with open(output_path, "wb") as f:
            writer.write(f)

        # Add to DB
        file_hash = compute_hash(output_path)
        
        new_id = db.insert_document(
            filename=output_filename,
            path=output_path,
            md5_hash=file_hash,
            timestamp=datetime.datetime.now(),
            duration=0.0,
            status="SPLIT",
            doc_type="Split PDF",
            tags=["Split"]
        )
        
        return jsonify({"success": True, "message": "Documento dividido.", "new_id": new_id})

    except Exception as e:
        get_logger().error(f"Split failed: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/serve/<path:filename>")
def serve_file(filename):
    try:
        # Securely serve file from PROJECT_ROOT
        safe_path = ensure_within_project(PROJECT_ROOT / filename)
        return send_from_directory(safe_path.parent, safe_path.name)
    except Exception:
        abort(404)


@app.route("/tables/<int:doc_id>/<int:index>/<fmt>")
def download_table(doc_id: int, index: int, fmt: str):
    tables = load_tables_for_document(doc_id)
    if index < 0 or index >= len(tables):
        flash("Tabla no encontrada.", "error")
        return redirect(url_for("document_detail", doc_id=doc_id))
    table = tables[index]
    path_key = "csv_path" if fmt == "csv" else "json_path"
    path_value = table.get(path_key)
    if not path_value:
        flash("Archivo no disponible.", "error")
        return redirect(url_for("document_detail", doc_id=doc_id))
    try:
        file_path = ensure_within_project(Path(path_value))
    except ValueError:
        flash("Acceso a archivo denegado.", "error")
        return redirect(url_for("document_detail", doc_id=doc_id))
    return send_from_directory(file_path.parent, file_path.name, as_attachment=True)


def process_uploaded_file(*args, **kwargs):
    """Backwards compatibility shim."""
    raise NotImplementedError("process_uploaded_file is handled by postbatch pipeline.")


if __name__ == "__main__":
    init_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
