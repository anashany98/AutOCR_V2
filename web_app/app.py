"""
AutOCR Web Interface.

Flask application for managing documents, OCR processing, table extraction and
vision search capabilities.
Reference implementation for Refactoring.
"""

from __future__ import annotations

import mimetypes
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Re-export PROJECT_ROOT and get_logger for serve.py compatibility
from web_app.services import (
    PROJECT_ROOT, 
    get_logger, 
    get_db, 
    get_pipeline, 
    get_classifier, 
    get_rag_manager, 
    get_tool_manager, 
    load_configuration,
    save_configuration
)

from web_app.utils import resolve_path
from modules.file_utils import ensure_directories
from modules.folder_watcher import FolderWatcher
from modules.tasks import process_document_task
from modules.db_manager import DBManager

# Add local path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
load_dotenv(PROJECT_ROOT / ".env")

# Register new image formats
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/jpeg', '.jfif')
mimetypes.add_type('image/avif', '.avif')


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "web_app" / "static" / "uploads"


app = Flask(__name__)
app.config["SECRET_KEY"] = "autocr-secret-key-change-in-production" # Should load from env/config
app.config["UPLOAD_FOLDER"] = str(DEFAULT_UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Security: Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per minute", "10 per second"],
    storage_uri="memory://",
)

# Add built-in functions to Jinja2 globals
app.jinja_env.globals.update(max=max, min=min)
import os
app.jinja_env.filters['basename'] = os.path.basename


# --------------------------------------------------------------------------- #
# Blueprints Registration
# --------------------------------------------------------------------------- #
from web_app.routes.main_routes import main_bp
from web_app.routes.api_routes import api_bp
from web_app.routes.chat_routes import chat_bp
from web_app.routes.error_handlers import errors_bp

app.register_blueprint(main_bp)
app.register_blueprint(api_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(errors_bp)

# --------------------------------------------------------------------------- #
# Hot Folder Logic (Kept here or moved to separate background manager)
# --------------------------------------------------------------------------- #
_watcher_instance: Optional[FolderWatcher] = None

def process_hot_file(path: Path) -> None:
    """Callback for hot folder watcher."""
    try:
        config = load_configuration()
        post_conf = config.get("postbatch", {})
        
        # Determine folders
        processed_folder = resolve_path(post_conf.get("processed_folder"), "data/scans_processed")
        failed_folder = resolve_path(post_conf.get("failed_folder"), "data/scans_failed")
        ensure_directories(processed_folder, failed_folder)
        
        # New DB Manager for this thread
        app_conf = config.get("app", {})
        db_path = resolve_path(app_conf.get("db_path"), "data/digitalizerai.db")
        # We instantiate DBManager directly here usually? 
        # But get_db is singleton. Here we want thread-safety if db_manager wasn't designed for shared use.
        # DBManager uses sqlite3.connect check_same_thread=False usually?
        # Let's trust get_db if it manages pooling, OR create new.
        # Original code created new DBManager(db_path).
        db = DBManager(config) # Use config constructor for consistency

        logger = get_logger()
        logger.info(f"⚡ Hot Folder: Enqueuing {path.name}")
        
        options = {
            "delete_original": True,
            "ocr_enabled": post_conf.get("ocr_enabled", True),
            "classification_enabled": post_conf.get("classification_enabled", True),
            "input_root": str(path.parent)
        }
        process_document_task(str(path), options)
        
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

# --------------------------------------------------------------------------- #
# App initialisation
# --------------------------------------------------------------------------- #

def init_app():
    try:
        ensure_directories(app.config["UPLOAD_FOLDER"])
        init_watcher()
        if os.environ.get("FLASK_DEBUG", "0") != "1" and os.environ.get("FLASK_ENV") != "development":
             # Pre-warm heavy singletons only in production
            get_db()
            get_pipeline()
            get_classifier()
            get_rag_manager()
            get_tool_manager()
            get_logger().info("✅ Application singletons pre-warmed and ready.")
        else:
            get_logger().info("⚡ DEV MODE: Skipped pre-warming for faster startup (Lazy Loading Enabled)")
    except Exception as exc:
        print(f"Error initialising AutOCR Web App: {exc}")
        import traceback
        traceback.print_exc()
        raise

# Auto-init if running via serve.py calls init_app usually? 
# serve.py imports app, but calls serve(app). It relies on app module executed to init things?
# Actually app.py top level code runs on import.
# `start_server.bat` calls `python serve.py`.
# `serve.py` imports `app`.
# But when does `init_app` run?
# Only if explicitly called.
# Original `app.py` didn't call `init_app` at top level?
# Checking original app.py... 
# Line 386 defines `init_app`. But it wasn't called at bottom ???
# Ah, I might have missed where it was called. 
# Or `serve.py` calls it? `serve.py` just imports app.
# Wait, maybe it WASN'T called? That would explain lazy loading.
# But `init_watcher` needs to run.
# Let's call `init_app()` at the end of `app.py` if we want exact behavior, or ensure `serve.py` calls it.
# Actually, Flask app creation is usually top level.
# I will call `init_app()` at the bottom of this file to ensure it starts up.
# Original app.py had `if __name__ == "__main__": app.run(...)`
# But `serve.py` imports it.
# If I look at `web_app/app.py` line 390, it pre-warms.
# I will invoke `init_app()` here.

# init_app()  <-- Removed to allow lazy init controlled by serve.py
