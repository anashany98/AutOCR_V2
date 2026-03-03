"""
AutOCR Web Interface.

Flask application for managing documents, OCR processing, table extraction and
vision search capabilities.
Reference implementation for Refactoring.
"""

from __future__ import annotations

import mimetypes
import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import Flask
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Try to import Flask-SocketIO (optional dependency)
try:
    from flask_socketio import SocketIO, emit, join_room
    FLASK_SOCKETIO_AVAILABLE = True
except ImportError:
    FLASK_SOCKETIO_AVAILABLE = False
    SocketIO = None

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
# Store uploads outside of `static/` to prevent unauthenticated public access.
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


app = Flask(__name__)


def create_app(testing=False):
    """Create and configure the Flask application.
    
    Args:
        testing: If True, configure app for testing mode.
        
    Returns:
        The configured Flask application.
    """
    # Import the global app and configure it for testing
    import web_app.app as app_module
    app_instance = app_module.app
    
    if testing:
        app_instance.config["TESTING"] = True
        app_instance.config["WTF_CSRF_ENABLED"] = False
    
    return app_instance


# Security: Force SECRET_KEY in production
_secret_key = os.environ.get("FLASK_SECRET_KEY") or os.environ.get("SECRET_KEY")
if not _secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "CRITICAL: FLASK_SECRET_KEY environment variable must be set in production! "
            "Set FLASK_ENV=development for local development."
        )
    _secret_key = secrets.token_hex(32)  # Development only

app.config["SECRET_KEY"] = _secret_key
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") == "production"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["UPLOAD_FOLDER"] = str(DEFAULT_UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Initialize SocketIO (optional)
socketio = None
if FLASK_SOCKETIO_AVAILABLE:
    cors_origins_env = (os.environ.get("SOCKETIO_CORS_ALLOWED_ORIGINS") or "").strip()
    cors_allowed_origins = None
    if cors_origins_env:
        cors_allowed_origins = [
            o.strip() for o in cors_origins_env.split(",") if o.strip()
        ] or None
    socketio = SocketIO(app, cors_allowed_origins=cors_allowed_origins, async_mode='threading')
    
    # SocketIO event handlers
    @socketio.on('connect')
    def handle_connect():
        from flask_login import current_user
        if current_user.is_authenticated:
            # Join user-specific room for targeted notifications
            join_room(f'user_{current_user.id}')
            emit('connected', {'status': 'ok', 'user_id': current_user.id})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        pass
    
    @socketio.on('join_room')
    def handle_join_room(data):
        room = data.get('room')
        if room:
            join_room(room)

# Function to emit events from other parts of the app
def emit_task_update(task_id, status, message):
    """Emit task update to connected clients."""
    if socketio:
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': status,
            'message': message
        }, room=f'user_0')  # Broadcast to admins

def emit_document_update(document_id, action):
    """Emit document update to connected clients."""
    if socketio:
        socketio.emit('document_update', {
            'document_id': document_id,
            'action': action
        }, broadcast=True)

# Cookie hardening for session-authenticated UI/API usage.
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
if os.environ.get("FLASK_COOKIE_SECURE", "0") == "1":
    app.config.setdefault("SESSION_COOKIE_SECURE", True)

# CSRF protection for cookie-authenticated web/API endpoints.
# JS clients must send `X-CSRFToken` for state-changing requests.
app.config.setdefault("WTF_CSRF_HEADERS", ["X-CSRFToken", "X-CSRF-Token"])
# Reduce UX breakage for long-lived dashboard sessions.
app.config.setdefault("WTF_CSRF_TIME_LIMIT", 8 * 60 * 60)  # 8 hours
csrf = CSRFProtect(app)

# Defensive: ensure templates never crash if CSRF globals are missing for any reason.
# In normal operation CSRFProtect registers `csrf_token()` automatically.
try:  # pragma: no cover - extremely defensive
    from flask_wtf.csrf import generate_csrf
    app.jinja_env.globals.setdefault("csrf_token", generate_csrf)
except Exception:
    app.jinja_env.globals.setdefault("csrf_token", lambda: "")

# Security: Rate Limiting
_rate_limit_storage = (os.environ.get("RATELIMIT_STORAGE_URI") or "").strip()
if not _rate_limit_storage:
    if os.environ.get("AUTOOCR_ENV") == "production":
        # Default to Redis-backed storage in production for multi-worker consistency.
        _rate_limit_storage = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/2")
    else:
        _rate_limit_storage = "memory://"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per minute", "10 per second"],
    storage_uri=_rate_limit_storage,
)

# Export socketio for other modules
app.socketio = socketio

# Add built-in functions to Jinja2 globals
app.jinja_env.globals.update(max=max, min=min)
app.jinja_env.filters['basename'] = os.path.basename


# --------------------------------------------------------------------------- #
# Blueprints Registration & Security
# --------------------------------------------------------------------------- #
from flask_login import LoginManager
from modules.auth_manager import AuthManager

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

@login_manager.unauthorized_handler
def _handle_unauthorized():
    from flask import jsonify, redirect, request, url_for

    if request.path.startswith("/api/"):
        return jsonify({"error": "authentication_required"}), 401
    return redirect(url_for("auth.login"))

@login_manager.user_loader
def load_user(user_id):
    from web_app.services import get_db
    auth = AuthManager(get_db())
    return auth.get_user(user_id)

from web_app.routes.main_routes import main_bp
from web_app.routes.api_routes import api_bp
from web_app.routes.chat_routes import chat_bp
from web_app.routes.error_handlers import errors_bp
from web_app.routes.auth_routes import auth_bp
from web_app.routes.admin_routes import admin_bp

# Telegram Bot integration (optional)
try:
    from web_app.routes.telegram_routes import telegram_bp
    TELEGRAM_AVAILABLE = True
except ImportError:
    telegram_bp = None
    TELEGRAM_AVAILABLE = False

# API Documentation (optional)
try:
    from web_app.routes.api_docs import api_docs_bp, init_swagger
    API_DOCS_AVAILABLE = True
except ImportError:
    api_docs_bp = None
    init_swagger = None
    API_DOCS_AVAILABLE = False

app.register_blueprint(main_bp)
app.register_blueprint(api_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(errors_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)

# Register API docs blueprint (optional)
if api_docs_bp and API_DOCS_AVAILABLE:
    app.register_blueprint(api_docs_bp)
    if init_swagger:
        init_swagger(app)
    print("✅ API documentation enabled at /api/docs")
else:
    print("⚠️ API docs not available (install flasgger)")

# Register Telegram bot blueprint (optional)
if telegram_bp and TELEGRAM_AVAILABLE:
    app.register_blueprint(telegram_bp)
    print("✅ Telegram bot integration enabled")
else:
    print("⚠️ Telegram bot not available (install python-telegram-bot)")

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
            get_logger().info("Application singletons pre-warmed and ready.")
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

if __name__ == "__main__":
    init_app()
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1" or os.environ.get("FLASK_ENV") == "development"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
