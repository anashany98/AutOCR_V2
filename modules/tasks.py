from huey import SqliteHuey
from pathlib import Path
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Huey instance using SQLite for persistence on Windows
# Huey instance: Redis (Production/Docker) vs SQLite (Local Windows)
redis_url = os.environ.get('REDIS_URL')
if redis_url:
    from huey import RedisHuey
    huey = RedisHuey('autoocr', url=redis_url)
else:
    huey = SqliteHuey(filename=str(PROJECT_ROOT / 'data' / 'huey_db.db'))



def _process_document_logic(file_path, options):
    """
    Core logic for processing a single document.
    """
    from postbatch_processor import process_single_file, initialise_pipeline
    from web_app.app import get_db, get_logger, resolve_path, load_configuration, get_pipeline, get_classifier
    
    logger = get_logger()
    db = get_db()
    
    # Reload config to get latest paths etc
    config = load_configuration(reload=True)
    post_conf = config.get("postbatch", {})
    
    # Inject Phase 4 Metadata into config for process_single_file
    for key in ["owner_id", "hotel_id", "doc_type", "visibility", "financial_level"]:
        if options.get(key):
            config[key] = options[key]
            post_conf[key] = options[key]
    
    processed_folder = resolve_path(post_conf.get("processed_folder"), "data/scans_processed")
    failed_folder = resolve_path(post_conf.get("failed_folder"), "data/scans_failed")
    
    pipeline = get_pipeline()
    classifier = get_classifier()
    
    try:
        result = process_single_file(
            file_path,
            pipeline,
            classifier,
            db,
            processed_folder,
            failed_folder,
            delete_original=options.get("delete_original", True),
            ocr_enabled=options.get("ocr_enabled", True),
            classification_enabled=options.get("classification_enabled", True),
            logger=logger,
            input_root=options.get("input_root", "input"),
            handwriting_mode=options.get("handwriting_mode", False),
            pipeline_conf=config,
        )
        return result
    except Exception as e:
        logger.error(f"Task failed for {file_path}: {e}", exc_info=True)
        return {"status": "FAILED", "error": str(e)}


# --------------------------------------------------------------------------- #
# Task Definitions (Huey vs Celery)
# --------------------------------------------------------------------------- #

# 1. Huey Task (Development / Windows)
@huey.task()
def _process_document_task_huey(file_path, options):
    return _process_document_logic(file_path, options)

# 2. Celery Task (Production / Docker)
try:
    from celery_app import celery_app
    @celery_app.task(queue="ocr")
    def _process_document_task_celery(file_path, options):
        return _process_document_logic(file_path, options)
except (ImportError, ModuleNotFoundError):
    _process_document_task_celery = None


# 3. Dispatcher
def process_document_task(file_path, options):
    """
    Dispatch task to appropriate queue backend based on environment.
    """
    if os.environ.get("AUTOOCR_ENV") == "production" and _process_document_task_celery:
        # Determine priority based on file size or options
        queue_name = "ocr_batch" # Default
        
        try:
            # Check file size (e.g., < 5MB goes to fast lane)
            if os.path.exists(file_path) and os.path.getsize(file_path) < 5 * 1024 * 1024:
                 queue_name = "ocr_fast"
            
            # Forced priority
            if options.get("priority") == "high":
                 queue_name = "ocr_fast"
                 
        except Exception:
            pass # Fallback to batch

        # Celery .delay() handles async enqueueing
        # We use apply_async to specify queue dynamically
        return _process_document_task_celery.apply_async(
            args=[file_path, options],
            queue=queue_name
        )
    else:
        # Huey task instance is callable to enqueue
        return _process_document_task_huey(file_path, options)
