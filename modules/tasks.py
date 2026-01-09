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


@huey.task()
def process_document_task(file_path, options):
    """
    Background task to process a single document with flexible options.
    """
    from postbatch_processor import process_single_file, initialise_pipeline
    from web_app.app import get_db, get_logger, resolve_path, load_configuration, get_pipeline, get_classifier
    
    logger = get_logger()
    db = get_db()
    
    # Reload config to get latest paths etc
    config = load_configuration(reload=True)
    post_conf = config.get("postbatch", {})
    
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
