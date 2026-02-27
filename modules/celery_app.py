"""
Celery Application — Async task queue for document processing.

Defines queues, routing, and pipeline task wrappers for the worker container.

Usage::

    # Start worker
    celery -A modules.celery_app worker --loglevel=info \\
        --queues=ocr_fast,ocr_batch,default --concurrency=2

    # Submit a document for processing
    from modules.celery_app import process_document_task
    process_document_task.delay("/path/to/doc.pdf", tenant_id="...", hotel_id="...")
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from celery import Celery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery("autoocr", broker=BROKER_URL, backend=RESULT_BACKEND)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Madrid",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    task_queues={
        "ocr_fast": {"exchange": "ocr_fast", "routing_key": "ocr.fast"},
        "ocr_batch": {"exchange": "ocr_batch", "routing_key": "ocr.batch"},
        "default": {"exchange": "default", "routing_key": "default"},
    },
    task_routes={
        "modules.celery_app.process_document_task": {"queue": "ocr_fast"},
        "modules.celery_app.process_batch_task": {"queue": "ocr_batch"},
        "modules.celery_app.generate_embeddings_task": {"queue": "default"},
    },
    task_soft_time_limit=300,   # 5 min soft limit
    task_time_limit=600,        # 10 min hard limit
)


# ---------------------------------------------------------------------------
# Lazy service initialization (per-worker process)
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    """Lazy-initialize the pipeline orchestrator in the worker."""
    global _pipeline
    if _pipeline is None:
        import yaml
        from modules.config_normalizer import normalize_config

        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                config = normalize_config(raw if isinstance(raw, dict) else {})

        from modules.db_manager import DBManager

        db = DBManager(config)

        # Import managers
        ocr_mgr = None
        layout_mgr = None
        try:
            from modules.ocr_manager import OCRManager
            ocr_mgr = OCRManager()
        except Exception as e:
            logger.warning("OCR manager init failed: %s", e)

        try:
            from modules.layout_manager import LayoutManager
            layout_mgr = LayoutManager()
        except Exception as e:
            logger.warning("Layout manager init failed: %s", e)

        from pipeline.orchestrator import PipelineOrchestrator

        _pipeline = PipelineOrchestrator(
            config=config,
            db=db,
            ocr_manager=ocr_mgr,
            layout_manager=layout_mgr,
            device="cuda" if os.environ.get("PADDLE_OCR_USE_GPU") == "True" else "cpu",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.task(bind=True, name="modules.celery_app.process_document_task", max_retries=2)
def process_document_task(
    self,
    source_path: str,
    tenant_id: str,
    hotel_id: str = None,
    project_id: str = None,
    owner_id: str = None,
    doc_type: str = "other",
    priority: int = 0,
):
    """
    Process a single document through the full pipeline.

    This is the primary entry point for async document processing.
    """
    logger.info("Processing document: %s (tenant=%s)", source_path, tenant_id[:8])

    try:
        pipeline = _get_pipeline()
        result = pipeline.process_document(
            source_path,
            tenant_id=tenant_id,
            hotel_id=hotel_id,
            project_id=project_id,
            owner_id=owner_id,
            doc_type=doc_type,
            priority=priority,
        )

        if result["status"] == "failed":
            raise Exception("; ".join(result.get("errors", ["Unknown error"])))

        return result

    except Exception as exc:
        logger.error("Document processing failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


@app.task(bind=True, name="modules.celery_app.process_batch_task")
def process_batch_task(
    self,
    file_paths: list,
    tenant_id: str,
    hotel_id: str = None,
    doc_type: str = "other",
):
    """Process a batch of documents. Submits individual tasks per file."""
    results = []
    for path in file_paths:
        task = process_document_task.apply_async(
            kwargs={
                "source_path": path,
                "tenant_id": tenant_id,
                "hotel_id": hotel_id,
                "doc_type": doc_type,
            },
            queue="ocr_batch",
        )
        results.append({"file": path, "task_id": task.id})

    return {"submitted": len(results), "tasks": results}


@app.task(name="modules.celery_app.generate_embeddings_task")
def generate_embeddings_task(document_id: str, tenant_id: str, hotel_id: str = None):
    """Re-generate embeddings for a document (e.g., after model change)."""
    pipeline = _get_pipeline()
    # Fetch chunks from DB and re-embed
    try:
        with pipeline.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, content, content_type FROM chunks WHERE document_id = %s",
                (document_id,),
            )
            rows = cursor.fetchall()

        if not rows:
            return {"status": "no_chunks", "document_id": document_id}

        chunk_dicts = [
            {"chunk_id": str(r[0]), "content": r[1], "content_type": r[2]}
            for r in rows
        ]

        result = pipeline.embedding.process(
            document_id=document_id,
            tenant_id=tenant_id,
            hotel_id=hotel_id,
            chunks=chunk_dicts,
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "chunks_embedded": result.num_chunks,
        }

    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        return {"status": "failed", "error": str(e)}


__all__ = ["app", "process_document_task", "process_batch_task", "generate_embeddings_task"]
