"""
Legacy Celery compatibility module.

Canonical Celery configuration now lives in `celery_app.py`, and canonical task
logic lives in `modules.tasks`. This module keeps old import paths and task
names stable for existing scripts/deployments:

- `modules.celery_app.app`
- `modules.celery_app.process_document_task`
- `modules.celery_app.process_batch_task`
- `modules.celery_app.generate_embeddings_task`
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from modules.tasks import _process_document_logic, _rebuild_index_logic

logger = logging.getLogger(__name__)


class _NoopInspector:
    def active(self):
        return None


class _NoopControl:
    def inspect(self):
        return _NoopInspector()


class _NoopCeleryApp:
    main = "autoocr"
    control = _NoopControl()

    @staticmethod
    def task(*_args, **_kwargs):
        def decorator(func):
            def _missing_celery(*__args, **__kwargs):
                raise RuntimeError("Celery is not installed in this environment")

            # Compatibility with call sites expecting Celery task methods.
            func.delay = _missing_celery  # type: ignore[attr-defined]
            func.apply_async = _missing_celery  # type: ignore[attr-defined]
            return func

        return decorator


try:
    from celery_app import celery_app as app
except Exception as exc:  # pragma: no cover - optional dependency fallback
    logger.warning("Celery app is unavailable, using no-op compatibility app: %s", exc)
    app = _NoopCeleryApp()


def _build_options(
    *,
    owner_id: Optional[str] = None,
    hotel_id: Optional[str] = None,
    doc_type: str = "other",
    priority: int = 0,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "delete_original": True,
        "ocr_enabled": True,
        "classification_enabled": True,
        "doc_type": doc_type or "other",
    }
    if owner_id is not None:
        options["owner_id"] = owner_id
    if hotel_id is not None:
        options["hotel_id"] = hotel_id
    if priority > 0:
        options["priority"] = "high"
    return options


@app.task(bind=True, name="modules.celery_app.process_document_task", max_retries=2)
def process_document_task(
    self=None,
    source_path: Optional[str] = None,
    tenant_id: Optional[str] = None,  # kept for compatibility
    hotel_id: Optional[str] = None,
    project_id: Optional[str] = None,  # kept for compatibility
    owner_id: Optional[str] = None,
    doc_type: str = "other",
    priority: int = 0,
):
    del tenant_id, project_id
    if not source_path:
        return {"status": "FAILED", "error": "source_path is required"}
    options = _build_options(
        owner_id=owner_id,
        hotel_id=hotel_id,
        doc_type=doc_type,
        priority=priority,
    )
    try:
        result = _process_document_logic(source_path, options)
        if isinstance(result, dict) and result.get("status") == "FAILED":
            raise RuntimeError(result.get("error") or "document processing failed")
        return result
    except Exception as exc:
        if self is not None and hasattr(self, "retry"):
            retries = int(getattr(getattr(self, "request", None), "retries", 0) or 0)
            countdown = min(300, 60 * (retries + 1))
            raise self.retry(exc=exc, countdown=countdown)
        return {"status": "FAILED", "error": str(exc)}


@app.task(bind=True, name="modules.celery_app.process_batch_task")
def process_batch_task(
    self=None,
    file_paths: Optional[Iterable[str]] = None,
    tenant_id: Optional[str] = None,  # kept for compatibility
    hotel_id: Optional[str] = None,
    doc_type: str = "other",
):
    del self, tenant_id
    submitted = []
    for source_path in file_paths or []:
        kwargs = {
            "source_path": source_path,
            "hotel_id": hotel_id,
            "doc_type": doc_type,
        }
        try:
            task = process_document_task.apply_async(kwargs=kwargs, queue="ocr_batch")
            submitted.append({"file": source_path, "task_id": task.id})
        except Exception:
            # No Celery runtime: process synchronously for compatibility tooling.
            result = _process_document_logic(source_path, _build_options(hotel_id=hotel_id, doc_type=doc_type))
            submitted.append({"file": source_path, "task_id": None, "status": result.get("status")})
    return {"submitted": len(submitted), "tasks": submitted}


@app.task(name="modules.celery_app.generate_embeddings_task")
def generate_embeddings_task(
    document_id: str,
    tenant_id: Optional[str] = None,  # kept for compatibility
    hotel_id: Optional[str] = None,  # kept for compatibility
):
    del tenant_id, hotel_id
    logger.warning(
        "Legacy task modules.celery_app.generate_embeddings_task was called for doc=%s; "
        "delegating to index rebuild compatibility path.",
        document_id,
    )
    try:
        _rebuild_index_logic()
    except Exception as exc:
        return {"status": "failed", "document_id": document_id, "error": str(exc)}
    return {
        "status": "completed",
        "document_id": document_id,
        "message": "Compatibility path executed via vision index rebuild.",
    }


__all__ = ["app", "process_document_task", "process_batch_task", "generate_embeddings_task"]
