import os
from celery import Celery
from celery.schedules import crontab

# Create Celery instance
celery_app = Celery("autoocr")

# Load configuration from environment variables
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Configure Celery
celery_app.conf.update(
    broker_url=broker_url,
    result_backend=result_backend,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        # Default routing (can be overridden by apply_async options)
        "modules.tasks._process_document_task_celery": {"queue": "ocr_batch"},
    },
    task_default_queue="ocr_default",
    task_acks_late=True,  # Ensure tasks are only acked after successful execution
    worker_prefetch_multiplier=1,  # Prevent worker from hogging tasks
    task_reject_on_worker_lost=True,  # Re-queue if worker crashes
    beat_schedule={
        # Keep ephemeral artifacts bounded in 24/7 deployments.
        "purge-ephemeral-data-daily": {
            "task": "modules.tasks.purge_ephemeral_data_daily_celery",
            "schedule": crontab(minute=15, hour=3),
            "options": {"queue": "ocr_default"},
        },
    },
)

# Auto-discover tasks from modules
celery_app.autodiscover_tasks(["modules"])

if __name__ == "__main__":
    celery_app.start()
