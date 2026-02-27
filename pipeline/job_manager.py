"""
Pipeline Job Manager — Async job tracking, retry, and idempotency.

Manages the lifecycle of document processing jobs with:
- Idempotency keys to prevent duplicate work
- Configurable retry with exponential backoff
- Status tracking per step
- Worker locking to prevent concurrent processing of the same document
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages async processing jobs for the document pipeline.

    Parameters
    ----------
    db:
        Database connection (PostgreSQL).
    max_attempts:
        Maximum retry attempts for failed jobs.
    retry_base_delay:
        Base delay (seconds) for exponential backoff.
    """

    def __init__(
        self,
        db: Any,
        max_attempts: int = 3,
        retry_base_delay: int = 30,
    ):
        self.db = db
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay

    def create_job(
        self,
        tenant_id: str,
        document_id: str,
        job_type: str,
        priority: int = 0,
    ) -> Optional[str]:
        """
        Create a processing job if no duplicate exists.

        Returns the job ID, or ``None`` if an identical pending job exists
        (idempotency check).
        """
        job_id = str(uuid.uuid4())
        idempotency_key = f"{document_id}:{job_type}"

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Check for existing pending/running job with same key
                cursor.execute(
                    """
                    SELECT id FROM processing_jobs
                    WHERE idempotency_key = %s
                      AND status IN ('pending', 'running', 'retrying')
                    LIMIT 1
                    """,
                    (idempotency_key,),
                )
                existing = cursor.fetchone()
                if existing:
                    logger.info(
                        "Job already exists for %s:%s — skipping",
                        document_id[:8],
                        job_type,
                    )
                    return None

                cursor.execute(
                    """
                    INSERT INTO processing_jobs (
                        id, tenant_id, document_id, job_type,
                        priority, status, idempotency_key, max_attempts
                    ) VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s)
                    """,
                    (
                        job_id,
                        tenant_id,
                        document_id,
                        job_type,
                        priority,
                        idempotency_key,
                        self.max_attempts,
                    ),
                )
                conn.commit()

            logger.info("Created job %s: %s for doc %s", job_id[:8], job_type, document_id[:8])
            return job_id

        except Exception as e:
            logger.error("Failed to create job: %s", e, exc_info=True)
            return None

    def claim_job(self, job_id: str, worker_id: str) -> bool:
        """
        Atomically claim a job for processing (worker lock).

        Returns True if successfully claimed, False if already taken.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'running',
                        worker_id = %s,
                        started_at = NOW(),
                        attempt = attempt + 1
                    WHERE id = %s AND status IN ('pending', 'retrying')
                    """,
                    (worker_id, job_id),
                )
                claimed = cursor.rowcount > 0
                conn.commit()
                return claimed

        except Exception as e:
            logger.error("Failed to claim job %s: %s", job_id, e)
            return False

    def complete_job(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None,
        processing_time_ms: int = 0,
    ) -> None:
        """Mark a job as completed."""
        try:
            import json

            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'completed',
                        completed_at = NOW(),
                        result = %s,
                        processing_time_ms = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(result or {}),
                        processing_time_ms,
                        job_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error("Failed to complete job %s: %s", job_id, e)

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        error_traceback: Optional[str] = None,
    ) -> bool:
        """
        Mark a job as failed.  Schedules a retry if attempts remain.

        Returns True if the job was scheduled for retry.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Check if retries remain
                cursor.execute(
                    "SELECT attempt, max_attempts FROM processing_jobs WHERE id = %s",
                    (job_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                attempt, max_attempts = row[0], row[1]
                should_retry = attempt < max_attempts

                if should_retry:
                    # Exponential backoff
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    next_retry = datetime.now(timezone.utc) + timedelta(seconds=delay)

                    cursor.execute(
                        """
                        UPDATE processing_jobs
                        SET status = 'retrying',
                            error_message = %s,
                            error_traceback = %s,
                            next_retry_at = %s
                        WHERE id = %s
                        """,
                        (error_message, error_traceback, next_retry, job_id),
                    )
                    logger.info(
                        "Job %s scheduled for retry #%d in %ds",
                        job_id[:8],
                        attempt + 1,
                        delay,
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE processing_jobs
                        SET status = 'failed',
                            error_message = %s,
                            error_traceback = %s,
                            completed_at = NOW()
                        WHERE id = %s
                        """,
                        (error_message, error_traceback, job_id),
                    )
                    logger.error(
                        "Job %s permanently failed after %d attempts: %s",
                        job_id[:8],
                        attempt,
                        error_message,
                    )

                conn.commit()
                return should_retry

        except Exception as e:
            logger.error("Failed to update job %s: %s", job_id, e)
            return False

    def get_pending_jobs(
        self,
        limit: int = 10,
        job_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch pending/retriable jobs ordered by priority and schedule time.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT id, tenant_id, document_id, job_type, priority, attempt
                    FROM processing_jobs
                    WHERE status IN ('pending', 'retrying')
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                """
                params: list = []

                if job_type:
                    query += " AND job_type = %s"
                    params.append(job_type)

                query += " ORDER BY priority DESC, scheduled_at ASC LIMIT %s"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [
                    {
                        "id": str(r[0]),
                        "tenant_id": str(r[1]),
                        "document_id": str(r[2]),
                        "job_type": r[3],
                        "priority": r[4],
                        "attempt": r[5],
                    }
                    for r in rows
                ]

        except Exception as e:
            logger.error("Failed to fetch pending jobs: %s", e)
            return []

    def create_pipeline_jobs(
        self,
        tenant_id: str,
        document_id: str,
        steps: Optional[List[str]] = None,
        priority: int = 0,
    ) -> List[str]:
        """
        Create a full pipeline of jobs for a document.

        Parameters
        ----------
        steps:
            List of step names. Defaults to the full pipeline.
        """
        if steps is None:
            steps = [
                "ingestion",
                "ocr",
                "layout",
                "visual_analysis",
                "chunking",
                "embedding",
            ]

        created = []
        for step in steps:
            job_id = self.create_job(
                tenant_id=tenant_id,
                document_id=document_id,
                job_type=step,
                priority=priority,
            )
            if job_id:
                created.append(job_id)

        return created


__all__ = ["JobManager"]
