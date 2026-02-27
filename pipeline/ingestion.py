"""
Pipeline Step A — Document Ingestion.

Normalizes file input, computes content hash, extracts metadata, stores the
original file, and creates a processing job record.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class IngestionResult:
    """Result of the ingestion step."""

    __slots__ = (
        "document_id",
        "file_path",
        "md5_hash",
        "mime_type",
        "file_size",
        "page_count",
        "metadata",
    )

    def __init__(
        self,
        document_id: str,
        file_path: str,
        md5_hash: str,
        mime_type: str,
        file_size: int,
        page_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.document_id = document_id
        self.file_path = file_path
        self.md5_hash = md5_hash
        self.mime_type = mime_type
        self.file_size = file_size
        self.page_count = page_count
        self.metadata = metadata or {}


class IngestionStep:
    """
    Normalize file input, generate hash & metadata, store original.

    Parameters
    ----------
    storage_root:
        Base directory for stored documents. Files are organized as:
        ``<storage_root>/<tenant_id>/<hotel_id>/<YYYY>/<MM>/<filename>``
    db:
        Database connection / manager for inserting document records.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp",
        ".gif", ".docx", ".xlsx", ".xlsm", ".csv", ".txt", ".json",
        ".eml", ".webp", ".jfif", ".avif",
    }

    def __init__(self, storage_root: str, db: Any = None):
        self.storage_root = Path(storage_root)
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        source_path: str,
        *,
        tenant_id: str,
        hotel_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        doc_type: str = "other",
        visibility: str = "private",
        tags: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest a single file into the platform.

        1. Validate file type
        2. Compute MD5 hash
        3. Check for duplicates
        4. Copy to organized storage
        5. Insert document record
        6. Create processing job

        Returns an :class:`IngestionResult` with the new document ID and path.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        ext = src.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # 1. Compute hash & file metadata
        md5_hash = self._compute_hash(src)
        file_size = src.stat().st_size
        mime_type = mimetypes.guess_type(str(src))[0] or "application/octet-stream"

        # 2. Check duplicates
        if self.db is not None:
            existing = self._check_duplicate(md5_hash, hotel_id)
            if existing:
                logger.info("Duplicate detected: %s (existing doc %s)", src.name, existing)
                raise DuplicateDocumentError(
                    f"Document already exists with hash {md5_hash}", existing_id=existing
                )

        # 3. Generate document ID and storage path
        doc_id = str(uuid.uuid4())
        dest_path = self._compute_storage_path(
            tenant_id=tenant_id,
            hotel_id=hotel_id,
            filename=src.name,
            doc_id=doc_id,
        )

        # 4. Copy file to organized storage
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest_path))
        logger.info("Stored: %s → %s", src.name, dest_path)

        # 5. Count pages (for PDFs)
        page_count = self._count_pages(dest_path, mime_type)

        # 6. Insert document record
        if self.db is not None:
            self._insert_document_record(
                doc_id=doc_id,
                tenant_id=tenant_id,
                hotel_id=hotel_id,
                project_id=project_id,
                owner_id=owner_id,
                filename=src.name,
                original_filename=src.name,
                file_path=str(dest_path.relative_to(self.storage_root)),
                file_size=file_size,
                mime_type=mime_type,
                md5_hash=md5_hash,
                page_count=page_count,
                doc_type=doc_type,
                visibility=visibility,
                tags=tags,
                metadata=metadata,
            )

        return IngestionResult(
            document_id=doc_id,
            file_path=str(dest_path),
            md5_hash=md5_hash,
            mime_type=mime_type,
            file_size=file_size,
            page_count=page_count,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(path: Path, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of file contents."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _compute_storage_path(
        self,
        tenant_id: str,
        hotel_id: Optional[str],
        filename: str,
        doc_id: str,
    ) -> Path:
        """Generate the organized storage path."""
        now = datetime.now(timezone.utc)
        parts = [
            self.storage_root,
            tenant_id[:8],
            hotel_id[:8] if hotel_id else "_unscoped",
            str(now.year),
            f"{now.month:02d}",
        ]
        # Prefix filename with short doc_id to avoid collisions
        safe_name = f"{doc_id[:8]}_{filename}"
        return Path(*[str(p) for p in parts]) / safe_name

    def _check_duplicate(self, md5_hash: str, hotel_id: Optional[str]) -> Optional[str]:
        """Check if a document with this hash already exists in the same hotel."""
        if self.db is None:
            return None
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if hotel_id:
                    cursor.execute(
                        "SELECT id FROM documents WHERE md5_hash = %s AND hotel_id = %s",
                        (md5_hash, hotel_id),
                    )
                else:
                    cursor.execute(
                        "SELECT id FROM documents WHERE md5_hash = %s",
                        (md5_hash,),
                    )
                row = cursor.fetchone()
                return str(row[0]) if row else None
        except Exception as e:
            logger.warning("Duplicate check failed: %s", e)
            return None

    @staticmethod
    def _count_pages(path: Path, mime_type: str) -> int:
        """Count pages in a PDF document."""
        if mime_type != "application/pdf":
            return 1
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def _insert_document_record(self, **kwargs: Any) -> None:
        """Insert a document record into the database."""
        if self.db is None:
            return
        try:
            import json

            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents (
                        id, tenant_id, hotel_id, project_id, owner_id,
                        filename, original_filename, file_path, file_size,
                        mime_type, md5_hash, page_count, doc_type,
                        visibility, tags, metadata, status
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, 'uploaded'
                    )
                    """,
                    (
                        kwargs["doc_id"],
                        kwargs["tenant_id"],
                        kwargs.get("hotel_id"),
                        kwargs.get("project_id"),
                        kwargs.get("owner_id"),
                        kwargs["filename"],
                        kwargs.get("original_filename"),
                        kwargs["file_path"],
                        kwargs["file_size"],
                        kwargs["mime_type"],
                        kwargs["md5_hash"],
                        kwargs["page_count"],
                        kwargs.get("doc_type", "other"),
                        kwargs.get("visibility", "private"),
                        json.dumps(kwargs.get("tags") or []),
                        json.dumps(kwargs.get("metadata") or {}),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to insert document record: %s", e, exc_info=True)
            raise


class DuplicateDocumentError(Exception):
    """Raised when a document with the same hash already exists."""

    def __init__(self, message: str, existing_id: str):
        super().__init__(message)
        self.existing_id = existing_id


__all__ = ["IngestionStep", "IngestionResult", "DuplicateDocumentError"]
