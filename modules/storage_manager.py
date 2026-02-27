"""
Tenant-Aware Storage Manager.

Organizes files on disk in a tenant → hotel → year/month hierarchy for
clean separation and easy backup/migration per tenant.

Structure::

    data/
    ├── uploads/
    │   └── <tenant_id[:8]>/
    │       └── <hotel_id[:8]>/
    │           └── 2026/
    │               └── 02/
    │                   └── <doc_id[:8]>_filename.pdf
    ├── assets/
    │   └── <tenant_id[:8]>/
    │       └── <doc_id[:8]>/
    │           └── <block_id[:8]>.png
    └── indexes/
        └── <tenant_id[:8]>/
            └── vision_index.faiss
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages the tenant-aware file storage layout.

    Parameters
    ----------
    base_dir:
        Root data directory (default: ``data/``).
    """

    UPLOADS = "uploads"
    ASSETS = "assets"
    INDEXES = "indexes"
    EXPORTS = "exports"
    TEMP = "temp"

    def __init__(self, base_dir: str = "data"):
        self.base = Path(base_dir)

    # ------------------------------------------------------------------
    # Path builders
    # ------------------------------------------------------------------

    def uploads_dir(
        self,
        tenant_id: str,
        hotel_id: Optional[str] = None,
    ) -> Path:
        """Get the uploads directory for a tenant/hotel."""
        parts = [self.base, self.UPLOADS, tenant_id[:8]]
        if hotel_id:
            parts.append(hotel_id[:8])
        p = Path(*[str(x) for x in parts])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def document_path(
        self,
        tenant_id: str,
        hotel_id: Optional[str],
        filename: str,
        doc_id: str,
    ) -> Path:
        """
        Generate the full path for storing a document.

        Format: ``uploads/<tenant>/<hotel>/<YYYY>/<MM>/<doc_id[:8]>_<filename>``
        """
        now = datetime.now(timezone.utc)
        base = self.uploads_dir(tenant_id, hotel_id)
        year_month = base / str(now.year) / f"{now.month:02d}"
        year_month.mkdir(parents=True, exist_ok=True)
        return year_month / f"{doc_id[:8]}_{filename}"

    def assets_dir(
        self,
        tenant_id: str,
        doc_id: str,
    ) -> Path:
        """Get the assets directory for a specific document."""
        p = self.base / self.ASSETS / tenant_id[:8] / doc_id[:8]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def indexes_dir(self, tenant_id: str) -> Path:
        """Get the indexes directory for a tenant."""
        p = self.base / self.INDEXES / tenant_id[:8]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exports_dir(self, tenant_id: str) -> Path:
        """Get the exports directory for a tenant."""
        p = self.base / self.EXPORTS / tenant_id[:8]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def temp_dir(self) -> Path:
        """Get a temporary working directory."""
        p = self.base / self.TEMP
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def store_file(
        self,
        source_path: str,
        tenant_id: str,
        hotel_id: Optional[str],
        doc_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Copy a file to the organized storage path. Returns the destination."""
        src = Path(source_path)
        fname = filename or src.name
        dest = self.document_path(tenant_id, hotel_id, fname, doc_id)
        shutil.copy2(str(src), str(dest))
        logger.info("Stored: %s → %s", src.name, dest)
        return dest

    def delete_document_files(self, tenant_id: str, doc_id: str) -> int:
        """Delete all files for a document (uploads + assets). Returns count."""
        count = 0

        # Delete assets
        assets_dir = self.base / self.ASSETS / tenant_id[:8] / doc_id[:8]
        if assets_dir.exists():
            for f in assets_dir.iterdir():
                f.unlink()
                count += 1
            assets_dir.rmdir()

        return count

    def get_tenant_size(self, tenant_id: str) -> dict:
        """Calculate total disk usage for a tenant."""
        total_bytes = 0
        file_count = 0

        for subdir in (self.UPLOADS, self.ASSETS, self.INDEXES):
            tenant_dir = self.base / subdir / tenant_id[:8]
            if tenant_dir.exists():
                for f in tenant_dir.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size
                        file_count += 1

        return {
            "tenant_id": tenant_id,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "file_count": file_count,
        }


__all__ = ["StorageManager"]
