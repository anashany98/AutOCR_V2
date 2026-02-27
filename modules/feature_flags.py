"""
Feature flags for AutOCR Document AI platform.

These flags control the activation of new pipeline features without
breaking existing functionality.  Import this module and check flags
before calling into new subsystems.

Usage::

    from modules.feature_flags import flags
    if flags.ENABLE_LAYOUT:
        ...
"""

from __future__ import annotations

import os


class _Flags:
    """Simple feature flag container backed by env vars with sensible defaults."""

    @property
    def ENABLE_LAYOUT(self) -> bool:
        """Enable PP-Structure layout detection pipeline."""
        return os.environ.get("AUTOOCR_ENABLE_LAYOUT", "1") == "1"

    @property
    def ENABLE_VL(self) -> bool:
        """Enable PaddleOCR-VL visual understanding (async)."""
        return os.environ.get("AUTOOCR_ENABLE_VL", "0") == "1"

    @property
    def ENABLE_RAG(self) -> bool:
        """Enable RAG (chunking + embedding + hybrid retrieval)."""
        return os.environ.get("AUTOOCR_ENABLE_RAG", "1") == "1"

    @property
    def ENABLE_PGVECTOR(self) -> bool:
        """Use pgvector for embeddings (vs legacy FAISS)."""
        return os.environ.get("AUTOOCR_ENABLE_PGVECTOR", "0") == "1"

    @property
    def ENABLE_MULTI_TENANT(self) -> bool:
        """Enable multi-tenant access control enforcement."""
        return os.environ.get("AUTOOCR_ENABLE_MULTI_TENANT", "0") == "1"


flags = _Flags()

__all__ = ["flags"]
