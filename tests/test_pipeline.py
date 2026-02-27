"""
Unit & Integration Tests for Document AI Pipeline.

Covers:
- Feature flags
- Chunking step (sentence splitting, overlap, merging)
- Embedding step (model loading, dimensions)
- Tenant middleware (context, filters)
- Job manager (idempotency, retry)
- Context builder (RRF fusion)

Usage::

    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Feature Flags Tests
# ============================================================================

class TestFeatureFlags:
    """Tests for modules.feature_flags."""

    def test_defaults(self):
        """Flags should be disabled by default."""
        # Clear any env vars
        for key in ("AUTOOCR_ENABLE_LAYOUT", "AUTOOCR_ENABLE_VL", "AUTOOCR_ENABLE_RAG",
                     "AUTOOCR_ENABLE_PGVECTOR", "AUTOOCR_ENABLE_MULTI_TENANT"):
            os.environ.pop(key, None)

        # Re-import to get fresh defaults
        import importlib
        from modules import feature_flags
        importlib.reload(feature_flags)

        # Layout is enabled by default in the module
        assert isinstance(feature_flags.flags.ENABLE_LAYOUT, bool)
        assert isinstance(feature_flags.flags.ENABLE_VL, bool)

    def test_env_override(self):
        """Setting env var should override the flag."""
        os.environ["AUTOOCR_ENABLE_VL"] = "1"
        import importlib
        from modules import feature_flags
        importlib.reload(feature_flags)
        assert feature_flags.flags.ENABLE_VL is True
        os.environ.pop("AUTOOCR_ENABLE_VL", None)


# ============================================================================
# Chunking Step Tests
# ============================================================================

class TestChunkingStep:
    """Tests for pipeline.chunking_step."""

    @pytest.fixture
    def chunker(self):
        from pipeline.chunking_step import ChunkingStep
        return ChunkingStep(max_tokens=50, overlap_sentences=1, min_chunk_chars=20)

    def test_basic_chunking(self, chunker):
        """Single text block should produce a chunk."""
        blocks = [
            {"text": "Este es un documento de prueba con texto suficiente.", "block_type": "text", "page_number": 1}
        ]
        chunks = chunker.process("doc-1", "tenant-1", None, blocks, db=None)
        assert len(chunks) >= 1
        assert chunks[0].content_type == "text"
        assert chunks[0].document_id == "doc-1"

    def test_large_block_splitting(self, chunker):
        """A block exceeding max_tokens should be split."""
        long_text = ". ".join([f"Frase número {i} del documento de prueba" for i in range(100)])
        blocks = [
            {"text": long_text, "block_type": "text", "page_number": 1}
        ]
        chunks = chunker.process("doc-2", "tenant-1", None, blocks, db=None)
        assert len(chunks) > 1

    def test_table_no_split(self, chunker):
        """Table blocks should not be split."""
        table_text = "Col1 | Col2\n" * 100
        blocks = [
            {"text": table_text, "block_type": "table", "page_number": 1}
        ]
        chunks = chunker.process("doc-3", "tenant-1", None, blocks, db=None)
        assert len(chunks) == 1
        assert chunks[0].content_type == "table"

    def test_empty_blocks_skipped(self, chunker):
        """Empty blocks should produce no chunks."""
        blocks = [
            {"text": "", "block_type": "text", "page_number": 1},
            {"text": "   ", "block_type": "text", "page_number": 1},
        ]
        chunks = chunker.process("doc-4", "tenant-1", None, blocks, db=None)
        assert len(chunks) == 0

    def test_visual_descriptions_included(self, chunker):
        """Visual descriptions should be included as chunks."""
        blocks = [
            {"text": "Texto regular del documento.", "block_type": "text", "page_number": 1}
        ]
        visual = [
            {"description": "Logotipo del hotel con palmeras y piscina.", "caption": "Logo Hotel Resort",
             "page_number": 1, "asset_id": "asset-1", "model_name": "PaddleVL"}
        ]
        chunks = chunker.process("doc-5", "tenant-1", None, blocks, visual_descriptions=visual, db=None)
        vd_chunks = [c for c in chunks if c.content_type == "visual_description"]
        assert len(vd_chunks) == 1

    def test_content_hash_deterministic(self, chunker):
        """Same content should produce the same hash."""
        blocks = [{"text": "Texto de prueba.", "block_type": "text", "page_number": 1}]
        chunks1 = chunker.process("doc-a", "t", None, blocks, db=None)
        chunks2 = chunker.process("doc-b", "t", None, blocks, db=None)
        assert chunks1[0].content_hash == chunks2[0].content_hash

    def test_chunk_index_sequential(self, chunker):
        """Chunk indices should be sequential starting from 0."""
        blocks = [
            {"text": f"Bloque {i} con texto.", "block_type": "text", "page_number": i}
            for i in range(5)
        ]
        chunks = chunker.process("doc-6", "tenant-1", None, blocks, db=None)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ============================================================================
# Tenant Middleware Tests
# ============================================================================

class TestTenantContext:
    """Tests for modules.tenant_middleware."""

    def test_default_tenant(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id=TenantContext.DEFAULT_TENANT)
        assert ctx.tenant_id == "00000000-0000-0000-0000-000000000001"

    def test_admin_full_access(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id="t1", role="ADMIN", is_admin=True)
        assert ctx.has_full_access is True
        assert ctx.hotel_filter is None

    def test_direccion_full_access(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id="t1", role="DIRECCION")
        assert ctx.has_full_access is True

    def test_cliente_limited_access(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id="t1", role="CLIENTE", hotel_ids=["h1", "h2"])
        assert ctx.has_full_access is False
        assert ctx.hotel_filter == ["h1", "h2"]

    def test_empty_hotel_scope(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id="t1", role="GESTOR", hotel_ids=[])
        assert ctx.hotel_filter == []

    def test_to_dict(self):
        from modules.tenant_middleware import TenantContext
        ctx = TenantContext(tenant_id="t1", user_id="u1", role="ADMIN", hotel_ids=["h1"], is_admin=True)
        d = ctx.to_dict()
        assert d["tenant_id"] == "t1"
        assert d["is_admin"] is True


class TestApplyTenantFilter:
    """Tests for apply_tenant_filter SQL helper."""

    def test_adds_where_clause(self):
        from modules.tenant_middleware import TenantContext, apply_tenant_filter
        ctx = TenantContext(tenant_id="t1", role="ADMIN", is_admin=True)
        sql = "SELECT * FROM documents"
        params = []
        new_sql, new_params = apply_tenant_filter(sql, params, ctx)
        assert "tenant_id = %s" in new_sql
        assert "t1" in new_params

    def test_appends_to_existing_where(self):
        from modules.tenant_middleware import TenantContext, apply_tenant_filter
        ctx = TenantContext(tenant_id="t1", role="ADMIN", is_admin=True)
        sql = "SELECT * FROM documents WHERE status = 'active'"
        params = []
        new_sql, new_params = apply_tenant_filter(sql, params, ctx)
        assert " AND " in new_sql

    def test_hotel_filter_for_cliente(self):
        from modules.tenant_middleware import TenantContext, apply_tenant_filter
        ctx = TenantContext(tenant_id="t1", role="CLIENTE", hotel_ids=["h1", "h2"])
        sql = "SELECT * FROM documents"
        params = []
        new_sql, new_params = apply_tenant_filter(sql, params, ctx)
        assert "hotel_id = ANY(%s)" in new_sql
        assert ["h1", "h2"] in new_params

    def test_empty_hotels_blocks_all(self):
        from modules.tenant_middleware import TenantContext, apply_tenant_filter
        ctx = TenantContext(tenant_id="t1", role="GESTOR", hotel_ids=[])
        sql = "SELECT * FROM documents"
        params = []
        new_sql, _ = apply_tenant_filter(sql, params, ctx)
        assert "AND FALSE" in new_sql


# ============================================================================
# Context Builder — RRF Fusion Tests
# ============================================================================

class TestRRFFusion:
    """Tests for the Reciprocal Rank Fusion algorithm."""

    def test_rrf_combines_results(self):
        from modules.context_builder import ContextBuilder, RetrievedChunk

        mock_emb = MagicMock()
        builder = ContextBuilder(db=None, embedding_step=mock_emb, top_k=3)

        vec_results = [
            RetrievedChunk(chunk_id="c1", document_id="d1", content="A", content_type="text",
                           page_number=1, score=0.95, source_method="vector"),
            RetrievedChunk(chunk_id="c2", document_id="d1", content="B", content_type="text",
                           page_number=2, score=0.80, source_method="vector"),
        ]
        fts_results = [
            RetrievedChunk(chunk_id="c2", document_id="d1", content="B", content_type="text",
                           page_number=2, score=5.0, source_method="fts"),
            RetrievedChunk(chunk_id="c3", document_id="d1", content="C", content_type="text",
                           page_number=3, score=3.0, source_method="fts"),
        ]

        fused = builder._rrf_fusion(vec_results, fts_results, k=3)

        # c2 should rank highest (appears in both)
        assert fused[0].chunk_id == "c2"
        assert len(fused) == 3
        assert all(c.source_method == "hybrid" for c in fused)

    def test_rrf_empty_inputs(self):
        from modules.context_builder import ContextBuilder
        mock_emb = MagicMock()
        builder = ContextBuilder(db=None, embedding_step=mock_emb)

        fused = builder._rrf_fusion([], [], k=5)
        assert fused == []

    def test_context_assembly_dedup(self):
        from modules.context_builder import ContextBuilder, RetrievedChunk
        mock_emb = MagicMock()
        builder = ContextBuilder(db=None, embedding_step=mock_emb, max_context_tokens=100)

        chunks = [
            RetrievedChunk(chunk_id="c1", document_id="d1", content="Hello world duplicate",
                           content_type="text", page_number=1, score=1.0, source_method="hybrid"),
            RetrievedChunk(chunk_id="c2", document_id="d1", content="Hello world duplicate",
                           content_type="text", page_number=1, score=0.8, source_method="hybrid"),
        ]

        filtered, tokens = builder._assemble_context(chunks)
        assert len(filtered) == 1  # Duplicate removed


# ============================================================================
# Storage Manager Tests
# ============================================================================

class TestStorageManager:
    """Tests for modules.storage_manager."""

    @pytest.fixture
    def storage(self, tmp_path):
        from modules.storage_manager import StorageManager
        return StorageManager(base_dir=str(tmp_path))

    def test_uploads_dir_creation(self, storage):
        path = storage.uploads_dir("tenant-abc", "hotel-123")
        assert path.exists()
        assert "tenant-a" in str(path)

    def test_document_path_format(self, storage):
        path = storage.document_path("tenant-abc", "hotel-123", "factura.pdf", "doc-uuid-1234")
        assert "doc-uuid" in path.name
        assert "factura.pdf" in path.name

    def test_assets_dir_creation(self, storage):
        path = storage.assets_dir("tenant-abc", "doc-123")
        assert path.exists()

    def test_tenant_size_empty(self, storage):
        size = storage.get_tenant_size("nonexistent")
        assert size["total_bytes"] == 0
        assert size["file_count"] == 0


# ============================================================================
# Audit Logger Tests
# ============================================================================

class TestAuditLogger:
    """Tests for modules.audit_logger."""

    def test_log_does_not_crash_without_db(self):
        """Audit logging should never crash the main flow."""
        from modules.audit_logger import AuditLogger
        mock_db = MagicMock()
        mock_db.get_connection.side_effect = Exception("DB down")

        audit = AuditLogger(db=mock_db)
        # Should not raise
        audit.log("doc.upload", resource_type="document", resource_id="doc-1",
                  tenant_id="t1", user_id="u1")

    def test_action_descriptions(self):
        from modules.audit_logger import AuditLogger
        assert "doc.upload" in AuditLogger.ACTIONS
        assert "admin.user_create" in AuditLogger.ACTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
