"""
Context Builder — Hybrid Retrieval (Vector + FTS) with access control.

This module replaces the FAISS-based RAG manager with a PostgreSQL-backed
hybrid retrieval system using pgvector for semantic search and tsvector
for lexical (keyword) search.  Results are combined using Reciprocal Rank
Fusion (RRF) for optimal retrieval quality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved during a search."""

    chunk_id: str
    document_id: str
    content: str
    content_type: str
    page_number: Optional[int]
    score: float
    source_method: str  # "vector", "fts", "hybrid"

    # Citation metadata
    filename: str = ""
    doc_type: str = ""
    hotel_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    """Result of a retrieval + context assembly."""

    query: str
    chunks: List[RetrievedChunk]
    context_text: str
    total_tokens: int
    retrieval_time_ms: int
    source_docs: List[Dict[str, Any]] = field(default_factory=list)


class ContextBuilder:
    """
    Hybrid retrieval engine combining pgvector similarity + PostgreSQL FTS.

    Parameters
    ----------
    db:
        Database connection (PostgreSQL with pgvector).
    embedding_step:
        EmbeddingStep instance for encoding queries.
    vector_weight:
        Weight for vector results in RRF fusion (0.0–1.0).
    fts_weight:
        Weight for FTS results in RRF fusion (0.0–1.0).
    top_k:
        Number of final chunks to return.
    max_context_tokens:
        Maximum total tokens in assembled context.
    rrf_k:
        RRF constant (higher = less penalty for low-ranked results).
    """

    def __init__(
        self,
        db: Any,
        embedding_step: Any,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        top_k: int = 8,
        max_context_tokens: int = 2048,
        rrf_k: int = 60,
    ):
        self.db = db
        self.embedding_step = embedding_step
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        hotel_ids: Optional[List[str]] = None,
        doc_type_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> ContextResult:
        """
        Retrieve relevant chunks using hybrid search (vector + FTS).

        Parameters
        ----------
        query:
            User question / search query.
        tenant_id:
            Required tenant scope for access control.
        hotel_ids:
            Optional hotel filter (user's authorized hotels).
        doc_type_filter:
            Optional document type filter.
        top_k:
            Override for number of results.

        Returns
        -------
        ContextResult with ranked chunks and assembled context.
        """
        k = top_k or self.top_k
        t0 = time.perf_counter()

        # 1. Vector search
        vector_results = self._vector_search(
            query,
            tenant_id=tenant_id,
            hotel_ids=hotel_ids,
            doc_type_filter=doc_type_filter,
            limit=k * 2,  # Fetch more for fusion
        )

        # 2. Full-text search
        fts_results = self._fts_search(
            query,
            tenant_id=tenant_id,
            hotel_ids=hotel_ids,
            doc_type_filter=doc_type_filter,
            limit=k * 2,
        )

        # 3. RRF Fusion
        fused = self._rrf_fusion(vector_results, fts_results, k)

        # 4. Deduplicate and enforce token limit
        final_chunks, total_tokens = self._assemble_context(fused)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # 5. Build source docs summary
        source_docs = self._extract_source_docs(final_chunks)

        # 6. Assemble context string for LLM
        context_text = self._format_context(final_chunks)

        logger.info(
            "Retrieval: query=%r, %d vector + %d fts → %d chunks (%d tokens) in %dms",
            query[:50],
            len(vector_results),
            len(fts_results),
            len(final_chunks),
            total_tokens,
            elapsed_ms,
        )

        return ContextResult(
            query=query,
            chunks=final_chunks,
            context_text=context_text,
            total_tokens=total_tokens,
            retrieval_time_ms=elapsed_ms,
            source_docs=source_docs,
        )

    # ------------------------------------------------------------------
    # Vector Search
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        tenant_id: str,
        hotel_ids: Optional[List[str]],
        doc_type_filter: Optional[str],
        limit: int,
    ) -> List[RetrievedChunk]:
        """Semantic search via pgvector cosine similarity."""
        try:
            query_vec = self.embedding_step.encode_query(query)
            vec_list = query_vec.tolist()

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Build query with access control filters
                sql = """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.content,
                        c.content_type,
                        c.page_number,
                        d.filename,
                        d.doc_type,
                        c.hotel_id,
                        1 - (e.embedding <=> %s::vector) AS similarity
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE e.tenant_id = %s
                """
                params: list = [vec_list, tenant_id]

                if hotel_ids:
                    sql += " AND e.hotel_id = ANY(%s)"
                    params.append(hotel_ids)

                if doc_type_filter:
                    sql += " AND d.doc_type = %s"
                    params.append(doc_type_filter)

                sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
                params.extend([vec_list, limit])

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                return [
                    RetrievedChunk(
                        chunk_id=str(r[0]),
                        document_id=str(r[1]),
                        content=r[2],
                        content_type=r[3],
                        page_number=r[4],
                        filename=r[5] or "",
                        doc_type=r[6] or "",
                        hotel_id=str(r[7]) if r[7] else None,
                        score=float(r[8]),
                        source_method="vector",
                    )
                    for r in rows
                ]

        except Exception as e:
            logger.error("Vector search failed: %s", e, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Full-Text Search
    # ------------------------------------------------------------------

    def _fts_search(
        self,
        query: str,
        tenant_id: str,
        hotel_ids: Optional[List[str]],
        doc_type_filter: Optional[str],
        limit: int,
    ) -> List[RetrievedChunk]:
        """Keyword search via PostgreSQL tsvector + GIN."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                sql = """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.content,
                        c.content_type,
                        c.page_number,
                        d.filename,
                        d.doc_type,
                        c.hotel_id,
                        ts_rank_cd(c.fts_vector, plainto_tsquery('spanish', %s)) AS rank
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.tenant_id = %s
                      AND c.fts_vector @@ plainto_tsquery('spanish', %s)
                """
                params: list = [query, tenant_id, query]

                if hotel_ids:
                    sql += " AND c.hotel_id = ANY(%s)"
                    params.append(hotel_ids)

                if doc_type_filter:
                    sql += " AND d.doc_type = %s"
                    params.append(doc_type_filter)

                sql += " ORDER BY rank DESC LIMIT %s"
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                return [
                    RetrievedChunk(
                        chunk_id=str(r[0]),
                        document_id=str(r[1]),
                        content=r[2],
                        content_type=r[3],
                        page_number=r[4],
                        filename=r[5] or "",
                        doc_type=r[6] or "",
                        hotel_id=str(r[7]) if r[7] else None,
                        score=float(r[8]),
                        source_method="fts",
                    )
                    for r in rows
                ]

        except Exception as e:
            logger.error("FTS search failed: %s", e, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        vector_results: List[RetrievedChunk],
        fts_results: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        """
        Combine vector and FTS results using Reciprocal Rank Fusion.

        RRF score = w1 / (k + rank_vector) + w2 / (k + rank_fts)
        """
        # Build rank maps
        vec_ranks: Dict[str, int] = {
            c.chunk_id: i + 1 for i, c in enumerate(vector_results)
        }
        fts_ranks: Dict[str, int] = {
            c.chunk_id: i + 1 for i, c in enumerate(fts_results)
        }

        # Collect all unique chunk IDs
        all_chunks: Dict[str, RetrievedChunk] = {}
        for c in vector_results:
            all_chunks[c.chunk_id] = c
        for c in fts_results:
            if c.chunk_id not in all_chunks:
                all_chunks[c.chunk_id] = c

        # Compute RRF scores
        scored: List[Tuple[str, float]] = []
        for chunk_id in all_chunks:
            vec_rank = vec_ranks.get(chunk_id, len(vector_results) + 100)
            fts_rank = fts_ranks.get(chunk_id, len(fts_results) + 100)

            rrf_score = (
                self.vector_weight / (self.rrf_k + vec_rank)
                + self.fts_weight / (self.rrf_k + fts_rank)
            )
            scored.append((chunk_id, rrf_score))

        # Sort by RRF score and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in scored[:k]]

        result = []
        for cid in top_ids:
            chunk = all_chunks[cid]
            chunk.score = dict(scored)[cid]
            chunk.source_method = "hybrid"
            result.append(chunk)

        return result

    # ------------------------------------------------------------------
    # Context Assembly
    # ------------------------------------------------------------------

    def _assemble_context(
        self, chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], int]:
        """Deduplicate and enforce token limit."""
        seen_hashes: set = set()
        filtered: List[RetrievedChunk] = []
        total_tokens = 0

        for chunk in chunks:
            # Dedup by content
            content_key = chunk.content.strip()[:200]
            if content_key in seen_hashes:
                continue
            seen_hashes.add(content_key)

            # Token budget check
            chunk_tokens = len(chunk.content.split())
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break

            filtered.append(chunk)
            total_tokens += chunk_tokens

        return filtered, total_tokens

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks into a context string for the LLM prompt."""
        if not chunks:
            return ""

        parts = []
        for i, chunk in enumerate(chunks):
            source = f"{chunk.filename}"
            if chunk.page_number:
                source += f", p.{chunk.page_number}"
            parts.append(
                f"[Fuente {i + 1}: {source}]\n{chunk.content}"
            )

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_source_docs(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract unique source documents from chunks for citations."""
        seen: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            if chunk.document_id not in seen:
                seen[chunk.document_id] = {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "doc_type": chunk.doc_type,
                    "hotel_id": chunk.hotel_id,
                    "pages": [],
                }
            if chunk.page_number and chunk.page_number not in seen[chunk.document_id]["pages"]:
                seen[chunk.document_id]["pages"].append(chunk.page_number)

        return list(seen.values())


__all__ = ["ContextBuilder", "ContextResult", "RetrievedChunk"]
