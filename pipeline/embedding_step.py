"""
Pipeline Step F — Embedding Generation.

Generates vector embeddings for chunks using sentence-transformers and
stores them in PostgreSQL with pgvector.

The default model is ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dim),
chosen for multilingual support (Spanish + English).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default model — multilingual 384-dim
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIM = 384
DEFAULT_BATCH_SIZE = 64


@dataclass
class EmbeddingResult:
    """Result of embedding a set of chunks."""

    document_id: str
    num_chunks: int
    model_name: str
    total_processing_time_ms: int = 0


class EmbeddingStep:
    """
    Generate and store chunk embeddings using sentence-transformers + pgvector.

    Parameters
    ----------
    model_name:
        Sentence-transformer model name.
    batch_size:
        Number of chunks embedded per batch call.
    db:
        Database connection (PostgreSQL with pgvector).
    device:
        Torch device (``cpu``, ``cuda``, ``cuda:0``, etc.)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        db: Any = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.db = db
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(
                    "Loaded embedding model: %s on %s", self.model_name, self.device
                )
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                raise
        return self._model

    def process(
        self,
        document_id: str,
        tenant_id: str,
        hotel_id: Optional[str],
        chunks: List[Dict[str, Any]],
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of chunks and store in pgvector.

        Parameters
        ----------
        document_id:
            UUID of the document.
        tenant_id / hotel_id:
            Scope for the embeddings (stored for filtered vector search).
        chunks:
            List of dicts with keys: ``chunk_id``, ``content``, ``content_type``.

        Returns
        -------
        EmbeddingResult summary.
        """
        if not chunks:
            return EmbeddingResult(
                document_id=document_id,
                num_chunks=0,
                model_name=self.model_name,
            )

        t0 = time.perf_counter()

        # Extract texts for embedding
        texts = [c.get("content", "") for c in chunks]
        chunk_ids = [c.get("chunk_id") for c in chunks]
        content_types = [c.get("content_type", "text") for c in chunks]

        # Generate embeddings in batches
        all_embeddings = self._encode_batched(texts)

        # Store in pgvector
        if self.db is not None:
            self._store_embeddings(
                chunk_ids=chunk_ids,
                embeddings=all_embeddings,
                document_id=document_id,
                tenant_id=tenant_id,
                hotel_id=hotel_id,
                content_types=content_types,
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Update document status
        if self.db is not None:
            self._update_document_status(document_id)

        logger.info(
            "Embedding complete: %s — %d chunks, model=%s, %dms",
            document_id,
            len(chunks),
            self.model_name,
            elapsed_ms,
        )

        return EmbeddingResult(
            document_id=document_id,
            num_chunks=len(chunks),
            model_name=self.model_name,
            total_processing_time_ms=elapsed_ms,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string for retrieval.

        Returns
        -------
        numpy array of shape (dim,).
        """
        return self.model.encode(query, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_batched(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches, returning (N, dim) array."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _store_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        document_id: str,
        tenant_id: str,
        hotel_id: Optional[str],
        content_types: List[str],
    ) -> None:
        """Batch-insert embeddings into pgvector."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                for i, (chunk_id, emb, ctype) in enumerate(
                    zip(chunk_ids, embeddings, content_types)
                ):
                    # Convert numpy array to list for pgvector
                    vec_list = emb.tolist()

                    cursor.execute(
                        """
                        INSERT INTO embeddings (
                            chunk_id, document_id, tenant_id, hotel_id,
                            embedding, model_name, content_type
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk_id,
                            document_id,
                            tenant_id,
                            hotel_id,
                            vec_list,
                            self.model_name,
                            ctype,
                        ),
                    )

                conn.commit()
        except Exception as e:
            logger.error("Failed to store embeddings: %s", e, exc_info=True)
            raise

    def _update_document_status(self, document_id: str) -> None:
        """Mark document as fully embedded."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE documents SET status = 'embedding_complete'
                    WHERE id = %s AND status IN ('ocr_complete', 'layout_complete')
                    """,
                    (document_id,),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to update document status: %s", e)


__all__ = ["EmbeddingStep", "EmbeddingResult"]
