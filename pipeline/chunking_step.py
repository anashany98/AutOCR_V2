"""
Pipeline Step E — Layout-Aware Chunking.

Splits document content into semantically meaningful chunks that respect
layout boundaries (paragraphs, sections, tables, captions).  Each chunk
includes source metadata for citation.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single RAG-ready chunk."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    content_type: str  # text, table, caption, visual_description
    token_count: int
    char_count: int
    content_hash: str
    page_number: Optional[int] = None
    block_id: Optional[str] = None
    asset_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingStep:
    """
    Layout-aware text chunking for RAG.

    Strategy:
    1. Split text by block/page boundaries (respecting layout paragraphs).
    2. If a block exceeds ``max_tokens``, split at sentence boundaries.
    3. Apply ``overlap_sentences`` between adjacent chunks within the same block.
    4. Table and visual descriptions become their own chunks (no splitting).

    Parameters
    ----------
    max_tokens:
        Maximum tokens per chunk (approximate, counted by whitespace splits).
    overlap_sentences:
        Number of sentences to overlap between adjacent chunks.
    min_chunk_chars:
        Minimum character count for a chunk (smaller ones are merged up).
    """

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_sentences: int = 2,
        min_chunk_chars: int = 50,
    ):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_chunk_chars = min_chunk_chars

    def process(
        self,
        document_id: str,
        tenant_id: str,
        hotel_id: Optional[str],
        blocks: List[Dict[str, Any]],
        visual_descriptions: Optional[List[Dict[str, Any]]] = None,
        db: Any = None,
    ) -> List[Chunk]:
        """
        Chunk a document's blocks into RAG units.

        Parameters
        ----------
        document_id:
            UUID of the document.
        tenant_id / hotel_id:
            Scope for the chunks.
        blocks:
            List of layout blocks with keys: ``text``, ``block_type``,
            ``page_number``, ``block_id``.
        visual_descriptions:
            Optional list of visual analysis results to include as chunks.
        db:
            Database connection for persisting chunks.

        Returns
        -------
        List of Chunk objects.
        """
        chunks: List[Chunk] = []
        chunk_index = 0

        # 1. Process text blocks
        for block in blocks:
            text = block.get("text", "").strip()
            btype = block.get("block_type", "text")
            page = block.get("page_number")
            block_id = block.get("block_id")

            if not text:
                continue

            if btype == "table":
                # Tables become single chunks (preserving structure)
                chunk = self._make_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=text,
                    content_type="table",
                    page_number=page,
                    block_id=block_id,
                    metadata={"source": "table_block"},
                )
                chunks.append(chunk)
                chunk_index += 1

            elif len(text.split()) > self.max_tokens:
                # Large text blocks: split at sentence boundaries with overlap
                sub_chunks = self._split_with_overlap(text)
                for sc in sub_chunks:
                    chunk = self._make_chunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=sc,
                        content_type="text",
                        page_number=page,
                        block_id=block_id,
                        metadata={"source": btype},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            else:
                # Normal text block
                chunk = self._make_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=text,
                    content_type="text",
                    page_number=page,
                    block_id=block_id,
                    metadata={"source": btype},
                )
                chunks.append(chunk)
                chunk_index += 1

        # 2. Process visual descriptions
        for vd in (visual_descriptions or []):
            desc = vd.get("description", "").strip()
            caption = vd.get("caption", "").strip()
            content = f"{caption}\n{desc}" if caption and desc else (desc or caption)

            if not content or len(content) < self.min_chunk_chars:
                continue

            chunk = self._make_chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                content_type="visual_description",
                page_number=vd.get("page_number"),
                asset_id=vd.get("asset_id"),
                metadata={"source": "visual_analysis", "model": vd.get("model_name", "")},
            )
            chunks.append(chunk)
            chunk_index += 1

        # 3. Merge small adjacent chunks
        chunks = self._merge_small_chunks(chunks)

        # 4. Persist
        if db is not None:
            self._store_chunks(chunks, tenant_id, hotel_id, db)

        logger.info(
            "Chunking complete: %s — %d chunks from %d blocks + %d visual descriptions",
            document_id,
            len(chunks),
            len(blocks),
            len(visual_descriptions or []),
        )

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        content_type: str,
        page_number: Optional[int] = None,
        block_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Create a Chunk object with computed hash and token count."""
        content_hash = hashlib.sha256(content.strip().lower().encode()).hexdigest()
        token_count = len(content.split())

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            content_type=content_type,
            token_count=token_count,
            char_count=len(content),
            content_hash=content_hash,
            page_number=page_number,
            block_id=block_id,
            asset_id=asset_id,
            metadata=metadata or {},
        )

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text at sentence boundaries with overlap."""
        # Simple sentence splitting (handles . ! ? followed by space/newline)
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent.split())
            if current_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                # Keep overlap
                if self.overlap_sentences > 0:
                    current = current[-self.overlap_sentences :]
                    current_tokens = sum(len(s.split()) for s in current)
                else:
                    current = []
                    current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge consecutive small chunks of the same type on the same page."""
        if len(chunks) <= 1:
            return chunks

        merged: List[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = merged[-1]
            curr = chunks[i]

            can_merge = (
                prev.char_count < self.min_chunk_chars
                and prev.content_type == curr.content_type
                and prev.page_number == curr.page_number
            )

            if can_merge:
                # Merge current into previous
                new_content = prev.content + "\n" + curr.content
                merged[-1] = self._make_chunk(
                    document_id=prev.document_id,
                    chunk_index=prev.chunk_index,
                    content=new_content,
                    content_type=prev.content_type,
                    page_number=prev.page_number,
                    block_id=prev.block_id,
                    metadata={**prev.metadata, "merged": True},
                )
            else:
                merged.append(curr)

        return merged

    def _store_chunks(
        self,
        chunks: List[Chunk],
        tenant_id: str,
        hotel_id: Optional[str],
        db: Any,
    ) -> None:
        """Persist chunks to the database."""
        try:
            import json

            with db.get_connection() as conn:
                cursor = conn.cursor()
                for c in chunks:
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, tenant_id, hotel_id,
                            chunk_index, content, content_type,
                            token_count, char_count, content_hash,
                            page_number, block_id, asset_id, metadata
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            c.chunk_id,
                            c.document_id,
                            tenant_id,
                            hotel_id,
                            c.chunk_index,
                            c.content,
                            c.content_type,
                            c.token_count,
                            c.char_count,
                            c.content_hash,
                            c.page_number,
                            c.block_id,
                            c.asset_id,
                            json.dumps(c.metadata),
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.error("Failed to store chunks: %s", e, exc_info=True)
            raise


__all__ = ["ChunkingStep", "Chunk"]
