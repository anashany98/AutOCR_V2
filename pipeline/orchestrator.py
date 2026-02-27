"""
Pipeline Orchestrator — Wires all pipeline steps together.

Provides a high-level ``process_document`` method that runs the full
document processing pipeline:

  ingest → OCR → layout → visual → chunk → embed

Each step is self-contained and can also be invoked independently.
The orchestrator manages step sequencing, error handling, and job tracking.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from modules.feature_flags import flags

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    High-level orchestrator for the Document AI pipeline.

    Parameters
    ----------
    config:
        Application config dict (from config.yaml).
    db:
        Database manager.
    ocr_manager:
        Existing OCRManager instance.
    layout_manager:
        Existing LayoutManager instance.
    vl_engine:
        Optional VL engine for visual understanding.
    storage_root:
        Base directory for document storage.
    assets_root:
        Base directory for asset storage.
    device:
        Torch device for embedding model.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db: Any,
        ocr_manager: Any = None,
        layout_manager: Any = None,
        vl_engine: Any = None,
        storage_root: str = "data/uploads",
        assets_root: str = "data/assets",
        device: str = "cpu",
    ):
        self.config = config
        self.db = db

        # Lazy-import pipeline steps to avoid circular imports
        from pipeline.ingestion import IngestionStep
        from pipeline.ocr_step import OCRStep
        from pipeline.layout_step import LayoutStep
        from pipeline.visual_step import VisualStep
        from pipeline.chunking_step import ChunkingStep
        from pipeline.embedding_step import EmbeddingStep
        from pipeline.job_manager import JobManager

        self.ingestion = IngestionStep(storage_root=storage_root, db=db)
        self.ocr = OCRStep(ocr_manager=ocr_manager, db=db)
        self.layout = LayoutStep(
            layout_manager=layout_manager,
            assets_root=assets_root,
            db=db,
        )
        self.visual = VisualStep(vl_engine=vl_engine, db=db)
        self.chunking = ChunkingStep(
            max_tokens=config.get("rag", {}).get("chunk_max_tokens", 256),
            overlap_sentences=config.get("rag", {}).get("chunk_overlap_sentences", 2),
        )
        self.embedding = EmbeddingStep(
            model_name=config.get("rag", {}).get(
                "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            batch_size=config.get("rag", {}).get("embedding_batch_size", 64),
            db=db,
            device=device,
        )
        self.jobs = JobManager(db=db)

    def process_document(
        self,
        source_path: str,
        *,
        tenant_id: str,
        hotel_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        doc_type: str = "other",
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on a document.

        Returns a summary dict with processing results and timing.
        """
        t0 = time.perf_counter()
        result: Dict[str, Any] = {
            "status": "pending",
            "steps_completed": [],
            "errors": [],
        }

        try:
            # Step 1: Ingest
            ingestion_result = self.ingestion.ingest(
                source_path,
                tenant_id=tenant_id,
                hotel_id=hotel_id,
                project_id=project_id,
                owner_id=owner_id,
                doc_type=doc_type,
            )
            doc_id = ingestion_result.document_id
            result["document_id"] = doc_id
            result["steps_completed"].append("ingestion")

            # Load pages as images
            pages = self._load_pages(ingestion_result.file_path, ingestion_result.mime_type)

            # Step 2: OCR
            ocr_result = self.ocr.process(doc_id, pages)
            result["steps_completed"].append("ocr")
            result["ocr_confidence"] = ocr_result.avg_confidence
            result["language"] = ocr_result.primary_language

            # Step 3: Layout (if enabled)
            blocks_for_chunking: List[Dict[str, Any]] = []
            if flags.ENABLE_LAYOUT and self.layout.layout_mgr is not None:
                layout_result = self.layout.process(doc_id, tenant_id, pages)
                result["steps_completed"].append("layout")
                result["total_blocks"] = len(layout_result.blocks)

                # Use layout blocks for chunking
                blocks_for_chunking = [
                    {
                        "text": b.text or ocr_result.pages[b.page_number - 1].text
                        if b.page_number <= len(ocr_result.pages) else b.text,
                        "block_type": b.block_type,
                        "page_number": b.page_number,
                        "block_id": b.block_id,
                    }
                    for b in layout_result.blocks
                    if b.block_type in ("text", "title", "table", "caption", "list")
                ]
            else:
                # Fallback: use OCR pages as blocks
                blocks_for_chunking = [
                    {
                        "text": p.text,
                        "block_type": "text",
                        "page_number": p.page_number,
                        "block_id": None,
                    }
                    for p in ocr_result.pages
                    if p.text.strip()
                ]

            # Step 4: Visual Analysis (if enabled)
            visual_descriptions: List[Dict[str, Any]] = []
            if flags.ENABLE_VL and self.visual.engine is not None:
                # Gather figure assets for VL analysis
                if flags.ENABLE_LAYOUT:
                    assets = self._get_figure_assets(doc_id)
                    if assets:
                        vl_results = self.visual.process(doc_id, assets)
                        result["steps_completed"].append("visual_analysis")
                        visual_descriptions = [
                            {
                                "description": vr.description,
                                "caption": vr.caption,
                                "page_number": vr.page_number,
                                "asset_id": vr.asset_id,
                                "model_name": vr.model_name,
                            }
                            for vr in vl_results
                        ]

            # Step 5: Chunking (if RAG enabled)
            if flags.ENABLE_RAG:
                chunks = self.chunking.process(
                    document_id=doc_id,
                    tenant_id=tenant_id,
                    hotel_id=hotel_id,
                    blocks=blocks_for_chunking,
                    visual_descriptions=visual_descriptions,
                    db=self.db,
                )
                result["steps_completed"].append("chunking")
                result["total_chunks"] = len(chunks)

                # Step 6: Embedding
                chunk_dicts = [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "content_type": c.content_type,
                    }
                    for c in chunks
                ]
                emb_result = self.embedding.process(
                    document_id=doc_id,
                    tenant_id=tenant_id,
                    hotel_id=hotel_id,
                    chunks=chunk_dicts,
                )
                result["steps_completed"].append("embedding")

            # Mark complete
            self._update_document_completed(doc_id)
            result["status"] = "completed"

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            logger.error("Pipeline failed: %s", e, exc_info=True)

            # Record error on document
            doc_id = result.get("document_id")
            if doc_id:
                self._update_document_failed(doc_id, str(e))

        result["total_time_ms"] = int((time.perf_counter() - t0) * 1000)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pages(file_path: str, mime_type: str) -> List[Image.Image]:
        """Convert a file into a list of page images."""
        path = Path(file_path)

        if mime_type == "application/pdf":
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(str(path))
                pages = []
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    pages.append(img)
                doc.close()
                return pages
            except Exception as e:
                logger.warning("Failed to load PDF pages: %s", e)
                return []

        elif mime_type and mime_type.startswith("image/"):
            try:
                img = Image.open(str(path)).convert("RGB")
                return [img]
            except Exception as e:
                logger.warning("Failed to load image: %s", e)
                return []

        return []

    def _get_figure_assets(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch figure assets from the database."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, page_number, file_path
                    FROM document_assets
                    WHERE document_id = %s
                      AND asset_type IN ('figure', 'seal', 'signature')
                    """,
                    (document_id,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "asset_id": str(r[0]),
                        "page_number": r[1],
                        "file_path": r[2],
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def _update_document_completed(self, document_id: str) -> None:
        """Mark document as completed."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE documents SET status = 'completed', processed_at = NOW() WHERE id = %s",
                    (document_id,),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to update document status: %s", e)

    def _update_document_failed(self, document_id: str, error: str) -> None:
        """Mark document as failed."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE documents SET status = 'failed', error_message = %s WHERE id = %s",
                    (error[:500], document_id),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to update document error: %s", e)


__all__ = ["PipelineOrchestrator"]
