"""
Pipeline Step C — Layout Detection & Image Extraction.

Runs PP-Structure layout analysis on each page, detects block types
(text, title, table, figure, seal, signature), crops figure regions,
and stores blocks + assets in the database.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BlockResult:
    """A detected layout block."""

    block_id: str
    page_number: int
    block_type: str
    bbox: List[int]  # [x1, y1, x2, y2]
    text: str = ""
    confidence: float = 0.0
    rotation: float = 0.0
    reading_order: int = 0
    table_data: Optional[Dict[str, Any]] = None
    has_image: bool = False
    image_path: Optional[str] = None


@dataclass
class LayoutResult:
    """Layout detection result for an entire document."""

    document_id: str
    blocks: List[BlockResult]
    total_figures: int = 0
    total_tables: int = 0
    processing_time_ms: int = 0


class LayoutStep:
    """
    Detect layout blocks and extract cropped images.

    Parameters
    ----------
    layout_manager:
        An initialized :class:`LayoutManager` from the existing codebase.
    assets_root:
        Base directory for storing cropped image assets.
    db:
        Database connection for storing results.
    """

    FIGURE_TYPES = {"figure", "seal", "signature"}

    def __init__(
        self,
        layout_manager: Any,
        assets_root: str,
        db: Any = None,
    ):
        self.layout_mgr = layout_manager
        self.assets_root = Path(assets_root)
        self.db = db

    def process(
        self,
        document_id: str,
        tenant_id: str,
        pages: Sequence[Image.Image],
        *,
        extract_images: bool = True,
    ) -> LayoutResult:
        """
        Run layout detection on all pages.

        Parameters
        ----------
        document_id:
            UUID of the document.
        tenant_id:
            Tenant UUID for organizing stored assets.
        pages:
            PIL images of each page.
        extract_images:
            Whether to crop and store figure regions.

        Returns
        -------
        LayoutResult with all detected blocks.
        """
        all_blocks: List[BlockResult] = []
        total_figures = 0
        total_tables = 0
        t0 = time.perf_counter()

        for page_idx, page_img in enumerate(pages):
            page_num = page_idx + 1

            try:
                raw_blocks = self.layout_mgr.detect_blocks(
                    page_img if isinstance(page_img, str) else page_img,
                    pages=[page_img] if isinstance(page_img, Image.Image) else None,
                )
            except Exception as e:
                logger.warning(
                    "Layout detection failed on page %d of %s: %s",
                    page_num,
                    document_id,
                    e,
                )
                raw_blocks = []

            for order_idx, raw in enumerate(raw_blocks):
                block_id = str(uuid.uuid4())
                btype = getattr(raw, "type", "text")
                bbox = getattr(raw, "bbox", [0, 0, 0, 0])
                conf = getattr(raw, "confidence", 0.0)
                rotation = getattr(raw, "rotation", 0.0)

                block = BlockResult(
                    block_id=block_id,
                    page_number=page_num,
                    block_type=btype,
                    bbox=list(bbox),
                    confidence=conf,
                    rotation=rotation,
                    reading_order=order_idx,
                )

                # Crop and save figure images
                if extract_images and btype in self.FIGURE_TYPES:
                    img_path = self._crop_and_save(
                        page_img, bbox, document_id, tenant_id, block_id
                    )
                    if img_path:
                        block.has_image = True
                        block.image_path = img_path
                        total_figures += 1

                if btype == "table":
                    total_tables += 1

                all_blocks.append(block)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = LayoutResult(
            document_id=document_id,
            blocks=all_blocks,
            total_figures=total_figures,
            total_tables=total_tables,
            processing_time_ms=elapsed_ms,
        )

        # Persist
        if self.db is not None:
            self._store_results(result, tenant_id)

        logger.info(
            "Layout complete: %s — %d blocks (%d figures, %d tables), %dms",
            document_id,
            len(all_blocks),
            total_figures,
            total_tables,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop_and_save(
        self,
        page_img: Image.Image,
        bbox: List[int],
        document_id: str,
        tenant_id: str,
        block_id: str,
    ) -> Optional[str]:
        """Crop region from page and save as PNG asset."""
        try:
            if not isinstance(page_img, Image.Image):
                return None

            x1, y1, x2, y2 = bbox
            w, h = page_img.size
            # Clamp to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return None

            cropped = page_img.crop((x1, y1, x2, y2))

            # Store under: assets/<tenant_id[:8]>/<doc_id[:8]>/
            dest_dir = self.assets_root / tenant_id[:8] / document_id[:8]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{block_id[:8]}.png"

            cropped.save(str(dest_path), "PNG")
            return str(dest_path.relative_to(self.assets_root))

        except Exception as e:
            logger.warning("Failed to crop block %s: %s", block_id, e)
            return None

    def _store_results(self, result: LayoutResult, tenant_id: str) -> None:
        """Persist layout blocks and assets to the database."""
        try:
            import json

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                for block in result.blocks:
                    # Insert block
                    cursor.execute(
                        """
                        INSERT INTO document_blocks (
                            id, document_id, page_number, block_type,
                            bbox, confidence, rotation, reading_order,
                            text_content, table_data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            block.block_id,
                            result.document_id,
                            block.page_number,
                            block.block_type,
                            block.bbox,
                            block.confidence,
                            block.rotation,
                            block.reading_order,
                            block.text,
                            json.dumps(block.table_data) if block.table_data else None,
                        ),
                    )

                    # Insert asset record if image was extracted
                    if block.has_image and block.image_path:
                        cursor.execute(
                            """
                            INSERT INTO document_assets (
                                document_id, block_id, page_number,
                                asset_type, file_path, source_bbox
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                result.document_id,
                                block.block_id,
                                block.page_number,
                                block.block_type,
                                block.image_path,
                                block.bbox,
                            ),
                        )

                # Update document status
                cursor.execute(
                    """
                    UPDATE documents SET status = 'layout_complete'
                    WHERE id = %s AND status = 'ocr_complete'
                    """,
                    (result.document_id,),
                )

                conn.commit()

        except Exception as e:
            logger.error("Failed to store layout results: %s", e, exc_info=True)
            raise


__all__ = ["LayoutStep", "BlockResult", "LayoutResult"]
