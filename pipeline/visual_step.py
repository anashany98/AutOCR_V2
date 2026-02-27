"""
Pipeline Step D — Visual Understanding (async).

Runs PaddleOCR-VL (or Florence/Qwen-VL) on document assets to produce
natural-language descriptions, captions, and labels for figures, seals,
and other non-text elements.  Results feed into the RAG chunking step.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VisualResult:
    """Visual analysis result for a single asset."""

    asset_id: str
    document_id: str
    page_number: int
    model_name: str
    description: str
    caption: str
    labels: List[str] = field(default_factory=list)
    confidence: float = 0.0
    processing_time_ms: int = 0


class VisualStep:
    """
    Run visual language model on extracted assets.

    Parameters
    ----------
    vl_engine:
        An initialized VL engine (PaddleVLEngine, FlorenceEngine, etc.)
    db:
        Database connection for storing results.
    """

    DEFAULT_PROMPT = (
        "Describe this image from a document. "
        "Include any visible text, diagrams, charts, logos, or notable features. "
        "Be concise and factual."
    )

    def __init__(self, vl_engine: Any = None, db: Any = None):
        self.engine = vl_engine
        self.db = db

    def process(
        self,
        document_id: str,
        assets: List[Dict[str, Any]],
        *,
        prompt: Optional[str] = None,
        model_name: str = "PaddleOCR-VL-1.5",
    ) -> List[VisualResult]:
        """
        Analyze a list of document assets with the VL model.

        Parameters
        ----------
        document_id:
            UUID of the parent document.
        assets:
            List of dicts with keys: ``asset_id``, ``page_number``, ``file_path``.
        model_name:
            Name of the VL model for tracking.

        Returns
        -------
        List of VisualResult, one per asset.
        """
        if self.engine is None:
            logger.warning("No VL engine configured; skipping visual analysis")
            return []

        prompt = prompt or self.DEFAULT_PROMPT
        results: List[VisualResult] = []

        for asset in assets:
            asset_id = asset.get("asset_id", str(uuid.uuid4()))
            page_num = asset.get("page_number", 0)
            file_path = asset.get("file_path", "")

            t0 = time.perf_counter()
            try:
                # Load image
                img = Image.open(file_path).convert("RGB")

                # Run VL model
                output = self.engine.analyze(img, prompt=prompt)
                description = output if isinstance(output, str) else output.get("text", "")
                caption = description[:200] if description else ""
                labels = output.get("labels", []) if isinstance(output, dict) else []
                conf = output.get("confidence", 0.8) if isinstance(output, dict) else 0.8

            except Exception as e:
                logger.warning("VL analysis failed for asset %s: %s", asset_id, e)
                description = ""
                caption = ""
                labels = []
                conf = 0.0

            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            result = VisualResult(
                asset_id=asset_id,
                document_id=document_id,
                page_number=page_num,
                model_name=model_name,
                description=description,
                caption=caption,
                labels=labels,
                confidence=conf,
                processing_time_ms=elapsed_ms,
            )
            results.append(result)

        # Persist
        if self.db is not None and results:
            self._store_results(results)

        logger.info(
            "Visual analysis complete: %s — %d assets analyzed",
            document_id,
            len(results),
        )
        return results

    def _store_results(self, results: List[VisualResult]) -> None:
        """Persist visual analysis results."""
        try:
            import json

            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                for r in results:
                    cursor.execute(
                        """
                        INSERT INTO visual_analysis (
                            document_id, asset_id, page_number,
                            model_name, description, caption,
                            labels, confidence, processing_time_ms
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            r.document_id,
                            r.asset_id,
                            r.page_number,
                            r.model_name,
                            r.description,
                            r.caption,
                            json.dumps(r.labels),
                            r.confidence,
                            r.processing_time_ms,
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.error("Failed to store visual results: %s", e, exc_info=True)
            raise


__all__ = ["VisualStep", "VisualResult"]
