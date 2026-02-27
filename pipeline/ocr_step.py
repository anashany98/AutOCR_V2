"""
Pipeline Step B — OCR Text Extraction.

Runs PaddleOCR on each page of a document, extracting text with language
detection and confidence scoring.  Results are stored per-page in the
``document_pages`` table.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageOCRResult:
    """OCR result for a single page."""

    page_number: int
    text: str
    confidence: float
    language: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: int = 0


@dataclass
class DocumentOCRResult:
    """Aggregated OCR result for an entire document."""

    document_id: str
    pages: List[PageOCRResult]
    full_text: str = ""
    avg_confidence: float = 0.0
    primary_language: str = "unknown"
    total_processing_time_ms: int = 0


class OCRStep:
    """
    Run PaddleOCR on document pages, producing per-page text output.

    Parameters
    ----------
    ocr_manager:
        An initialized :class:`OCRManager` instance from the existing codebase.
    db:
        Database connection for storing results.
    """

    def __init__(self, ocr_manager: Any, db: Any = None):
        self.ocr = ocr_manager
        self.db = db

    def process(
        self,
        document_id: str,
        pages: Sequence[Image.Image],
        *,
        min_confidence: float = 0.6,
    ) -> DocumentOCRResult:
        """
        Extract text from all pages.

        Parameters
        ----------
        document_id:
            UUID of the document record.
        pages:
            List of PIL images (one per page).
        min_confidence:
            Minimum confidence threshold for primary engine.

        Returns
        -------
        DocumentOCRResult
            Aggregated OCR output with per-page details.
        """
        page_results: List[PageOCRResult] = []
        all_text_parts: List[str] = []
        total_conf = 0.0
        total_ms = 0

        for idx, page_img in enumerate(pages):
            page_num = idx + 1
            t0 = time.perf_counter()

            try:
                text, lang, conf, _is_handwritten = self.ocr.extract_text(
                    page_img if isinstance(page_img, str) else page_img,
                    min_confidence=min_confidence,
                )
            except Exception as e:
                logger.warning("OCR failed on page %d of %s: %s", page_num, document_id, e)
                text, lang, conf = "", "unknown", 0.0

            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            result = PageOCRResult(
                page_number=page_num,
                text=text or "",
                confidence=conf or 0.0,
                language=lang or "unknown",
                processing_time_ms=elapsed_ms,
            )
            page_results.append(result)

            if text:
                all_text_parts.append(text)
            total_conf += result.confidence
            total_ms += elapsed_ms

        # Aggregate
        full_text = "\n\n".join(all_text_parts)
        avg_conf = total_conf / max(len(page_results), 1)

        # Determine primary language (majority vote)
        lang_counts: Dict[str, int] = {}
        for pr in page_results:
            lang_counts[pr.language] = lang_counts.get(pr.language, 0) + 1
        primary_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"

        doc_result = DocumentOCRResult(
            document_id=document_id,
            pages=page_results,
            full_text=full_text,
            avg_confidence=round(avg_conf, 3),
            primary_language=primary_lang,
            total_processing_time_ms=total_ms,
        )

        # Persist results
        if self.db is not None:
            self._store_results(doc_result)

        logger.info(
            "OCR complete: %s — %d pages, avg_conf=%.2f, lang=%s, %dms",
            document_id,
            len(page_results),
            avg_conf,
            primary_lang,
            total_ms,
        )

        return doc_result

    def _store_results(self, result: DocumentOCRResult) -> None:
        """Persist OCR results to the database."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Update document-level text
                cursor.execute(
                    """
                    UPDATE documents
                    SET text_content = %s,
                        language = %s,
                        confidence = %s,
                        status = 'ocr_complete',
                        processed_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        result.full_text,
                        result.primary_language,
                        result.avg_confidence,
                        result.document_id,
                    ),
                )

                # Insert per-page records
                for page in result.pages:
                    cursor.execute(
                        """
                        INSERT INTO document_pages (
                            document_id, page_number, text_content, confidence
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id, page_number)
                        DO UPDATE SET
                            text_content = EXCLUDED.text_content,
                            confidence = EXCLUDED.confidence
                        """,
                        (
                            result.document_id,
                            page.page_number,
                            page.text,
                            page.confidence,
                        ),
                    )

                conn.commit()
        except Exception as e:
            logger.error("Failed to store OCR results: %s", e, exc_info=True)
            raise


__all__ = ["OCRStep", "PageOCRResult", "DocumentOCRResult"]
