"""Regression tests for PDF chunking and QC workflow gating.

These tests intentionally avoid importing PaddleOCR. The batch processor should
be able to process PDFs in memory-safe chunks using a stubbed pdf2image backend.
"""

from __future__ import annotations

import datetime as dt
import logging
from types import SimpleNamespace

import fitz  # PyMuPDF
from PIL import Image

import postbatch_processor as pb
from modules.db_manager import DBManager
from modules.fusion_manager import FusionConfig, FusionManager


def _make_sqlite_db(tmp_path) -> DBManager:
    db_path = tmp_path / "test.db"
    config = {
        "database": {
            "engine": "sqlite",
            "pool_size": 2,
            "sqlite": {"path": str(db_path)},
        }
    }
    return DBManager(config)


def _make_blank_pdf(tmp_path, *, pages: int) -> str:
    doc = fitz.open()
    for _ in range(int(pages)):
        doc.new_page(width=100, height=100)
    out = tmp_path / "blank.pdf"
    doc.save(str(out))
    doc.close()
    return str(out)


class _DummyOCR:
    def __init__(self):
        self.languages = ("spa",)
        self.primary_engine = "paddleocr"
        self.secondary_engine = "easyocr"
        self.calls = 0
        self.poppler_path = None

    def extract_block(self, image, bbox, engine="primary", min_confidence=None):
        self.calls += 1
        return "", 0.0


class _DummyLayoutPrefilled:
    def detect_blocks(self, file_path, pages):
        blocks = []
        for i, page in enumerate(pages):
            w, h = page.size
            blocks.append(
                {
                    "page": i,
                    "bbox": [0, 0, w, h],
                    "type": "text",
                    "rotation": 0.0,
                    "confidence": 0.9,
                    "text": f"page-{i}",
                    "text_confidence": 0.9,
                }
            )
        return blocks


class _DummyLayoutNoText:
    def detect_blocks(self, file_path, pages):
        blocks = []
        for i, page in enumerate(pages):
            w, h = page.size
            blocks.append(
                {
                    "page": i,
                    "bbox": [0, 0, w, h],
                    "type": "text",
                    "rotation": 0.0,
                    "confidence": 0.0,
                }
            )
        return blocks


def test_pdf_chunking_uses_ranges_and_global_pages(tmp_path, monkeypatch):
    pdf_path = _make_blank_pdf(tmp_path, pages=25)
    calls: list[tuple[int, int]] = []

    def _fake_convert_from_path(path, first_page=None, last_page=None, **kwargs):
        assert path == pdf_path
        assert first_page is not None and last_page is not None
        calls.append((int(first_page), int(last_page)))
        n = int(last_page) - int(first_page) + 1
        return [Image.new("RGB", (20, 20), "white") for _ in range(n)]

    monkeypatch.setattr(pb, "convert_from_path", _fake_convert_from_path)

    db = _make_sqlite_db(tmp_path)
    pipeline = SimpleNamespace(
        ocr_manager=_DummyOCR(),
        layout_manager=_DummyLayoutPrefilled(),
        table_manager=None,
        fusion_manager=FusionManager(FusionConfig()),
        vision_manager=None,
        mineru_engine=None,
        recheck_threshold=0.0,
        output_formats=[],
        save_markdown_in_db=False,
    )

    processed = tmp_path / "processed"
    failed = tmp_path / "failed"

    result = pb.process_single_file(
        pdf_path,
        pipeline,
        classifier=None,
        db=db,
        processed_folder=str(processed),
        failed_folder=str(failed),
        delete_original=False,
        ocr_enabled=True,
        classification_enabled=False,
        logger=logging.getLogger("test"),
        input_root=str(tmp_path),
        handwriting_mode=False,
        pipeline_conf={
            "ocr_pipeline": {"pdf": {"pages_per_chunk": 10}},
            "postbatch": {},
            "owner_id": 1,
            "hotel_id": 1,
        },
    )

    assert result["status"] == "OK"
    assert calls == [(1, 10), (11, 20), (21, 25)]

    doc = db.get_document(int(result["doc_id"]))
    assert doc is not None
    blocks = doc["blocks"]
    assert len(blocks) == 25
    assert [b["id"] for b in blocks] == list(range(25))
    assert sorted({int(b["page"]) for b in blocks}) == list(range(25))

    # Prefilled path must not call OCRManager.extract_block.
    assert pipeline.ocr_manager.calls == 0


def test_workflow_pending_when_visual_text_missing(tmp_path):
    # Minimal image input with empty OCR -> must be flagged as pending.
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (20, 20), "white").save(str(img_path))

    db = _make_sqlite_db(tmp_path)
    pipeline = SimpleNamespace(
        ocr_manager=_DummyOCR(),
        layout_manager=_DummyLayoutNoText(),
        table_manager=None,
        fusion_manager=FusionManager(FusionConfig()),
        vision_manager=None,
        mineru_engine=None,
        recheck_threshold=0.0,
        output_formats=[],
        save_markdown_in_db=False,
    )

    result = pb.process_single_file(
        str(img_path),
        pipeline,
        classifier=None,
        db=db,
        processed_folder=str(tmp_path / "processed"),
        failed_folder=str(tmp_path / "failed"),
        delete_original=False,
        ocr_enabled=True,
        classification_enabled=False,
        logger=logging.getLogger("test"),
        input_root=str(tmp_path),
        handwriting_mode=False,
        pipeline_conf={"postbatch": {"auto_verify": True}},
    )
    assert result["status"] == "OK"

    row = db.execute("SELECT workflow_state FROM documents WHERE id = ?", (int(result["doc_id"]),)).fetchone()
    assert row is not None
    workflow_state = row[0] if isinstance(row, (tuple, list)) else row["workflow_state"]
    assert workflow_state == "pending"
