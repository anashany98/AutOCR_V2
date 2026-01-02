"""Tests for block processing and fallback logic."""

from types import SimpleNamespace

from PIL import Image

from modules.fusion_manager import FusionConfig, FusionManager
from postbatch_processor import process_text_blocks


class DummyOCR:
    def __init__(self):
        self.use_gpu = False
        self.primary_engine = "paddleocr"
        self.secondary_engine = "easyocr"
        self.calls = []

    def extract_block(self, image, bbox, engine="primary"):
        self.calls.append(engine)
        if engine == "primary":
            return ("primary-result", 0.3)
        return ("secondary-result", 0.9)


def test_secondary_called_when_confidence_low(tmp_path):
    pipeline = SimpleNamespace(
        ocr_manager=DummyOCR(),
        fusion_manager=FusionManager(FusionConfig()),
        recheck_threshold=0.5,
    )
    page = Image.new("RGB", (20, 20), "white")
    blocks = [
        {"id": 0, "page": 0, "type": "text", "bbox": [0, 0, 10, 10], "rotation": 0.0}
    ]
    results = process_text_blocks(pipeline, [page], blocks, logger=SimpleNamespace(debug=lambda *a, **k: None))
    assert results[0]["text"] == "secondary-result"
    assert pipeline.ocr_manager.calls == ["primary", "secondary"]
