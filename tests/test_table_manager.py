"""Tests for TableManager JSON export."""

import json
from pathlib import Path

from PIL import Image

from modules.table_manager import TableManager, TableManagerConfig


class DummyTableEngine:
    def __call__(self, image):
        return [
            {
                "type": "table",
                "res": {
                    "cells": [
                        {"row": 0, "col": 0, "text": "A", "row_span": 1, "col_span": 1},
                        {"row": 0, "col": 1, "text": "B", "row_span": 1, "col_span": 1},
                        {"row": 1, "col": 0, "text": "1", "row_span": 1, "col_span": 1},
                        {"row": 1, "col": 1, "text": "2", "row_span": 1, "col_span": 1},
                    ]
                },
            }
        ]


def test_table_manager_exports_structures(tmp_path):
    output_dir = tmp_path / "tables"
    manager = TableManager(TableManagerConfig(use_gpu=False, output_dir=str(output_dir)))
    manager._engine = DummyTableEngine()
    manager._load_document_images = lambda path: [Image.new("RGB", (20, 20), "white")]

    blocks = [
        {"page": 0, "type": "table", "bbox": [0, 0, 10, 10], "rotation": 0.0}
    ]
    results = manager.extract_tables("dummy.png", blocks)
    assert len(results) == 1
    table = results[0]
    assert Path(table["csv_path"]).exists()
    assert Path(table["json_path"]).exists()
    structure = table["structure"]
    assert isinstance(structure, dict)
    assert len(structure.get("cells", [])) == 4

    with open(table["json_path"], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "cells" in data
