"""
AutOCR modular package consolidating reusable components.

Available submodules:

* `classifier` - keyword-based document classification helpers.
* `content_extractor` - unified content loading for multiple formats.
* `db_manager` - database access layer for SQLite/SQL Server.
* `file_utils` - file system helpers for hashing, moving and scanning.
* `fusion_manager` - fusion helpers for PaddleOCR/EasyOCR outputs.
* `inactivity_monitor` - idle detection before batch processing.
* `layout_manager` - layout detection via PaddleOCR.
* `logger_manager` - logging configuration and persistence.
* `metrics_reporter` - metrics aggregation and reporting.
* `ocr_manager` - OCR abstraction with PaddleOCR/EasyOCR.
* `table_manager` - table detection and export utilities.
* `vision_manager` - CLIP embeddings and FAISS similarity search.
"""

__all__ = [
    "classifier",
    "content_extractor",
    "db_manager",
    "file_utils",
    "fusion_manager",
    "inactivity_monitor",
    "layout_manager",
    "logger_manager",
    "metrics_reporter",
    "ocr_manager",
    "table_manager",
    "vision_manager",
]
