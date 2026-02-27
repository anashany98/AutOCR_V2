"""
AutOCR post-batch processor.

This script orchestrates the cascaded OCR + layout + table + vision pipeline for
documents produced by Epson Scan 2. It preserves backwards compatibility with
the original flows while adding GPU-enabled PaddleOCR, EasyOCR fallback,
table extraction and JSON/Markdown outputs.
"""

from __future__ import annotations

import argparse
import datetime
import fitz # PyMuPDF
import json
import logging
import os
import re
import shutil
import statistics
from modules.interpretation_manager import AdvancedInterpretationRouter
from modules.llm_client import LLMClient
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml  # type: ignore
from PIL import Image

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

from modules.classifier import DocumentClassifier
from modules.content_extractor import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    extract_content,
)
from modules.db_manager import DBManager
from modules.file_utils import (
    compute_hash,
    ensure_directories,
    list_scannable_files,
    move_file,
)
from modules.fusion_manager import FusionConfig, FusionManager
from modules.inactivity_monitor import InactivityMonitor
from modules.layout_manager import LayoutManager, LayoutManagerConfig
from modules.logger_manager import setup_logger
from modules.metrics_reporter import generate_summary_report
from modules.ocr_manager import OCRConfig, OCRManager, ocr_text_to_markdown
from modules.table_manager import TableManager, TableManagerConfig, TableResult
from modules.vision_manager import VisionManager, VisionManagerConfig
from modules.decor_advisor import DecorAdvisor
from modules.color_extractor import ColorExtractor
from modules.config_normalizer import normalize_config


TEXT_BLOCK_TYPES = {"text", "title", "other"}
BYTES_PER_GIB = 1024 ** 3


@dataclass
class PipelineComponents:
    """Container aggregating OCR pipeline managers and settings."""

    ocr_manager: OCRManager
    layout_manager: Optional[LayoutManager]
    table_manager: Optional[TableManager]
    fusion_manager: FusionManager
    vision_manager: Optional[VisionManager]
    mineru_engine: Optional[Any]  # MinerUEngine for structured document extraction
    recheck_threshold: float
    output_formats: List[str]
    save_markdown_in_db: bool


WorkerComponents = Tuple[PipelineComponents, Optional[DocumentClassifier]]
_worker_local: threading.local = threading.local()
_gpu_counter = 0
_gpu_lock = threading.Lock()

def _get_next_gpu_id(num_gpus: int) -> int:
    global _gpu_counter
    if num_gpus <= 1:
        return 0
    with _gpu_lock:
        gpu_id = _gpu_counter % num_gpus
        _gpu_counter += 1
        return gpu_id

def _get_worker_components(
    pipeline_factory: Callable[[int], PipelineComponents],
    classifier_factory: Optional[Callable[[], DocumentClassifier]],
    num_gpus: int = 1,
) -> WorkerComponents:
    components: Optional[WorkerComponents] = getattr(_worker_local, "components", None)
    if components is None:
        gpu_id = _get_next_gpu_id(num_gpus)
        pipeline = pipeline_factory(gpu_id)
        classifier = classifier_factory() if classifier_factory else None
        components = (pipeline, classifier)
        _worker_local.components = components
    return components


def _clear_worker_components() -> None:
    if hasattr(_worker_local, "components"):
        delattr(_worker_local, "components")


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
        return normalize_config(raw if isinstance(raw, dict) else {})


def resolve_path(base_dir: str, value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(base_dir) / path)


def _resolve_disk_probe_path(path: str) -> str:
    probe_path = path
    if not os.path.exists(probe_path):
        probe_path = os.path.dirname(probe_path) or "."
    return probe_path


def _safe_file_size(path: str) -> int:
    try:
        return max(0, int(os.path.getsize(path)))
    except OSError:
        return 0


def resolve_max_input_bytes_per_run(
    target_path: str,
    *,
    max_input_gb_per_run: float = 0.0,
    max_input_free_disk_ratio: float = 0.0,
) -> Tuple[int, Dict[str, int | float]]:
    """
    Resolve the effective per-run input cap in bytes.

    The cap is the minimum of:
    - fixed absolute cap (GiB), if configured
    - ratio of current free disk, if configured
    """
    fixed_bytes = int(max(0.0, float(max_input_gb_per_run or 0.0)) * BYTES_PER_GIB)
    ratio = max(0.0, min(1.0, float(max_input_free_disk_ratio or 0.0)))

    free_bytes = 0
    ratio_bytes = 0
    if ratio > 0.0:
        usage = shutil.disk_usage(_resolve_disk_probe_path(target_path))
        free_bytes = int(usage.free)
        ratio_bytes = int(free_bytes * ratio)

    candidates = [cap for cap in (fixed_bytes, ratio_bytes) if cap > 0]
    effective_bytes = min(candidates) if candidates else 0
    return effective_bytes, {
        "effective_bytes": int(effective_bytes),
        "fixed_bytes": int(fixed_bytes),
        "ratio": float(ratio),
        "ratio_bytes": int(ratio_bytes),
        "free_bytes": int(free_bytes),
    }


def plan_processing_batch(
    files: List[str],
    *,
    max_input_gb_per_run: float = 0.0,
    max_input_bytes_per_run: int = 0,
    max_files_per_run: int = 0,
    processing_order: str = "as_found",
) -> Tuple[List[str], List[str], int, int]:
    """
    Select a bounded processing batch by total input size.

    Returns:
    - selected files
    - deferred files
    - selected bytes
    - discovered bytes
    """
    if not files:
        return [], [], 0, 0

    order = str(processing_order or "as_found").strip().lower()
    indexed_items = [
        (path, _safe_file_size(path), idx)
        for idx, path in enumerate(files)
    ]

    if order == "small_first":
        indexed_items.sort(key=lambda item: (item[1], item[0]))
    elif order == "large_first":
        indexed_items.sort(key=lambda item: (-item[1], item[0]))
    else:
        order = "as_found"

    discovered_bytes = sum(size for _, size, _ in indexed_items)
    max_bytes = int(max(0, int(max_input_bytes_per_run or 0)))
    if max_bytes <= 0:
        max_bytes = int(max(0.0, float(max_input_gb_per_run or 0.0)) * BYTES_PER_GIB)
    if max_bytes <= 0:
        selected_all = [path for path, _, _ in indexed_items]
        if int(max_files_per_run or 0) > 0:
            selected_all = selected_all[: int(max_files_per_run)]
            selected_set = set(selected_all)
            deferred_all = [path for path, _, _ in indexed_items if path not in selected_set]
            selected_bytes_all = sum(size for path, size, _ in indexed_items if path in selected_set)
            return selected_all, deferred_all, selected_bytes_all, discovered_bytes
        return selected_all, [], discovered_bytes, discovered_bytes

    selected_paths: List[str] = []
    size_by_path: Dict[str, int] = {}
    selected_bytes = 0

    for path, size, _ in indexed_items:
        if not selected_paths:
            # Always take at least one file to avoid starvation with very large inputs.
            selected_paths.append(path)
            size_by_path[path] = size
            selected_bytes += size
            continue

        if selected_bytes + size <= max_bytes:
            selected_paths.append(path)
            size_by_path[path] = size
            selected_bytes += size

    max_files = int(max_files_per_run or 0)
    if max_files > 0 and len(selected_paths) > max_files:
        selected_paths = selected_paths[:max_files]
        selected_bytes = sum(size_by_path[path] for path in selected_paths)

    selected_set = set(selected_paths)
    deferred_paths = [path for path, _, _ in indexed_items if path not in selected_set]
    return selected_paths, deferred_paths, selected_bytes, discovered_bytes


def check_disk_headroom(
    target_path: str,
    *,
    min_free_gb: float = 0.0,
    required_bytes: int = 0,
) -> Tuple[bool, Dict[str, int]]:
    """Return whether free disk space satisfies configured headroom requirements."""
    usage = shutil.disk_usage(_resolve_disk_probe_path(target_path))
    free_bytes = int(usage.free)
    min_free_bytes = int(max(0.0, float(min_free_gb or 0.0)) * BYTES_PER_GIB)
    required_bytes = max(0, int(required_bytes))
    required_min = max(min_free_bytes, required_bytes)

    return (
        free_bytes >= required_min,
        {
            "free_bytes": free_bytes,
            "min_free_bytes": min_free_bytes,
            "required_bytes": required_bytes,
            "required_min_bytes": required_min,
        },
    )


def is_visual_document(file_path: str) -> bool:
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in PDF_EXTENSIONS


def load_document_pages(file_path: str, poppler_path: Optional[str] = None) -> List[Image.Image]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        if convert_from_path is None:
            raise RuntimeError(
                "pdf2image is required for PDF processing but is not installed"
            )
        kwargs = {}
        if poppler_path and os.path.exists(poppler_path):
            kwargs["poppler_path"] = poppler_path
             
        pages = convert_from_path(file_path, **kwargs)
        return [page.convert("RGB") for page in pages]

    with Image.open(file_path) as image:
        frames: List[Image.Image] = []
        frame_count = getattr(image, "n_frames", 1)
        for frame_index in range(frame_count):
            try:
                image.seek(frame_index)
            except EOFError:
                break
            frames.append(image.convert("RGB").copy())
        if not frames:
            frames.append(image.convert("RGB").copy())
        return frames

def _validate_poppler_path(poppler_path: Optional[str], logger: logging.Logger) -> Optional[str]:
    if not poppler_path:
        return None
    if os.path.exists(poppler_path):
        return poppler_path
    logger.warning("Poppler path does not exist; ignoring: %s", poppler_path)
    return None


def _get_pdf_page_count(file_path: str, logger: logging.Logger) -> int:
    try:
        with fitz.open(file_path) as doc:
            return int(len(doc))
    except Exception as exc:
        raise RuntimeError(f"Unable to open PDF for page count: {exc}") from exc


def iter_pdf_page_chunks(
    file_path: str,
    *,
    logger: logging.Logger,
    pages_per_chunk: int = 8,
    poppler_path: Optional[str] = None,
    total_pages: Optional[int] = None,
) -> Iterable[Tuple[int, List[Image.Image]]]:
    """
    Yield (page_offset, [PIL pages]) for a PDF without loading it entirely in RAM.

    page_offset is 0-based and must be added to the local page indices of the chunk.
    """

    if convert_from_path is None:
        raise RuntimeError("pdf2image is required for PDF processing but is not installed")

    pages_per_chunk = int(pages_per_chunk or 0)
    if pages_per_chunk < 1:
        pages_per_chunk = 8

    poppler_path = _validate_poppler_path(poppler_path, logger)

    if total_pages is None:
        total_pages = _get_pdf_page_count(file_path, logger)
    total_pages = int(total_pages or 0)
    if total_pages <= 0:
        raise RuntimeError("PDF has zero pages")

    kwargs = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    # pdf2image first_page/last_page are 1-based inclusive.
    for first in range(1, total_pages + 1, pages_per_chunk):
        last = min(first + pages_per_chunk - 1, total_pages)
        try:
            chunk = convert_from_path(file_path, first_page=first, last_page=last, **kwargs)
        except Exception as exc:
            logger.error("PDF rasterization failed for pages %s-%s: %s", first, last, exc, exc_info=True)
            raise

        pages = [page.convert("RGB") for page in chunk]
        yield (first - 1), pages


import json
import threading

class RecoveryManager:
    """
    Manages recovery from crashes by tracking in-flight files.
    If the process terminates unexpectedly, files remaining in the 'processing' list
    are considered potential causes of the crash (Poison Pills) and are quarantined
    on the next run.
    """
    def __init__(self, recovery_file: str, quarantine_folder: str, logger: logging.Logger):
        self.recovery_file = recovery_file
        self.quarantine_folder = quarantine_folder
        self.logger = logger
        self.lock = threading.Lock()
        self.in_flight: set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.recovery_file):
            try:
                with open(self.recovery_file, 'r', encoding='utf-8') as f:
                    self.in_flight = set(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load recovery file: {e}")
                self.in_flight = set()

    def _save(self):
        try:
            with open(self.recovery_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.in_flight), f)
        except Exception as e:
            self.logger.warning(f"Failed to save recovery file: {e}")

    def register_start(self, file_path: str):
        with self.lock:
            self.in_flight.add(os.path.abspath(file_path))
            self._save()

    def register_complete(self, file_path: str):
        with self.lock:
            path = os.path.abspath(file_path)
            if path in self.in_flight:
                self.in_flight.remove(path)
                self._save()

    def recover(self):
        """Check for files left over from a crash and move them to quarantine."""
        if not self.in_flight:
            return

        self.logger.warning(f"Found {len(self.in_flight)} files from previous crashed run. Moving to quarantine.")
        os.makedirs(self.quarantine_folder, exist_ok=True)
        
        for file_path in list(self.in_flight):
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                dest = os.path.join(self.quarantine_folder, file_name)
                try:
                    # Move logic similar to main move_file
                    import shutil
                    shutil.move(file_path, dest)
                    self.logger.error(f"QUARANTINED suspected poison pill: {file_name}")
                except Exception as e:
                    self.logger.error(f"Failed to quarantine {file_name}: {e}")
            
            # Remove from tracking once handled (or if file is gone)
            with self.lock:
                self.in_flight.discard(file_path)
        
        self._save() # Should be empty now
def fallback_blocks(pages: Iterable[Image.Image]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        width, height = page.size
        blocks.append(
            {
                "id": page_index,
                "bbox": [0, 0, width, height],
                "type": "text",
                "page": page_index,
                "rotation": 0.0,
                "confidence": 0.0,
            }
        )
    return blocks


def try_extract_native_pdf(
    file_path: str, logger: logging.Logger, text_threshold: int = 50
) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to extract text directly from a PDF using PyMuPDF (fitz).
    Returns a list of block dictionaries if the document appears to be native
    (sufficient text density). Returns None if it looks like a scan.
    """
    try:
        with fitz.open(file_path) as doc:
            total_text_len = 0
            total_pages = len(doc)
        
            if total_pages == 0:
                return None

            # 1. Quick Density Check (Check first few pages)
            check_pages = min(3, total_pages)
            for i in range(check_pages):
                total_text_len += len(doc[i].get_text())
            
            avg_chars = total_text_len / check_pages
            if avg_chars < text_threshold:
                logger.info(f"PDF text density low ({avg_chars:.1f} chars/page). Treating as SCAN.")
                return None

            # 2. Extract Blocks
            logger.info(f"PDF appears natively digital ({avg_chars:.1f} chars/page). Extracting text directly.")
            output_blocks = []
            block_id_counter = 0

            for page_index, page in enumerate(doc):
                # get_text("dict") returns blocks with bbox and text spans
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:  # 0 is text
                        continue
                    
                    # Extract text lines
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "") + " "
                        block_text += "\n"
                    
                    block_text = block_text.strip()
                    if not block_text:
                        continue

                    output_blocks.append({
                        "id": block_id_counter,
                        "page": page_index,
                        "bbox": block.get("bbox"), # [x0, y0, x1, y1]
                        "type": "text",
                        "rotation": 0.0,
                        "text": block_text,
                        "confidence": 0.99, # Native text is high confidence
                        "primary_confidence": 0.99,
                        "secondary_confidence": 0.0
                    })
                    block_id_counter += 1
                    
            return output_blocks

    except Exception as e:
        logger.warning(f"Native extraction failed: {e}. Falling back to OCR.")
        return None


def sort_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_blocks = sorted(
        blocks,
        key=lambda blk: (
            int(blk.get("page", 0)),
            blk.get("bbox", [0, 0, 0, 0])[1],
            blk.get("bbox", [0, 0, 0, 0])[0],
        ),
    )
    for index, block in enumerate(sorted_blocks):
        block.setdefault("id", index)
    return sorted_blocks


def process_text_blocks(
    pipeline: PipelineComponents,
    pages: List[Image.Image],
    blocks: List[Dict[str, Any]],
    logger: logging.Logger,
    handwriting_mode: bool = False,
) -> List[Dict[str, Any]]:
    text_blocks = [
        block for block in blocks if str(block.get("type", "")).lower() in TEXT_BLOCK_TYPES
    ]
    if not text_blocks:
        text_blocks = fallback_blocks(pages)

    def _run(block: Dict[str, Any]) -> Dict[str, Any]:
        page_index = int(block.get("page", 0))
        if page_index >= len(pages):
            logger.debug("Block %s references missing page %s", block.get("id"), page_index)
            return {
                "id": block.get("id"),
                "page": page_index,
                "bbox": block.get("bbox"),
                "type": block.get("type"),
                "text": "",
                "confidence": 0.0,
                "primary_confidence": 0.0,
                "secondary_confidence": 0.0,
            }

        image = pages[page_index]

        min_conf = 0.4 if handwriting_mode else None
        primary_text, primary_conf = pipeline.ocr_manager.extract_block(
            image, block.get("bbox", []), engine="primary", min_confidence=min_conf
        )

        secondary_text = ""
        secondary_conf = 0.0
        if (
            pipeline.recheck_threshold > 0
            and primary_conf < pipeline.recheck_threshold
            and pipeline.ocr_manager.secondary_engine
        ):
            secondary_text, secondary_conf = pipeline.ocr_manager.extract_block(
                image, block.get("bbox", []), engine="secondary"
            )

        fused_text, fused_conf = pipeline.fusion_manager.fuse(
            primary_text,
            primary_conf,
            secondary_text,
            secondary_conf,
            {
                "type": block.get("type"),
                "primary_engine": pipeline.ocr_manager.primary_engine,
                "secondary_engine": pipeline.ocr_manager.secondary_engine,
            },
        )

        return {
            "id": block.get("id"),
            "page": page_index,
            "bbox": block.get("bbox"),
            "type": block.get("type"),
            "text": fused_text,
            "confidence": fused_conf,
            "primary_confidence": primary_conf,
                "secondary_confidence": secondary_conf,
            }

    return [_run(block) for block in text_blocks]


def process_layout_blocks(
    pipeline: PipelineComponents,
    pages: List[Image.Image],
    layout_blocks: List[Dict[str, Any]],
    logger: logging.Logger,
    *,
    handwriting_mode: bool = False,
    page_offset: int = 0,
    block_id_start: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str], List[float], int, List[Dict[str, Any]]]:
    """
    Convert layout blocks + page images into merged block outputs and aggregated stats.

    Supports two modes:
    - Fast path: when layout blocks already contain OCR text (PPStructure `res` captured by LayoutManager).
    - Fallback path: OCR each text block crop via OCRManager.

    Returns:
    - block_outputs (global ids/pages applied)
    - texts_join
    - confidences
    - next_block_id
    - sorted layout_blocks (local pages preserved; ids local)
    """

    sorted_blocks = sort_blocks(layout_blocks)
    block_outputs: List[Dict[str, Any]] = []
    texts_join: List[str] = []
    confidences: List[float] = []

    next_id = int(block_id_start)

    has_prefilled_text = any(
        (str(b.get("text", "") or "").strip())
        for b in sorted_blocks
        if str(b.get("type", "")).lower() in TEXT_BLOCK_TYPES
    )

    if has_prefilled_text:
        for block in sorted_blocks:
            block_type = str(block.get("type", "")).lower()
            page_index_local = int(block.get("page", 0))
            page_index_global = page_offset + page_index_local

            if block_type not in TEXT_BLOCK_TYPES:
                block_outputs.append(
                    {
                        "id": next_id,
                        "page": page_index_global,
                        "bbox": block.get("bbox"),
                        "type": block.get("type"),
                        "rotation": block.get("rotation", 0.0),
                        "text": "",
                        "confidence": 0.0,
                        "primary_confidence": 0.0,
                        "secondary_confidence": 0.0,
                    }
                )
                next_id += 1
                continue

            primary_text = str(block.get("text", "") or "")
            primary_conf = float(
                block.get("text_confidence")
                or block.get("confidence", 0.0)
                or 0.0
            )
            if not primary_text.strip():
                # If a text block is present but contains no OCR text, force recheck.
                primary_conf = 0.0

            secondary_text = ""
            secondary_conf = 0.0
            if (
                pipeline.recheck_threshold > 0
                and primary_conf < pipeline.recheck_threshold
                and pipeline.ocr_manager.secondary_engine
                and 0 <= page_index_local < len(pages)
            ):
                secondary_text, secondary_conf = pipeline.ocr_manager.extract_block(
                    pages[page_index_local], block.get("bbox", []), engine="secondary"
                )

            fused_text, fused_conf = pipeline.fusion_manager.fuse(
                primary_text,
                primary_conf,
                secondary_text,
                secondary_conf,
                {
                    "type": block.get("type"),
                    "primary_engine": "paddleocr_ppstructure",
                    "secondary_engine": pipeline.ocr_manager.secondary_engine,
                },
            )

            merged_block = {
                "id": next_id,
                "page": page_index_global,
                "bbox": block.get("bbox"),
                "type": block.get("type"),
                "rotation": block.get("rotation", 0.0),
                "text": fused_text,
                "confidence": fused_conf,
                "primary_confidence": primary_conf,
                "secondary_confidence": secondary_conf,
            }
            block_outputs.append(merged_block)
            next_id += 1

            if fused_text:
                texts_join.append(fused_text)
                confidences.append(float(fused_conf))

        return block_outputs, texts_join, confidences, next_id, sorted_blocks

    # Fallback: OCR crops block-by-block.
    text_results = process_text_blocks(
        pipeline,
        pages,
        sorted_blocks,
        logger,
        handwriting_mode=handwriting_mode,
    )
    results_by_id = {
        result.get("id"): result for result in text_results if result.get("id") is not None
    }

    for block in sorted_blocks:
        page_index_local = int(block.get("page", 0))
        page_index_global = page_offset + page_index_local
        result = results_by_id.get(block.get("id"))

        if result:
            merged_block = {
                "id": next_id,
                "page": page_index_global,
                "bbox": block.get("bbox"),
                "type": block.get("type"),
                "rotation": block.get("rotation", 0.0),
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "primary_confidence": result.get("primary_confidence", 0.0),
                "secondary_confidence": result.get("secondary_confidence", 0.0),
            }
            block_outputs.append(merged_block)
            text_value = merged_block["text"]
            if text_value:
                texts_join.append(text_value)
                confidences.append(float(merged_block["confidence"]))
        else:
            block_outputs.append(
                {
                    "id": next_id,
                    "page": page_index_global,
                    "bbox": block.get("bbox"),
                    "type": block.get("type"),
                    "rotation": block.get("rotation", 0.0),
                    "text": "",
                    "confidence": 0.0,
                    "primary_confidence": 0.0,
                    "secondary_confidence": 0.0,
                }
            )

        next_id += 1

    return block_outputs, texts_join, confidences, next_id, sorted_blocks


def save_additional_outputs(
    dest_path: str,
    summary: Dict[str, Any],
    markdown_text: str,
    pipeline: PipelineComponents,
) -> None:
    base_path = Path(dest_path)
    summary.setdefault("path", dest_path)

    if "json" in pipeline.output_formats:
        # Never overwrite the source file (e.g. if the input itself is `.json`).
        # Also avoid naming collisions with input scans when `.json` is included in `postbatch.file_types`.
        json_path = base_path.with_name(base_path.name + ".ocr.json")
        summary.setdefault("summary_json_path", str(json_path))
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    if markdown_text and "markdown" in pipeline.output_formats:
        markdown_path = base_path.with_name(base_path.name + ".ocr.md")
        summary.setdefault("markdown_path", str(markdown_path))
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(markdown_text)


def initialise_pipeline(
    config: dict,
    project_root: str,
    logger: logging.Logger,
    gpu_id: int = 0,
) -> PipelineComponents:
    post_conf = config.get("postbatch", {})
    pipeline_conf = config.get("ocr_pipeline", {})
    engines_conf = pipeline_conf.get("engines", [])
    engine_configs = {
        str(engine.get("name", "")).lower(): engine for engine in engines_conf if engine.get("name")
    }
    if "poppler_path" in pipeline_conf:
        resolved_poppler = resolve_path(project_root, pipeline_conf["poppler_path"])
        engine_configs["poppler_path"] = resolved_poppler
        logger.info(f"Using Poppler path: {resolved_poppler}")
        
    paddle_conf = engine_configs.get("paddleocr", {})
    if paddle_conf is not None and "model_storage_dir" not in paddle_conf:
        paddle_conf["model_storage_dir"] = os.path.join(project_root, "models", "paddle")
    easy_conf = engine_configs.get("easyocr", {})

    languages: List[str] = []
    if paddle_conf.get("lang"):
        languages.append(str(paddle_conf["lang"]))
    if easy_conf.get("langs"):
        languages.extend(str(code) for code in easy_conf.get("langs", []))
    if not languages:
        languages = ["spa", "eng"]
    else:
        seen = set()
        languages = [lang for lang in languages if not (lang in seen or seen.add(lang))]

    fusion_conf = pipeline_conf.get("fusion", {}) or {}

    # Engine selection:
    # - If fusion.priority is explicitly set and non-empty, it controls primary/secondary ordering.
    # - Otherwise respect ocr_pipeline.primary_engine/secondary_engine (UI writes these).
    priority_raw = fusion_conf.get("priority")
    if isinstance(priority_raw, (list, tuple)):
        priority_list = [str(item).strip().lower() for item in priority_raw if str(item).strip()]
    else:
        priority_list = []

    configured_primary = str(pipeline_conf.get("primary_engine") or "").strip().lower()
    configured_secondary = str(pipeline_conf.get("secondary_engine") or "").strip().lower()

    known_ocr_engines = {"auto", "paddleocr", "easyocr", "surya", "paddlevl"}

    def _normalise_engine(name: str, fallback: str) -> str:
        if not name:
            return fallback
        if name in known_ocr_engines:
            return name
        logger.warning("Unknown OCR engine '%s'; falling back to '%s'.", name, fallback)
        return fallback

    if priority_list:
        primary_engine = _normalise_engine(priority_list[0], "paddleocr")
        secondary_default = "easyocr" if primary_engine != "easyocr" else "paddleocr"
        secondary_engine = _normalise_engine(priority_list[1] if len(priority_list) > 1 else "", secondary_default)
    else:
        primary_engine = _normalise_engine(configured_primary, "paddleocr")
        secondary_default = "easyocr" if primary_engine != "easyocr" else "paddleocr"
        secondary_engine = _normalise_engine(configured_secondary, secondary_default)

    if secondary_engine == primary_engine:
        secondary_engine = "easyocr" if primary_engine != "easyocr" else "paddleocr"

    # Fusion priority list is only used for tie-breaking; default to the selected engines.
    priority = tuple(priority_list) if priority_list else (primary_engine, secondary_engine)

    ocr_conf = OCRConfig(
        enabled=bool(post_conf.get("ocr_enabled", True)),
        languages=languages,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        fusion_strategy=str(fusion_conf.get("strategy", "confidence_vote")).lower(),
        min_confidence_primary=float(fusion_conf.get("min_confidence", 0.6)),
        confidence_margin=float(fusion_conf.get("confidence_margin", 0.05)),
        min_similarity=float(fusion_conf.get("min_similarity", 0.82)),
        engine_configs=engine_configs,
        preprocessing=pipeline_conf.get("preprocessing", {}),
    )
    ocr_manager = OCRManager(config=ocr_conf, logger=logger, gpu_id=gpu_id)

    layout_manager: Optional[LayoutManager] = None
    if ocr_conf.enabled and bool(paddle_conf.get("layout", True)) and paddle_conf.get("enabled", True):
        try:
            logger.info("Initializing LayoutManager...")
            layout_manager = LayoutManager(
                LayoutManagerConfig(
                    use_gpu=ocr_manager.use_gpu,
                    languages=languages,
                ),
                logger=logger,
            )
            logger.info("LayoutManager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LayoutManager: {e}")
            import traceback
            logger.error(traceback.format_exc())

    table_manager: Optional[TableManager] = None
    if ocr_conf.enabled and bool(paddle_conf.get("tables", True)) and paddle_conf.get("enabled", True):
        try:
            logger.info("Initializing TableManager...")
            table_manager = TableManager(
                TableManagerConfig(
                    use_gpu=ocr_manager.use_gpu,
                    languages=languages,
                    output_dir=resolve_path(
                        project_root, pipeline_conf.get("output", {}).get("tables_dir", "data/tables")
                    ),
                ),
                logger=logger,
            )
            logger.info("TableManager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize TableManager: {e}")
            import traceback
            logger.error(traceback.format_exc())

    fusion_manager = FusionManager(
        FusionConfig(
            strategy=str(fusion_conf.get("strategy", "confidence_vote")).lower(),
            min_confidence_primary=float(fusion_conf.get("min_confidence", 0.6)),
            confidence_margin=float(fusion_conf.get("confidence_margin", 0.05)),
            min_similarity=float(fusion_conf.get("min_similarity", 0.82)),
            priority=tuple(str(engine).lower() for engine in priority if engine),
        )
    )
    recheck_threshold = float(fusion_conf.get("recheck_threshold", fusion_conf.get("min_confidence", 0.6)))

    # Vision config is canonical at top-level `vision` (the web UI writes it),
    # but `config_test.yaml` and older configs used `ocr_pipeline.vision`.
    vision_conf: Dict[str, Any] = {}
    top_vision = config.get("vision")
    if isinstance(top_vision, dict):
        vision_conf = top_vision
    if not vision_conf:
        legacy_vision = pipeline_conf.get("vision")
        if isinstance(legacy_vision, dict):
            vision_conf = legacy_vision
    vision_manager: Optional[VisionManager] = None
    # Default to disabled unless explicitly enabled in config to avoid unexpected
    # heavy imports/DLL issues in production and test environments.
    if vision_conf.get("enabled", False):
        logger.info("Initializing VisionManager...")
        try:
            model_name = (
                vision_conf.get("model_name")
                or vision_conf.get("model")
                or "ViT-B-32"
            )
            vision_manager = VisionManager(
                VisionManagerConfig(
                    enabled=True,
                    index_path=resolve_path(project_root, vision_conf.get("index_path", "data/vision_index.faiss")),
                    embeddings_dir=resolve_path(
                        project_root, vision_conf.get("embeddings_dir", "data/vision_embeddings")
                    ),
                    model_name=model_name,
                    pretrained=str(vision_conf.get("pretrained", "laion2b_s34b_b79k")),
                    use_gpu=ocr_manager.use_gpu,
                ),
                logger=logger,
            )
            logger.info("VisionManager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VisionManager: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Vision is optional; continue without it.
            vision_manager = None

    output_conf = pipeline_conf.get("output", {})
    output_formats = [fmt.lower() for fmt in output_conf.get("formats", ["markdown", "json"])]
    save_markdown_in_db = bool(output_conf.get("save_markdown_in_db", True))

    # Initialize MinerU secondary engine if enabled
    mineru_engine = None
    mineru_conf = pipeline_conf.get("mineru", {})
    if mineru_conf.get("enabled", False):
        try:
            from modules.engines.mineru_wrapper import MinerUEngine
            logger.info("Initializing MinerU secondary engine...")
            mineru_engine = MinerUEngine(mineru_conf, logger=logger)
            if mineru_engine.initialize():
                logger.info("MinerU engine initialized successfully.")
            else:
                logger.warning("MinerU engine failed to initialize; continuing without it.")
                mineru_engine = None
        except Exception as e:
            logger.warning(f"Failed to load MinerU: {e}; continuing without it.")
            mineru_engine = None

    return PipelineComponents(
        ocr_manager=ocr_manager,
        layout_manager=layout_manager,
        table_manager=table_manager,
        fusion_manager=fusion_manager,
        vision_manager=vision_manager,
        mineru_engine=mineru_engine,
        recheck_threshold=recheck_threshold,
        output_formats=output_formats,
        save_markdown_in_db=save_markdown_in_db,
    )


def process_single_file(
    file_path: str,
    pipeline: PipelineComponents,
    classifier: Optional[DocumentClassifier],
    db: DBManager,
    processed_folder: str,
    failed_folder: str,
    delete_original: bool,
    ocr_enabled: bool,
    classification_enabled: bool,
    logger: logging.Logger,
    input_root: str,
    handwriting_mode: bool = False,
    pipeline_conf: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if pipeline_conf is None:
        pipeline_conf = {}
    filename = os.path.basename(file_path)
    start_time = time.time()
    status = "OK"
    doc_type: str = "Unknown"
    tags: List[str] = []
    dest_path = file_path

    try:
        file_hash = compute_hash(file_path)
        duplicate_id = db.check_duplicate(file_hash)
        if duplicate_id is not None:
            logger.info(
                "Duplicate detected for %s (document id %s); skipping insertion",
                filename,
                duplicate_id,
            )
            existing_path = db.get_document_path(duplicate_id)
            try:
                os.remove(file_path)
            except OSError:
                logger.debug("Failed to remove duplicate source %s", file_path, exc_info=True)
            dest_path = existing_path or file_path
            return {
                "filename": filename,
                "status": "DUPLICATE",
                "duration": 0.0,
                "type": doc_type,
                "path": dest_path,
                "doc_id": duplicate_id,
            }

        aggregated_text = ""
        markdown_text = ""
        language: Optional[str] = None
        confidence = 0.0
        block_outputs: List[Dict[str, Any]] = []
        table_results: List[TableResult] = []
        summary_payload: Dict[str, Any] = {"filename": filename}
        texts_join: List[str] = []
        confidences: List[float] = []
        layout_blocks: List[Dict[str, Any]] = []

        mineru_result = None
        if pipeline.mineru_engine and pipeline.mineru_engine.is_complex_document(file_path):
             try:
                 logger.info(f"MinerU: Complex document detected. Attempting extraction for {filename}")
                 mineru_result = pipeline.mineru_engine.process(file_path)
             except Exception as e:
                 logger.error(f"MinerU extraction failed: {e}")
                 mineru_result = None

        if mineru_result and mineru_result.get("text"):
            # Use MinerU result as primary text
            logger.info("MinerU: Extraction successful. Using MinerU output.")
            aggregated_text = mineru_result.get("text", "")
            markdown_text = mineru_result.get("text", "") # MinerU output is markdown
            
            # Populate tables/formulas in structured_data later
            # We skip standard OCR text extraction but might still want layout blocks if MinerU provides them or we can infer them
            # MinerU doesn't provide standard blocks easily comparable to Paddle/EasyOCR.
            # We will use dummy blocks or skip block-based features for this doc.
            
            language = "eng" # Todo: detect?
            confidence = 0.95 # MinerU doesn't give confidence, assume high if success
            
            # Map MinerU tables to TableResult objects? Or keep raw HTML?
            # We'll keep raw HTML in structured_data
            
        elif ocr_enabled and is_visual_document(file_path):
            # OPTIMIZATION: Try Native Extraction first
            native_blocks = None
            if file_path.lower().endswith(".pdf"):
                native_blocks = try_extract_native_pdf(file_path, logger)
            
            if native_blocks:
                logger.info("Using Native PDF Extraction (Skipping OCR)")
                block_outputs = native_blocks
                layout_blocks = native_blocks
                # Populate stats
                for b in block_outputs:
                    txt = b.get("text", "")
                    if txt:
                        texts_join.append(txt)
                        confidences.append(b.get("confidence", 0.99))
            else:
                # Fallback to standard OCR
                poppler_path = pipeline.ocr_manager.poppler_path if pipeline.ocr_manager else None

                if file_path.lower().endswith(".pdf"):
                    # Chunk PDF pages to avoid loading large PDFs fully into RAM.
                    pdf_conf = (pipeline_conf.get("ocr_pipeline", {}) or {}).get("pdf", {}) if pipeline_conf else {}
                    pages_per_chunk = int(pdf_conf.get("pages_per_chunk", 8) or 8)
                    max_pages = int(pdf_conf.get("max_pages", 0) or 0)

                    total_pages = _get_pdf_page_count(file_path, logger)
                    if max_pages > 0 and total_pages > max_pages:
                        logger.warning(
                            "PDF has %d pages; limiting OCR to first %d page(s) (ocr_pipeline.pdf.max_pages).",
                            total_pages,
                            max_pages,
                        )
                        total_pages = max_pages

                    block_id_counter = 0
                    for page_offset, chunk_pages in iter_pdf_page_chunks(
                        file_path,
                        logger=logger,
                        pages_per_chunk=pages_per_chunk,
                        poppler_path=poppler_path,
                        total_pages=total_pages,
                    ):
                        chunk_layout_blocks: List[Dict[str, Any]] = []
                        if pipeline.layout_manager:
                            try:
                                chunk_layout_blocks = pipeline.layout_manager.detect_blocks(
                                    file_path, chunk_pages
                                )
                            except Exception as exc:
                                logger.error(
                                    "Layout detection failed for %s (pages %s-%s): %s",
                                    filename,
                                    page_offset + 1,
                                    page_offset + len(chunk_pages),
                                    exc,
                                    exc_info=True,
                                )

                        if not chunk_layout_blocks:
                            chunk_layout_blocks = fallback_blocks(chunk_pages)

                        chunk_block_outputs, chunk_texts, chunk_confs, block_id_counter, sorted_blocks = process_layout_blocks(
                            pipeline,
                            chunk_pages,
                            chunk_layout_blocks,
                            logger,
                            handwriting_mode=handwriting_mode,
                            page_offset=page_offset,
                            block_id_start=block_id_counter,
                        )
                        block_outputs.extend(chunk_block_outputs)
                        texts_join.extend(chunk_texts)
                        confidences.extend(chunk_confs)

                        if pipeline.table_manager:
                            try:
                                chunk_tables = pipeline.table_manager.extract_tables(
                                    file_path,
                                    sorted_blocks,
                                    pages=chunk_pages,
                                )
                                for table in chunk_tables:
                                    if "page" in table and table["page"] is not None:
                                        table["page"] = int(table["page"]) + int(page_offset)
                                table_results.extend(chunk_tables)
                            except Exception as exc:
                                logger.error(
                                    "Table extraction failed for %s (pages %s-%s): %s",
                                    filename,
                                    page_offset + 1,
                                    page_offset + len(chunk_pages),
                                    exc,
                                    exc_info=True,
                                )

                    aggregated_text = "\n".join(texts_join).strip()
                    confidence = statistics.mean(confidences) if confidences else 0.0
                    language = pipeline.ocr_manager.languages[0] if aggregated_text else None

                else:
                    pages = load_document_pages(file_path, poppler_path=poppler_path)

                    layout_blocks: List[Dict[str, Any]] = []
                    if pipeline.layout_manager:
                        try:
                            layout_blocks = pipeline.layout_manager.detect_blocks(file_path, pages)
                        except Exception as exc:
                            logger.error("Layout detection failed for %s: %s", filename, exc, exc_info=True)

                    if not layout_blocks:
                        layout_blocks = fallback_blocks(pages)

                    block_outputs_chunk, chunk_texts, chunk_confs, _, sorted_blocks = process_layout_blocks(
                        pipeline,
                        pages,
                        layout_blocks,
                        logger,
                        handwriting_mode=handwriting_mode,
                        page_offset=0,
                        block_id_start=0,
                    )
                    block_outputs.extend(block_outputs_chunk)
                    texts_join.extend(chunk_texts)
                    confidences.extend(chunk_confs)

                    aggregated_text = "\n".join(texts_join).strip()
                    confidence = statistics.mean(confidences) if confidences else 0.0
                    language = pipeline.ocr_manager.languages[0] if aggregated_text else None

                    if pipeline.table_manager:
                        try:
                            table_results = pipeline.table_manager.extract_tables(
                                file_path,
                                sorted_blocks,
                                pages=pages,
                            )
                        except Exception as exc:
                            logger.error("Table extraction failed for %s: %s", filename, exc, exc_info=True)
                            table_results = []

            summary_payload.update(
                {
                    "language": language,
                    "confidence": confidence,
                    "blocks": block_outputs,
                    "tables": table_results,
                }
            )
        elif ocr_enabled:
            is_handwritten = False
            aggregated_text, language, confidence, is_handwritten = extract_content(
                file_path, pipeline.ocr_manager, logger
            )
            if is_handwritten:
                if "Manuscrito" not in tags:
                    tags.append("Manuscrito")
                logger.info(f"Auto-detected handwriting in {filename}")

            summary_payload.update(
                {
                    "language": language,
                    "confidence": confidence,
                    "blocks": [],
                    "tables": [],
                    "is_handwritten": is_handwritten
                }
            )
        else:
            summary_payload.update(
                {"language": None, "confidence": 0.0, "blocks": [], "tables": []}
            )

        if aggregated_text:
            needs_markdown = pipeline.save_markdown_in_db or (
                "markdown" in pipeline.output_formats
            )
            if needs_markdown:
                markdown_text = ocr_text_to_markdown(aggregated_text)
        summary_payload["text"] = aggregated_text

        if classification_enabled and classifier:
            doc_type, tags = classifier.classify(aggregated_text)

        # ------------------------------------------------------------------ #
        # Intelligent Auto-Detection Fallback (Vision-based)
        # ------------------------------------------------------------------ #
        vision_conf = pipeline_conf.get("vision", {})
        if (
            vision_conf.get("enabled", False)
            and pipeline.vision_manager
            and (doc_type == "Unknown" or confidence < 0.6)
            and is_visual_document(file_path)
        ):
            try:
                doc_candidates = ["factura", "recibo", "contrato", "informe", "carta", "mueble", "decoracion"]
                visual_doc_results = pipeline.vision_manager.classify_image(file_path, doc_candidates)
                if visual_doc_results:
                    top_doc, top_score = visual_doc_results[0]
                    if top_score > 0.7:
                        # Map to internal types
                        mapping = {
                            "factura": "Invoice",
                            "recibo": "Receipt",
                            "contrato": "Contract",
                            "informe": "Report",
                            "carta": "Letter",
                            "mueble": "Imagen",
                            "decoracion": "Imagen"
                        }
                        doc_type = mapping.get(top_doc, "Unknown")
                        tags.append(f"VisualClass: {top_doc} ({int(top_score*100)}%)")
                        logger.info(f"Intelligent Auto-Detection: {top_doc} ({top_score:.2f})")
            except Exception as e:
                logger.warning(f"Intelligent auto-detection failed: {e}")

        # Fallback to "Imagen" for visual documents with no specific type found
        if doc_type == "Unknown" and is_visual_document(file_path):
            doc_type = "Imagen"

        # Default workflow state; final decision happens later once structured_data is available.
        workflow_state = "verified"

        # ------------------------------------------------------------------ #
        # NUEVO: Enrutador de Interpretación Avanzada (LLM) - DISABLED BY USER REQUEST
        # ------------------------------------------------------------------ #
        # try:
        #     # 1. Preparar datos para el router
        #     router_input = {
        #         "document_id": file_hash,
        #         "tipo_archivo": "pdf" if file_path.lower().endswith(".pdf") else "imagen",
        #         "paginas": len(pages) if 'pages' in locals() else 1,
        #         "es_pdf_nativo": bool(native_blocks) if 'native_blocks' in locals() and native_blocks else False,
        #         "clasificacion_previa": doc_type,
        #         "metricas_ocr": {
        #             "confianza_media": float(confidence),
        #             "bloques_baja_confianza": len([b for b in block_outputs if b.get("confidence", 1.0) < 0.5]),
        #             "texto_legible_global": bool(aggregated_text and len(aggregated_text) > 50)
        #         },
        #         "indicadores_graficos": {
        #             "escritura_mano_detectada": "Manuscrito" in tags,
        #             "dibujos_o_lineas_no_textuales": "VisualClass" in str(tags),
        #             "estructura_visual_irregular": False
        #         },
        #         "resumen_ocr": aggregated_text[:500] if aggregated_text else ""
        #     }
        #
        #     # 2. Instanciar y Evaluar
        #     router = AdvancedInterpretationRouter(logger=logger)
        #     decision_llm = router.evaluate_document(router_input)
        #
        #     # 3. Actuar según decisión
        #     if decision_llm["activar_interpretacion_avanzada"]:
        #         logger.info(f"🤖 IA Activada: {decision_llm['motivo']} (Confianza: {decision_llm['confianza_decision']})")
        #         tags.append("Requires_Advanced_Review")
        #         summary_payload["interpretation_needed"] = True
        #         summary_payload["interpretation_reason"] = decision_llm["motivo"]
        #         
        #         # --- INICIO LLM INVOCATION ---
        #         try:
        #             llm_config = pipeline_conf.get("llm", {})
        #             if llm_config.get("enabled", False):
        #                 llm_client = LLMClient(llm_config, logger=logger)
        #                 analysis_result = llm_client.analyze_document(
        #                     text=aggregated_text,
        #                     reason=decision_llm["motivo"],
        #                     doc_type=doc_type
        #                 )
        #                 if analysis_result.get("success"):
        #                     summary_payload["llm_analysis"] = analysis_result["analysis"]
        #                     logger.info("✅ Análisis LLM completado y adjunto.")
        #                 else:
        #                     logger.warning(f"⚠️ Análisis LLM falló: {analysis_result.get('error')}")
        #         except Exception as e_llm:
        #             logger.error(f"Error crítico invocando LLM: {e_llm}")
        #         # --- FIN LLM INVOCATION ---
        #         
        #     else:
        #         logger.debug(f"IA Omitida: {decision_llm['motivo']}")
        #
        # except Exception as e_router:
        #     logger.warning(f"Fallo no crítico en router de interpretación: {e_router}")

        # ------------------------------------------------------------------ #
        # Visual Auto-Tagging (Zero-Shot)
        # ------------------------------------------------------------------ #
        vision_conf = pipeline_conf.get("vision", {})
        auto_tag_conf = vision_conf.get("auto_tagging", {})
        if (
            vision_conf.get("enabled", False)
            and auto_tag_conf.get("enabled", False)
            and pipeline.vision_manager
        ):
            candidates = auto_tag_conf.get("candidates", [])
            if candidates:
                try:
                    visual_tags = pipeline.vision_manager.classify_image(file_path, candidates)
                    for tag, score in visual_tags:
                        tags.append(f"{tag} ({int(score*100)}%)")
                    if visual_tags:
                        logger.info(f"👁️ Visual Tags: {[t[0] for t in visual_tags]}")
                        
                        # Decor Advice
                        advisor = DecorAdvisor()
                        advice_list = advisor.generate_advice(tags)
                        for tip in advice_list:
                            tags.append(tip)
                            
                except Exception as e:
                    logger.warning(f"Visual tagging failed: {e}")

            # -------------------------------------------------------------- #
            # Color Palette Extraction
            # -------------------------------------------------------------- #
            try:
                if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS:
                    color_extractor = ColorExtractor()
                    palette = color_extractor.extract_palette(file_path, k=5)
                    if palette:
                        for color in palette:
                            tags.append(f"color:{color}")
                        logger.info(f"🎨 Palette extracted: {palette}")
            except Exception as e:
                logger.warning(f"Color extraction failed: {e}")

        # ------------------------------------------------------------------ #
        # Project Grouping Logic
        # ------------------------------------------------------------------ #
        grouping_conf = pipeline_conf.get("postbatch", {}).get("project_grouping", {})
        if grouping_conf.get("enabled", False) and aggregated_text:
            for pattern in grouping_conf.get("patterns", []):
                match = re.search(pattern, aggregated_text, re.IGNORECASE)
                if match:
                    project_code = match.group(0).upper().strip().replace(" ", "-")
                    # Create subfolder for project
                    project_folder = os.path.join(processed_folder, project_code)
                    if not os.path.exists(project_folder):
                        try:
                            os.makedirs(project_folder, exist_ok=True)
                            logger.info(f"Created project folder: {project_folder}")
                        except OSError as e:
                            logger.error(f"Failed to create project folder {project_folder}: {e}")
                            project_folder = processed_folder # Fallback
                    
                    # Update destination folder logic
                    processed_folder = project_folder 
                    tags.append(f"Project: {project_code}")
                    logger.info(f"File {filename} grouped into project {project_code}")
                    break

        # ------------------------------------------------------------------ #
        # Vendor Alias Logic
        # ------------------------------------------------------------------ #
        alias_conf = pipeline_conf.get("postbatch", {}).get("vendor_aliases", {})
        if alias_conf.get("enabled", False) and aggregated_text:
            normalized_text = aggregated_text.lower()
            for main_vendor, aliases in alias_conf.get("mappings", {}).items():
                # Check main vendor name first
                found = main_vendor.lower() in normalized_text
                # If not found, check aliases
                if not found:
                    for alias in aliases:
                        if alias.lower() in normalized_text:
                            found = True
                            break
                
                if found:
                    tags.append(f"Vendor: {main_vendor}")
                    logger.info(f"detected vendor alias for {main_vendor}")
                    # We might want to stop after first vendor or allow multiple? 
                    # For now, let's allow multiple.

        # ------------------------------------------------------------------ #
        # Smart Renaming Logic
        # ------------------------------------------------------------------ #
        renaming_conf = pipeline_conf.get("postbatch", {}).get("smart_renaming", {})
        original_filename = filename
        if renaming_conf.get("enabled", False):
             try:
                # Extract components
                # Date: Try to find a date in OCR text or use today
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                # (Simple OCR date extraction placeholder - could be improved with regex)
                
                # Project
                project_str = "NoProject"
                for tag in tags:
                     if tag.startswith("Project:"):
                          project_str = tag.split(":", 1)[1].strip()
                          break
                
                # Vendor
                vendor_str = "NoVendor"
                for tag in tags:
                     if tag.startswith("Vendor:"):
                          vendor_str = tag.split(":", 1)[1].strip()
                          break
                
                # Type
                type_str = doc_type if doc_type != "Unknown" else "Doc"

                # Construct new name
                fmt = renaming_conf.get("format", "{date}_{type}_{project}_{vendor}_{filename}")
                new_name = fmt.format(
                     date=date_str,
                     type=type_str,
                     project=project_str,
                     vendor=vendor_str,
                     filename=os.path.splitext(original_filename)[0]
                )
                
                # Sanitize
                new_name = re.sub(r'[<>:"/\\|?*]', '', new_name) # Remove illegal chars
                new_name = new_name.replace(" ", "_").strip("_")
                new_name += os.path.splitext(original_filename)[1]
                
                logger.info(f"Smart Renaming: {filename} -> {new_name}")
                filename = new_name # Update filename variable for move_file
             except Exception as e:
                logger.error(f"Smart renaming failed: {e}")
                filename = original_filename

        # ------------------------------------------------------------------ #
        # Destination Path (Processed Folder)
        # ------------------------------------------------------------------ #
        # In production, leaving files in the input folder causes re-ingestion loops and
        # output collisions. Allow disabling movement explicitly via config, but default
        # to moving/copying into `processed_folder`.
        disable_movement = bool(
            pipeline_conf.get("postbatch", {}).get("disable_file_movement", False)
        )
        if disable_movement:
            logger.info("Configuration: File movement disabled. Keeping file at source.")
            dest_path = file_path
        else:
            try:
                dest_path = move_file(
                    file_path,
                    processed_folder,
                    delete_original=delete_original,
                    relative_to=input_root,
                    new_filename=filename,
                )
            except Exception as move_error:
                logger.error(
                    "Unable to move %s to processed folder: %s",
                    filename,
                    move_error,
                    exc_info=True,
                )
                dest_path = file_path

        if not classification_enabled:
            tags = []

        # ------------------------------------------------------------------ #
        # Smart Field Extraction & Validation
        # ------------------------------------------------------------------ #
        structured_data: Dict[str, Any] = {}
        if 'mineru_result' in locals() and mineru_result:
             structured_data["mineru"] = {
                 "tables": mineru_result.get("tables", []),
                 "formulas": mineru_result.get("formulas", []),
                 "metadata": mineru_result.get("metadata", {})
             }

        if aggregated_text and classification_enabled:
            try:
                from modules.smart_extractor import FieldExtractor
                from modules.normalizer import DataNormalizer
                from modules.anomaly_detector import AnomalyDetector
                
                extractor = FieldExtractor(pipeline_conf)
                fields = extractor.extract_fields(aggregated_text, block_outputs)
                
                if fields:
                    normalizer = DataNormalizer(pipeline_conf)
                    fields = normalizer.normalize(fields)
                    
                    # Detect anomalies
                    detector = AnomalyDetector(pipeline_conf)
                    anomalies = detector.detect(fields)

                    # Preserve any prior structured keys (e.g. MinerU extraction).
                    structured_data["fields"] = fields
                    structured_data["anomalies"] = anomalies

                    logger.info(f"📊 Extracted fields: {list(fields.keys())}")
                    if anomalies:
                        logger.warning(f"⚠️ Anomalies detected: {anomalies}")
            except Exception as e:
                logger.error(f"Field extraction failed: {e}")

        # ------------------------------------------------------------------ #
        # Workflow State (QC gating)
        # ------------------------------------------------------------------ #
        post_conf = (pipeline_conf.get("postbatch", {}) or {}) if pipeline_conf else {}
        auto_verify = bool(post_conf.get("auto_verify", True))
        force_pending_below = float(post_conf.get("force_pending_below", 0.45))
        review_conf_threshold = float(post_conf.get("review_confidence_threshold", 0.8))
        min_text_chars = int(post_conf.get("review_min_text_chars", 30))

        text_len = len((aggregated_text or "").strip())
        anomalies_present = bool(
            isinstance(structured_data, dict) and structured_data.get("anomalies")
        )
        handwriting_present = "Manuscrito" in tags

        severe_issue = (
            (is_visual_document(file_path) and text_len == 0)
            or confidence < force_pending_below
            or anomalies_present
            or handwriting_present
        )

        if auto_verify:
            # Unattended ingestion: only escalate clearly broken cases.
            workflow_state = "pending" if severe_issue else "verified"
        else:
            needs_review = severe_issue or confidence < review_conf_threshold or text_len < min_text_chars
            workflow_state = "pending" if needs_review else "verified"

        logger.info(
            "Workflow=%s (auto_verify=%s, confidence=%.2f, text_len=%d, anomalies=%s, handwriting=%s)",
            workflow_state,
            auto_verify,
            confidence,
            text_len,
            anomalies_present,
            handwriting_present,
        )

        # ------------------------------------------------------------------ #
        # Persist Document + OCR Output
        # ------------------------------------------------------------------ #
        duration = time.time() - start_time

        # Phase 4 Metadata extraction from pipeline_conf
        owner_id = pipeline_conf.get("owner_id")
        hotel_id = pipeline_conf.get("hotel_id")
        # Use user-provided doc_type if provided, otherwise fallback to classifier's doc_type
        final_doc_type = pipeline_conf.get("doc_type") or doc_type
        visibility = pipeline_conf.get("visibility", "private")
        financial_level = pipeline_conf.get("financial_level", "none")

        doc_id = db.insert_document(
            filename,
            dest_path,
            file_hash,
            datetime.datetime.now(),
            duration,
            status,
            doc_type=final_doc_type,
            tags=tags,
            workflow_state=workflow_state,
            owner_id=owner_id,
            hotel_id=hotel_id,
            visibility=visibility,
            financial_level=financial_level,
        )

        if aggregated_text:
            db.insert_ocr_text(
                doc_id,
                aggregated_text,
                markdown_text=markdown_text if pipeline.save_markdown_in_db else None,
                language=language,
                confidence=confidence,
                blocks=block_outputs or None,
                tables=table_results or None,
                structured_data=structured_data,
            )

        save_additional_outputs(dest_path, summary_payload, markdown_text, pipeline)

        logger.info(
            "Processed %s: type=%s, confidence=%.2f, duration=%.2fs",
            filename,
            doc_type,
            confidence,
            duration,
        )

        return {
            "filename": filename,
            "status": status,
            "duration": round(duration, 2),
            "type": doc_type,
            "path": dest_path,
            "doc_id": doc_id,
        }

    except Exception as exc:
        logger.error("Processing failed for %s: %s", filename, exc, exc_info=True)
        status = "FAILED"
        duration = time.time() - start_time

        try:
            dest_path = move_file(
                file_path,
                failed_folder,
                delete_original=True,
                relative_to=input_root,
            )
        except Exception as move_error:
            logger.error(
                "Unable to move %s to failed folder: %s", filename, move_error, exc_info=True
            )
            dest_path = file_path

        try:
            doc_id = db.insert_document(
                filename=filename,
                path=dest_path,
                md5_hash=compute_hash(file_path) if os.path.exists(file_path) else "unknown",
                timestamp=datetime.datetime.fromtimestamp(start_time),
                duration=duration,
                status=status,
                doc_type=doc_type,
                tags=["FAILED"],
                error_message=str(exc),
            )
            logger.info(f"Recorded failure in DB for {filename} (ID: {doc_id})")
        except Exception as db_err:
            logger.error(f"Failed to insert failure record for {filename}: {db_err}")

        return {
            "filename": filename,
            "status": status,
            "duration": round(duration, 2),
            "type": doc_type,
            "path": dest_path,
            "doc_id": None,
        }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutOCR post-batch processor")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Process immediately without waiting for inactivity",
    )
    parser.add_argument(
        "--input-folder",
        help="Override input folder from configuration",
    )
    args = parser.parse_args(argv)

    logger = logging.getLogger("AutOCR")

    config = load_config(args.config)
    post_conf = config.get("postbatch", {})
    app_conf = config.get("app", {})

    project_root = os.path.dirname(os.path.abspath(args.config))

    input_folder = args.input_folder or post_conf.get("input_folder", "")
    input_folder = resolve_path(project_root, input_folder)
    processed_folder = resolve_path(project_root, post_conf.get("processed_folder"))
    failed_folder = resolve_path(project_root, post_conf.get("failed_folder"))
    reports_folder = resolve_path(project_root, post_conf.get("reports_folder"))

    file_types = post_conf.get("file_types", [".pdf", ".tif", ".tiff", ".jpg", ".jpeg"])
    ocr_enabled = bool(post_conf.get("ocr_enabled", True))
    classification_enabled = bool(post_conf.get("classification_enabled", True))
    delete_original = bool(post_conf.get("delete_original", False))
    batch_summary_report = bool(post_conf.get("batch_summary_report", True))
    inactivity_minutes = int(post_conf.get("inactivity_trigger_minutes", 0))
    max_input_gb_per_run = float(post_conf.get("max_input_gb_per_run", 0) or 0.0)
    max_input_free_disk_ratio = float(post_conf.get("max_input_free_disk_ratio", 0) or 0.0)
    if max_input_free_disk_ratio < 0:
        max_input_free_disk_ratio = 0.0
    if max_input_free_disk_ratio > 1:
        max_input_free_disk_ratio = 1.0
    max_files_per_run = int(post_conf.get("max_files_per_run", 0) or 0)
    processing_order = str(post_conf.get("processing_order", "as_found") or "as_found")
    min_free_disk_gb = float(post_conf.get("min_free_disk_gb", 20) or 20)
    expected_output_multiplier = float(post_conf.get("expected_output_multiplier", 1.2) or 1.2)
    if expected_output_multiplier < 0:
        expected_output_multiplier = 0.0
    # Optimize workers for Ryzen 9 9950X (32 threads) + Dual RTX 4070
    # If using GPU, we must limit workers to avoid VRAM saturation.
    # If CPU only, we can go higher but 64 (2x32) might be too much context switching.
    default_workers = 24  # Good baseline for high-end CPU
    if post_conf.get("max_workers"):
        max_workers = int(post_conf["max_workers"])
    else:
        # Auto-tuning
        cpu_threads = os.cpu_count() or 1
        if bool(config.get("ocr_pipeline", {}).get("engines", [{}])[0].get("use_gpu", False)):
             # GPU Mode: Limit to avoid OOM (e.g. 10-12 workers is usually safe for 24GB VRAM)
             max_workers = 12 
        else:
             # CPU Mode: Use ~75% of threads to leave room for OS/IO
             max_workers = max(4, int(cpu_threads * 0.75))
             
     # max_workers = int(post_conf.get("max_workers", max(1, (os.cpu_count() or 1) * 2)))

    db_path = resolve_path(project_root, app_conf.get("db_path", "data/digitalizerai.db"))
    use_sql_server = bool(app_conf.get("use_sql_server", False))
    sql_server_dsn = app_conf.get("sql_server_dsn")
    log_level = app_conf.get("log_level", "INFO")

    if not processed_folder or not failed_folder or not reports_folder:
        logger.error("Processed, failed and reports folders must be configured.")
        return 1

    ensure_directories(processed_folder, failed_folder, reports_folder)

    db = DBManager(config)

    log_name = datetime.datetime.now().strftime("postbatch_%Y%m%d.log")
    log_file_path = os.path.join(reports_folder, log_name)
    logger = setup_logger(log_file_path, level=log_level, db_manager=db)
    logger.info("AutOCR post-batch processor started")

    # Recovery / Crash Protection
    recovery_path = os.path.join(reports_folder, "recovery_state.json")
    quarantine_path = os.path.join(failed_folder, "CRASH_QUARANTINE")
    recovery_mgr = RecoveryManager(recovery_path, quarantine_path, logger)
    recovery_mgr.recover()

    try:
        if not args.immediate and inactivity_minutes > 0:
            logger.info(
                "Waiting for %s minutes of inactivity in %s before processing",
                inactivity_minutes,
                input_folder,
            )
            monitor = InactivityMonitor(folder=input_folder, inactivity_minutes=inactivity_minutes)
            monitor.wait()
        else:
            logger.info("Immediate processing mode active; skipping inactivity wait.")

        if not input_folder:
            logger.error("No input folder specified. Use config.yaml or --input-folder.")
            return 1

        if not os.path.exists(input_folder):
            logger.error("Input folder does not exist: %s", input_folder)
            return 1

        files = list_scannable_files(input_folder, file_types)
        if not files:
            logger.info("No files to process in %s. Exiting.", input_folder)
            return 0

        max_input_bytes_per_run, cap_info = resolve_max_input_bytes_per_run(
            processed_folder,
            max_input_gb_per_run=max_input_gb_per_run,
            max_input_free_disk_ratio=max_input_free_disk_ratio,
        )
        if cap_info["effective_bytes"] > 0:
            logger.info(
                "Input cap resolved: %.2f GiB (fixed=%.2f GiB, free_ratio=%.2f => %.2f GiB of %.2f GiB free)",
                cap_info["effective_bytes"] / BYTES_PER_GIB,
                cap_info["fixed_bytes"] / BYTES_PER_GIB,
                cap_info["ratio"],
                cap_info["ratio_bytes"] / BYTES_PER_GIB,
                cap_info["free_bytes"] / BYTES_PER_GIB,
            )
        else:
            logger.info(
                "Input cap resolved: unlimited (fixed_gb=%.2f, free_ratio=%.2f).",
                max_input_gb_per_run,
                max_input_free_disk_ratio,
            )

        files, deferred_files, selected_bytes, discovered_bytes = plan_processing_batch(
            files,
            max_input_gb_per_run=max_input_gb_per_run,
            max_input_bytes_per_run=max_input_bytes_per_run,
            max_files_per_run=max_files_per_run,
            processing_order=processing_order,
        )
        if not files:
            logger.info("No files selected after batch planning. Exiting.")
            return 0

        selected_gb = selected_bytes / BYTES_PER_GIB
        discovered_gb = discovered_bytes / BYTES_PER_GIB
        logger.info(
            "Batch planning: selected %d file(s) %.2f GiB out of %d discovered file(s) %.2f GiB "
            "(order=%s, max_input_gb_per_run=%.2f)",
            len(files),
            selected_gb,
            len(files) + len(deferred_files),
            discovered_gb,
            processing_order,
            max_input_gb_per_run,
        )
        if max_files_per_run > 0:
            logger.info("Per-run file cap: max_files_per_run=%d", max_files_per_run)
        if deferred_files:
            deferred_bytes = max(0, discovered_bytes - selected_bytes)
            logger.info(
                "Deferred %d file(s) %.2f GiB for next run(s).",
                len(deferred_files),
                deferred_bytes / BYTES_PER_GIB,
            )

        estimated_required_bytes = int(selected_bytes * expected_output_multiplier)
        headroom_ok, headroom = check_disk_headroom(
            processed_folder,
            min_free_gb=min_free_disk_gb,
            required_bytes=estimated_required_bytes,
        )
        logger.info(
            "Disk headroom: free=%.2f GiB, min_free=%.2f GiB, estimated_required=%.2f GiB",
            headroom["free_bytes"] / BYTES_PER_GIB,
            headroom["min_free_bytes"] / BYTES_PER_GIB,
            headroom["required_bytes"] / BYTES_PER_GIB,
        )
        if not headroom_ok:
            logger.error(
                "Insufficient disk headroom for this batch. Required minimum %.2f GiB, available %.2f GiB. "
                "Tune postbatch.min_free_disk_gb / expected_output_multiplier / max_input_gb_per_run.",
                headroom["required_min_bytes"] / BYTES_PER_GIB,
                headroom["free_bytes"] / BYTES_PER_GIB,
            )
            return 1

        if not delete_original and selected_bytes >= int(50 * BYTES_PER_GIB):
            logger.warning(
                "Large batch with delete_original=false (%.2f GiB selected). "
                "Storage usage may spike due to source copy + outputs.",
                selected_gb,
            )

        # Detect number of GPUs.
        # Note: Paddle device selection is process-global; thread pools cannot reliably isolate GPUs per worker.
        gpu_count = 0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = int(torch.cuda.device_count())
        except Exception:
            try:
                import paddle

                if paddle.device.is_compiled_with_cuda():
                    gpu_count = int(paddle.device.cuda.device_count())
            except Exception:
                gpu_count = 0

        num_gpus = gpu_count if gpu_count > 0 else 1
        logger.info("Detected %d GPU(s).", gpu_count)

        # Safety: PaddleOCR/PPStructure runs on a process-global device context. Running multiple threads that
        # share a single GPU model instance is a common source of VRAM OOMs and hard crashes. Default to
        # sequential processing when PaddleOCR GPU is enabled unless explicitly opted-in.
        try:
            if gpu_count > 0 and ocr_enabled and max_workers > 1:
                ocr_pipeline_conf = config.get("ocr_pipeline", {})
                engines_conf = ocr_pipeline_conf.get("engines", []) or []
                paddle_engine_conf = next(
                    (
                        engine
                        for engine in engines_conf
                        if str(engine.get("name", "")).lower() == "paddleocr"
                    ),
                    {},
                )
                paddle_gpu_requested = bool(paddle_engine_conf.get("use_gpu", False)) and bool(
                    paddle_engine_conf.get("enabled", True)
                )
                allow_gpu_threadpool = bool(post_conf.get("allow_gpu_threadpool", False))
                if paddle_gpu_requested and not allow_gpu_threadpool:
                    logger.warning(
                        "PaddleOCR GPU detected; forcing max_workers=1 for stability. "
                        "Set postbatch.allow_gpu_threadpool=true to override (not recommended)."
                    )
                    max_workers = 1
        except Exception:
            # Defensive: never abort the batch due to tuning logic.
            pass

        pipeline_factory = lambda gid: initialise_pipeline(config, project_root, logger, gpu_id=gid)
        classifier_factory: Optional[Callable[[], DocumentClassifier]] = (
            (lambda: DocumentClassifier()) if classification_enabled else None
        )

        def process_with_context(file_path: str) -> Dict[str, Any]:
            recovery_mgr.register_start(file_path)
            try:
                pipeline_components, classifier_instance = _get_worker_components(
                    pipeline_factory,
                    classifier_factory,
                    num_gpus=num_gpus,
                )
                result = process_single_file(
                    file_path,
                    pipeline_components,
                    classifier_instance,
                    db,
                    processed_folder,
                    failed_folder,
                    delete_original=delete_original,
                    ocr_enabled=ocr_enabled,
                    classification_enabled=classification_enabled,
                    logger=logger,
                    input_root=input_folder,
                    pipeline_conf=config,
                )
                return result
            finally:
                recovery_mgr.register_complete(file_path)

        metrics_records: List[Dict[str, Any]] = []
        ok_count = 0
        fail_count = 0
        total_start = time.time()

        logger.info("Processing %s files using %s workers", len(files), max_workers)

        def append_metrics(result: Dict[str, Any]) -> None:
            nonlocal ok_count, fail_count
            metrics_records.append(
                {
                    "filename": result["filename"],
                    "status": result["status"],
                    "duration": result["duration"],
                    "type": result["type"],
                }
            )
            if result["status"] == "OK":
                ok_count += 1
            elif result["status"] == "FAILED":
                fail_count += 1

        future_to_file: Dict[Any, str] = {}
        executor_failed = False

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for file_path in files:
                    future = executor.submit(process_with_context, file_path)
                    future_to_file[future] = file_path

                for future in as_completed(future_to_file):
                    append_metrics(future.result())
        except RuntimeError as exc:
            message = str(exc).lower()
            if "cannot schedule new futures after interpreter shutdown" in message:
                logger.warning(
                    "Thread pool unavailable during shutdown; processing remaining files sequentially."
                )
                executor_failed = True
            else:
                raise

        if executor_failed:
            processed_names = {record["filename"] for record in metrics_records}
            # Explicitly process any files that were not completed before shutdown.
            for file_path in files:
                if os.path.basename(file_path) in processed_names:
                    continue
                result = process_with_context(file_path)
                append_metrics(result)
                processed_names.add(result["filename"])

        _clear_worker_components()

        total_duration = time.time() - total_start
        total_docs = ok_count + fail_count
        avg_time = (total_duration / total_docs) if total_docs else 0.0
        reliability_pct = ((ok_count / total_docs) * 100.0) if total_docs else 0.0

        db.insert_metrics(
            timestamp=datetime.datetime.now(),
            ok_docs=ok_count,
            failed_docs=fail_count,
            avg_time=avg_time,
            reliability_pct=reliability_pct,
        )

        logger.info(
            "Batch complete: ok={}, failed={}, avg_time={:.2f}s, reliability={:.2f}%".format(
                ok_count, fail_count, avg_time, reliability_pct
            )
        )
        # Vision index maintenance should be triggered explicitly (e.g. via the web settings
        # background task). Avoid implicit rebuilds here, as they may require heavy optional
        # dependencies and can fail the whole batch run.

        if batch_summary_report:
            metrics_summary = {
                "ok_docs": ok_count,
                "failed_docs": fail_count,
                "avg_time": avg_time,
                "reliability_pct": reliability_pct,
            }
            generate_summary_report(
                records=metrics_records,
                report_folder=reports_folder,
                metrics=metrics_summary,
            )
            logger.info("Summary report generated in %s", reports_folder)

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
