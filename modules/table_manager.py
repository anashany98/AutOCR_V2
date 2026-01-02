"""
Table extraction utilities using PaddleOCR PP-Structure.

The :class:`TableManager` consumes layout blocks and runs PaddleOCR's table
recognition on detected table regions, exporting both CSV files and structured
JSON suitable for downstream processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd
from PIL import Image

from .layout_manager import LayoutBlock
from .paddle_singleton import get_paddle_ocr

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore


class TableResult(TypedDict, total=False):
    """Structured information about an extracted table."""

    page: int
    bbox: List[int]
    csv_path: str
    json_path: str
    structure: dict


@dataclass
class TableManagerConfig:
    """Configuration for :class:`TableManager`."""

    use_gpu: bool = False
    languages: Sequence[str] | None = None
    output_dir: str = os.path.join("data", "tables")


class TableManager:
    """
    Extract tables from documents based on layout analysis results.

    Parameters
    ----------
    config:
        Behavioural options including GPU usage and output directory.
    logger:
        Optional logger used for diagnostics.
    """

    def __init__(
        self,
        config: TableManagerConfig | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or TableManagerConfig()
        self._engine = None
        self._initialise_engine()

    def extract_tables(
        self,
        image_or_pdf_path: str,
        blocks: Iterable[LayoutBlock],
        pages: Optional[Sequence[Image.Image]] = None,
    ) -> List[TableResult]:
        """
        Extract tables based on layout ``blocks``.

        Returns
        -------
        list[TableResult]
            A list with references to exported CSV/JSON files and in-memory
            structures.
        """
        table_blocks_by_page = self._group_table_blocks(blocks)
        if not table_blocks_by_page:
            return []

        page_images = list(pages) if pages is not None else self._load_document_images(image_or_pdf_path)
        results: List[TableResult] = []
        ensure_directory(self.config.output_dir)

        for page_index, page_blocks in table_blocks_by_page.items():
            if page_index >= len(page_images):
                continue
            page_image = page_images[page_index]

            for table_index, block in enumerate(page_blocks):
                table_id = self._table_identifier(
                    image_or_pdf_path, page_index, table_index
                )
                crop = crop_bbox(page_image, block["bbox"])
                if crop is None:
                    continue

                if self._engine is None:
                    self.logger.warning(
                        "Table engine unavailable; skipping table extraction for %s",
                        table_id,
                    )
                    continue

                page_array = pil_to_np(crop)
                try:
                    engine_results = self._engine(page_array)
                except Exception as exc:  # pragma: no cover - Paddle runtime errors
                    self.logger.error(
                        "Paddle table extraction failed for %s: %s", table_id, exc
                    )
                    continue

                table_items = [
                    item for item in engine_results if item.get("type") == "table"
                ]
                if not table_items:
                    continue

                table_item = table_items[0]
                structure = table_item.get("res", {}) or {}
                csv_path = os.path.join(self.config.output_dir, f"{table_id}.csv")
                json_path = os.path.join(self.config.output_dir, f"{table_id}.json")

                try:
                    dataframe = cells_to_dataframe(structure.get("cells", []))
                    dataframe.to_csv(csv_path, index=False)
                    with open(json_path, "w", encoding="utf-8") as fh:
                        json.dump(structure, fh, ensure_ascii=False, indent=2)
                except Exception as exc:
                    self.logger.error(
                        "Failed to export table %s: %s", table_id, exc
                    )
                    continue

                results.append(
                    TableResult(
                        page=page_index,
                        bbox=block["bbox"],
                        csv_path=csv_path,
                        json_path=json_path,
                        structure=structure,
                    )
                )

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _initialise_engine(self) -> None:
        try:
            engine = get_paddle_ocr()
        except Exception as exc:  # pragma: no cover - Paddle runtime errors
            self.logger.warning(
                "Failed to initialise PaddleOCR table engine (%s); disabling table extraction.",
                exc,
            )
            self._engine = None
            return

        self._engine = engine

    @staticmethod
    def _group_table_blocks(
        blocks: Iterable[LayoutBlock],
    ) -> dict[int, List[LayoutBlock]]:
        grouped: dict[int, List[LayoutBlock]] = {}
        for block in blocks:
            if block.get("type") != "table":
                continue
            page = int(block.get("page", 0))
            grouped.setdefault(page, []).append(block)
        return grouped

    @staticmethod
    def _table_identifier(path: str, page_index: int, table_index: int) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{digest}_p{page_index:02d}_t{table_index:02d}"

    def _load_document_images(self, path: str) -> List[Image.Image]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            if convert_from_path is None:
                raise RuntimeError(
                    "pdf2image is required for PDF table extraction but is not installed"
                )
            pages = convert_from_path(path)
            return [page.convert("RGB") for page in pages]

        with Image.open(path) as image:
            images: List[Image.Image] = []
            try:
                n_frames = getattr(image, "n_frames", 1)
            except Exception:
                n_frames = 1

            for idx in range(n_frames):
                try:
                    image.seek(idx)
                except EOFError:
                    break
                images.append(image.convert("RGB").copy())
            if not images:
                images.append(image.convert("RGB").copy())
            return images


def pil_to_np(image: Image.Image) -> np.ndarray:
    array = np.array(image)
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return array[:, :, ::-1].copy()


def crop_bbox(image: Image.Image, bbox: Iterable[int]) -> Optional[Image.Image]:
    coords = list(bbox)
    if len(coords) != 4:
        return None
    left, top, right, bottom = coords
    left = max(0, left)
    top = max(0, top)
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return image.crop((left, top, right, bottom))


def cells_to_dataframe(cells: Iterable[dict]) -> pd.DataFrame:
    if not cells:
        return pd.DataFrame()

    max_row = 0
    max_col = 0
    normalised_cells = []

    for cell in cells:
        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        row_span = int(cell.get("row_span", 1) or 1)
        col_span = int(cell.get("col_span", 1) or 1)
        text = (cell.get("text") or "").strip()

        max_row = max(max_row, row + row_span)
        max_col = max(max_col, col + col_span)

        normalised_cells.append(
            {
                "row": row,
                "col": col,
                "row_span": row_span,
                "col_span": col_span,
                "text": text,
            }
        )

    data = [["" for _ in range(max_col)] for _ in range(max_row)]
    for cell in normalised_cells:
        for r in range(cell["row"], cell["row"] + cell["row_span"]):
            for c in range(cell["col"], cell["col"] + cell["col_span"]):
                current = data[r][c]
                text = cell["text"]
                if current:
                    data[r][c] = f"{current} {text}".strip()
                else:
                    data[r][c] = text

    return pd.DataFrame(data)


def ensure_directory(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


__all__ = ["TableManager", "TableManagerConfig", "TableResult"]
