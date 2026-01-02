"""
Layout detection management for AutOCR.

This module wraps PaddleOCR's layout analysis to provide a consistent block
representation for downstream processing. When PaddleOCR is not available the
manager falls back to a single "text" block per page so the rest of the
pipeline can continue to operate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, TypedDict

import numpy as np
from PIL import Image

from .paddle_singleton import get_paddle_ocr

# Disable PaddleOCR's Doc-VLM components to avoid reinitialising PaddleX in multi-import scenarios.
os.environ.setdefault("PADDLEOCR_DISABLE_VLM", "1")

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore


class LayoutBlock(TypedDict):
    """Structure describing a detected layout block."""

    bbox: List[int]
    type: str
    page: int
    rotation: float
    confidence: float


_NORMALISED_LABELS = {
    "text": "text",
    "title": "title",
    "table": "table",
    "figure": "figure",
    "header": "title",
    "footer": "other",
    "caption": "text",
    "list": "text",
    "figure_caption": "figure",
    "equation": "other",
}

@dataclass
class LayoutManagerConfig:
    """Configuration for :class:`LayoutManager`."""

    use_gpu: bool = False
    languages: Sequence[str] | None = None
    detector: str = "paddleocr"


class LayoutManager:
    """
    Run layout detection on PDF or image documents.

    Parameters
    ----------
    config:
        Settings describing which detector to use and whether GPU should be
        enabled.
    logger:
        Optional logger used for diagnostics.
    """

    def __init__(
        self,
        config: LayoutManagerConfig | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or LayoutManagerConfig()
        self._engine = None
        self._initialise_engine()

    def detect_blocks(
        self,
        image_or_pdf_path: str,
        pages: Optional[Sequence[Image.Image]] = None,
    ) -> List[LayoutBlock]:
        """
        Detect layout blocks for the given file.

        Returns
        -------
        list[LayoutBlock]
            Normalised list of blocks with bounding boxes and metadata.
        """
        page_images = list(pages) if pages is not None else self._load_document_images(image_or_pdf_path)
        blocks: List[LayoutBlock] = []

        for page_index, page_image in enumerate(page_images):
            if self._engine is None:
                width, height = page_image.size
                blocks.append(
                    LayoutBlock(
                        bbox=[0, 0, width, height],
                        type="text",
                        page=page_index,
                        rotation=0.0,
                        confidence=1.0,
                    )
                )
                continue

            page_array = self._pil_to_np(page_image)
            try:
                results = self._engine(page_array)
            except Exception as exc:  # pragma: no cover - Paddle runtime errors
                self.logger.error(
                    "Layout detection failed on page %s of %s: %s",
                    page_index,
                    os.path.basename(image_or_pdf_path),
                    exc,
                )
                width, height = page_image.size
                blocks.append(
                    LayoutBlock(
                        bbox=[0, 0, width, height],
                        type="text",
                        page=page_index,
                        rotation=0.0,
                        confidence=0.0,
                    )
                )
                continue

            for item in results:
                raw_label = item.get("type", "text")
                label = _NORMALISED_LABELS.get(raw_label, "other")
                bbox = _ensure_bbox(item.get("bbox"))
                conf = float(item.get("score", 0.0) or item.get("confidence", 0.0) or 0.0)
                rotation = float(item.get("rotation", 0.0) or 0.0)
                if not bbox:
                    continue
                blocks.append(
                    LayoutBlock(
                        bbox=bbox,
                        type=label,
                        page=page_index,
                        rotation=rotation,
                        confidence=conf,
                    )
                )

        return blocks

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _initialise_engine(self) -> None:
        if self.config.detector != "paddleocr":
            self._engine = None
            return

        try:
            engine = get_paddle_ocr()
        except Exception as exc:  # pragma: no cover - Paddle runtime errors
            self.logger.warning(
                "Failed to initialise PaddleOCR layout detector (%s); falling "
                "back to naive blocks.",
                exc,
            )
            self._engine = None
            return

        self._engine = engine

    def _load_document_images(self, path: str) -> List[Image.Image]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            if convert_from_path is None:
                raise RuntimeError(
                    "pdf2image is required for PDF layout detection but is not installed"
                )
            pages = convert_from_path(path)
            return [page.convert("RGB") for page in pages]

        image = Image.open(path)
        images = []
        try:
            n_frames = getattr(image, "n_frames", 1)
        except Exception:
            n_frames = 1

        if n_frames > 1:
            for idx in range(n_frames):
                image.seek(idx)
                images.append(image.convert("RGB"))
        else:
            images.append(image.convert("RGB"))

        return images

    @staticmethod
    def _pil_to_np(image: Image.Image) -> np.ndarray:
        array = np.array(image)
        if array.ndim == 2:
            return np.stack([array, array, array], axis=-1)
        if array.shape[2] == 4:
            # Drop alpha channel
            array = array[:, :, :3]
        # Convert RGB -> BGR for Paddle expectations
        return array[:, :, ::-1].copy()


def _ensure_bbox(value: Optional[Iterable[float]]) -> List[int]:
    if value is None:
        return []
    numbers = list(value)
    if len(numbers) == 4:
        return [int(round(n)) for n in numbers]
    if len(numbers) == 8:
        xs = numbers[0::2]
        ys = numbers[1::2]
        return [
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        ]
    return []


__all__ = ["LayoutManager", "LayoutManagerConfig", "LayoutBlock"]
