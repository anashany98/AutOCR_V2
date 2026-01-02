"""
OCR management for AutOCR using PaddleOCR and EasyOCR with fusion hooks.

This module provides a unified interface for the batch processor and web
application.  PaddleOCR remains the primary engine while EasyOCR supplies a
GPU-capable fallback.  Fusion strategies are delegated to ``FusionManager``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger as loguru_logger
from PIL import Image
import cv2

from .lang_map import map_code, map_codes
from .paddle_models import ensure_ppocrv4_models
from .paddle_models import ensure_ppocrv4_models
from .engines import SuryaOCREngine, OCREngine
from .image_utils import detect_handwriting_probability, enhance_image

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore

try:
    import paddle  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    paddle = None  # type: ignore

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


DEFAULT_LANGUAGES: Tuple[str, ...] = ("spa", "eng")
_LOGGER = logging.getLogger(__name__)

_PADDLE_SINGLETON: Optional["PaddleOCR"] = None
_PADDLE_CONFIG: Dict[str, Any] | None = None


def get_paddle_ocr(lang: str, use_gpu: bool, **kwargs: Any) -> Optional["PaddleOCR"]:
    """
    Lazily create and reuse a single PaddleOCR instance per process.
    Compatible with PaddleOCR >=3.2 (use_cuda instead of use_gpu, show_log removed).
    """
    global _PADDLE_SINGLETON, _PADDLE_CONFIG

    if PaddleOCR is None:
        return None

    requested_cuda = bool(use_gpu)
    params: Dict[str, Any] = {
        "use_angle_cls": kwargs.pop("use_angle_cls", True),
        "lang": lang,
    }

    # PaddleOCR >=3.3.0 handles GPU automatically - no explicit use_cuda parameter needed
    # GPU detection is automatic based on CUDA availability

    # merge any other custom kwargs
    if kwargs:
        params.update(kwargs)

    if _PADDLE_SINGLETON is None:
        try:
            _PADDLE_SINGLETON = PaddleOCR(**params)
            _PADDLE_CONFIG = {"lang": params["lang"], "use_cuda": requested_cuda}
            _LOGGER.info("PaddleOCR loaded successfully.")
            if requested_cuda:
                loguru_logger.success("PaddleOCR (GPU) initialized successfully.")
        except RuntimeError as exc:
            if "PDX has already been initialized" in str(exc):
                _LOGGER.warning("PaddleOCR already initialised; reusing existing singleton.")
            else:
                raise
        except TypeError as exc:
            _LOGGER.error("Failed to initialise PaddleOCR: %s", exc)
            return None
    else:
        params_cuda = requested_cuda if "use_cuda" not in params else bool(params.get("use_cuda"))
        if _PADDLE_CONFIG and (
            params["lang"] != _PADDLE_CONFIG.get("lang")
            or params_cuda != _PADDLE_CONFIG.get("use_cuda")
        ):
            _LOGGER.warning(
                "Requested PaddleOCR configuration (lang=%s, cuda=%s) differs from singleton (lang=%s, cuda=%s); reusing existing instance.",
                params["lang"],
                params_cuda,
                _PADDLE_CONFIG.get("lang"),
                _PADDLE_CONFIG.get("use_cuda"),
            )

    return _PADDLE_SINGLETON



def ocr_text_to_markdown(text: str) -> str:
    """
    Convert plain OCR text to lightweight Markdown.
    """
    if not text:
        return ""

    lines = text.splitlines()
    markdown_lines: List[str] = []
    paragraph: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if paragraph:
                markdown_lines.append(_normalise_whitespace(" ".join(paragraph)))
                markdown_lines.append("")
                paragraph = []
            continue

        if _looks_like_heading(line):
            if paragraph:
                markdown_lines.append(_normalise_whitespace(" ".join(paragraph)))
                markdown_lines.append("")
                paragraph = []
            markdown_lines.append(f"## {line}")
            markdown_lines.append("")
        else:
            paragraph.append(line)

    if paragraph:
        markdown_lines.append(_normalise_whitespace(" ".join(paragraph)))

    markdown = "\n".join(markdown_lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


@dataclass
class OCRConfig:
    """Configuration options for :class:`OCRManager`."""

    enabled: bool = True
    languages: Sequence[str] = DEFAULT_LANGUAGES
    primary_engine: str = "paddleocr"
    secondary_engine: str = "easyocr"
    fusion_strategy: str = "levenshtein"
    min_confidence_primary: float = 0.6
    confidence_margin: float = 0.05
    min_similarity: float = 0.82
    engine_configs: Dict[str, Dict[str, object]] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)


class OCRManager:
    """
    Manage OCR extraction using PaddleOCR (GPU/CPU) with EasyOCR fallback.
    """

    def __init__(
        self,
        config: OCRConfig | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or OCRConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = self.config.enabled

        self.engine_configs: Dict[str, Dict[str, object]] = {
            key.lower(): value for key, value in (self.config.engine_configs or {}).items()
        }
        self.primary_engine = self.config.primary_engine.lower()
        self.secondary_engine = self.config.secondary_engine.lower()
        self.languages = tuple(self.config.languages) if self.config.languages else DEFAULT_LANGUAGES

        paddle_conf = self.engine_configs.get("paddleocr", {})
        easy_conf = self.engine_configs.get("easyocr", {})

        self._paddle_enabled = bool(paddle_conf.get("enabled", True))
        self._easy_enabled = bool(easy_conf.get("enabled", True))

        self._paddle_use_gpu = (
            self._determine_paddle_gpu(bool(paddle_conf.get("gpu", True))) if self._paddle_enabled else False
        )
        self._easy_use_gpu = (
            self._determine_easy_gpu(bool(easy_conf.get("gpu", True))) if self._easy_enabled else False
        )
        self.use_gpu = self._paddle_use_gpu or self._easy_use_gpu

        self._paddle_lang = map_code(str(paddle_conf.get("lang", self.languages[0] if self.languages else "spa")))
        easy_langs = easy_conf.get("langs")
        self._easy_langs = map_codes(easy_langs if isinstance(easy_langs, (list, tuple)) else self.languages)

        self._paddle_model_variant = str(paddle_conf.get("model_variant", "ppocrv4")).lower()
        self._paddle_model_profile = str(paddle_conf.get("model_profile", "latin")).lower()
        self._paddle_model_storage = str(paddle_conf.get("model_storage_dir", os.path.join("models", "paddle")))
        self._paddle_auto_models = bool(paddle_conf.get("autodownload_models", True))
        self._paddle_model_dirs: Dict[str, str] = {}
        self._prepare_paddle_models(paddle_conf)

        self._paddle_ocr: Optional[PaddleOCR] = None
        self._easy_reader: Optional[object] = None
        self.extra_engines: Dict[str, OCREngine] = {}
        
        self._initialise_paddle()
        self._initialise_easy()
        self._initialise_extra_engines()

        if not self._paddle_ocr and not self._easy_reader:
            raise RuntimeError(
                "Neither PaddleOCR nor EasyOCR is available. Install at least one OCR backend."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_text(self, file_path: str, min_confidence: Optional[float] = None) -> Tuple[str, Optional[str], float, bool]:
        """
        Extract text (and aggregated confidence) from a file.
        Returns: (text, language, confidence, is_handwritten)
        :param min_confidence: Optional override for primary confidence threshold.
        """
        if not self.enabled:
            return "", None, 0.0, False

        images = self._load_document_images(file_path)
        page_texts: List[str] = []
        confidences: List[float] = []
        is_handwritten_scores: List[float] = []

        for image in images:
            # Check handwriting first on original image
            hw_prob = detect_handwriting_probability(image)
            is_handwritten_scores.append(hw_prob)

            # ✅ Preprocess image for better accuracy
            processed_image = self._preprocess_image(image)
            
            text, confidence = self._run_primary_engine(processed_image)
            if (
                self.secondary_engine
                and self.secondary_engine != self.primary_engine
                and confidence < (min_confidence if min_confidence is not None else self.config.min_confidence_primary)
            ):
                secondary_text, secondary_conf = self._run_secondary_engine(image)
                if secondary_text and secondary_conf > confidence:
                    text, confidence = secondary_text, secondary_conf

            page_texts.append(text)
            confidences.append(confidence)

        aggregated_text = "\n".join(part for part in page_texts if part).strip()
        aggregated_conf = float(np.mean(confidences)) if confidences else 0.0
        
        # Aggregate handwriting probability (max score across pages)
        is_handwritten = max(is_handwritten_scores) > 0.5 if is_handwritten_scores else False
        
        language = self.languages[0] if aggregated_text else None
        return aggregated_text, language, aggregated_conf, is_handwritten

    def extract_text_with_markdown(
        self,
        file_path: str,
    ) -> Tuple[str, str, Optional[str], float, bool]:
        """
        Extract text and convert it to Markdown, returning both representations.
        Returns: (text, markdown, language, confidence, is_handwritten)
        """
        text, language, confidence, is_handwritten = self.extract_text(file_path)
        markdown = ocr_text_to_markdown(text) if text else ""
        return text, markdown, language, confidence, is_handwritten

    def extract_block(
        self,
        image: Image.Image,
        bbox: Sequence[int],
        engine: str = "primary",
        min_confidence: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Run OCR on a cropped region defined by ``bbox``.
        """
        crop = self._crop(image, bbox)
        if crop is None:
            return "", 0.0

        engine = engine.lower()
        if engine == "primary":
            return self._run_primary_engine(crop)
        if engine == "secondary":
            return self._run_secondary_engine(crop)
        if engine == "paddleocr":
            return self._run_paddle(crop)
        if engine == "easyocr":
            return self._run_easy(crop)

        self.logger.warning("Unknown OCR engine '%s'; defaulting to primary.", engine)
        return self._run_primary_engine(crop)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_primary_engine(self, image: Image.Image) -> Tuple[str, float]:
        engine_to_use = self.primary_engine

        # Smart Routing Logic
        if engine_to_use == "auto":
            analysis = self._analyze_image_content(image)
            if analysis == "table":
                # Tables -> Prefer Surya > Paddle > Easy
                if "surya" in self.extra_engines:
                    engine_to_use = "surya"
                elif self._paddle_ocr:
                    engine_to_use = "paddleocr"
                else:
                    engine_to_use = "easyocr"
            else:
                # Default Text -> Paddle > Easy > Surya
                if self._paddle_ocr:
                    engine_to_use = "paddleocr"
                elif self._easy_reader:
                    engine_to_use = "easyocr"
                elif "surya" in self.extra_engines:
                    engine_to_use = "surya"
                
            self.logger.debug(f"🤖 Smart Routing: Content='{analysis}' -> Selected Engine='{engine_to_use}'")

        if engine_to_use == "surya" and "surya" in self.extra_engines:
             return self.extra_engines["surya"].extract_text(image)

        if engine_to_use == "paddleocr" and self._paddle_ocr:
            text, conf = self._run_paddle(image)
            if text:
                return text, conf
        if engine_to_use == "easyocr" and self._easy_reader:
            return self._run_easy(image)
            
        # Fallbacks if selected engine failed or wasn't available
        if self._paddle_ocr:
            return self._run_paddle(image)
        if self._easy_reader:
            return self._run_easy(image)
        return "", 0.0

    def _run_secondary_engine(self, image: Image.Image) -> Tuple[str, float]:
        if self.secondary_engine == "easyocr" and self._easy_reader:
            return self._run_easy(image)
        if self.secondary_engine == "paddleocr" and self._paddle_ocr:
            return self._run_paddle(image)
        if self._easy_reader and self.secondary_engine != "easyocr":
            return self._run_easy(image)
        if self._paddle_ocr and self.secondary_engine != "paddleocr":
            return self._run_paddle(image)
        return "", 0.0

    def _run_paddle(self, image: Image.Image) -> Tuple[str, float]:
        if not self._paddle_ocr:
            return "", 0.0

        array = self._pil_to_np(image)
        try:
            result = self._paddle_ocr.ocr(array)  # cls argument causing issues with current version/config
        except Exception as exc:  # pragma: no cover - Paddle runtime errors
            self.logger.error("PaddleOCR failed: %s", exc)
            return "", 0.0

        texts: List[str] = []
        confidences: List[float] = []

        for line in result:
            for _, data in line:
                text = (data[0] or "").strip()
                conf = float(data[1]) if len(data) > 1 and data[1] is not None else 0.0
                if text:
                    texts.append(text)
                    confidences.append(conf)

        if not texts:
            return "", 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(texts), avg_conf

    def _run_easy(self, image: Image.Image) -> Tuple[str, float]:
        if not self._easy_reader:
            return "", 0.0

        array = np.array(image.convert("RGB"))
        try:
            result = self._easy_reader.readtext(array, detail=1)
        except Exception as exc:  # pragma: no cover - EasyOCR runtime errors
            self.logger.error("EasyOCR failed: %s", exc)
            return "", 0.0

        texts: List[str] = []
        confidences: List[float] = []

        for item in result:
            if len(item) < 2:
                continue
            text = (item[1] or "").strip()
            conf_value = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0
            if text:
                texts.append(text)
                confidences.append(conf_value)

        if not texts:
            return "", 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(texts), avg_conf

    def _initialise_paddle(self) -> None:
        if not self._paddle_enabled or PaddleOCR is None:
            return
        if self.primary_engine != "paddleocr" and self.secondary_engine != "paddleocr":
            return
        try:
            kwargs: Dict[str, Any] = {"use_angle_cls": True}
            if self._paddle_model_dirs:
                kwargs.update({k: v for k, v in self._paddle_model_dirs.items() if v})
            self._paddle_ocr = get_paddle_ocr(
                lang=self._paddle_lang,
                use_gpu=self._paddle_use_gpu,
                **kwargs,
            )
            if self._paddle_ocr is not None:
                self.logger.info("PaddleOCR initialised (lang=%s, gpu=%s).", self._paddle_lang, self._paddle_use_gpu)
                if self._paddle_use_gpu:
                    loguru_logger.success("PaddleOCR (GPU) initialized successfully.")
        except AttributeError as exc:
            if "set_optimization_level" in str(exc):
                self.logger.warning("PaddleOCR initialization failed due to PaddlePaddle API incompatibility (set_optimization_level). This is expected with PaddlePaddle 2.6.1 and PaddleOCR 3.3.0. PaddleOCR will be disabled.")
            else:
                self.logger.warning("Failed to initialise PaddleOCR due to AttributeError: %s", exc)
            self._paddle_ocr = None
        except Exception as exc:  # pragma: no cover - Paddle runtime errors
            self.logger.warning("Failed to initialise PaddleOCR: %s", exc)
            self._paddle_ocr = None

    def _initialise_extra_engines(self) -> None:
        """Initialize any additional engines defined in config (e.g. Surya)."""
        surya_conf = self.engine_configs.get("surya", {})
        if surya_conf.get("enabled", False):
            try:
                surya = SuryaOCREngine(surya_conf, logger=self.logger)
                if surya.initialize():
                    self.extra_engines["surya"] = surya
                    self.logger.info("✅ Surya OCR Engine initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Surya OCR: {e}")

    def _prepare_paddle_models(self, paddle_conf: Dict[str, Any]) -> None:
        if not self._paddle_enabled:
            return

        explicit_dirs = {
            "det_model_dir": paddle_conf.get("det_model_dir"),
            "rec_model_dir": paddle_conf.get("rec_model_dir"),
            "cls_model_dir": paddle_conf.get("cls_model_dir"),
        }
        if any(explicit_dirs.values()):
            self._paddle_model_dirs = {k: str(v) for k, v in explicit_dirs.items() if v}
            return

        if self._paddle_model_variant != "ppocrv4" or not self._paddle_auto_models:
            return

        try:
            self._paddle_model_dirs = ensure_ppocrv4_models(
                base_dir=self._paddle_model_storage,
                profile=self._paddle_model_profile,
            )
            self.logger.info(
                "PP-OCRv4 models ready in %s (profile=%s)",
                self._paddle_model_storage,
                self._paddle_model_profile,
            )
        except Exception as exc:
            self.logger.warning("Automatic PP-OCRv4 model setup failed: %s", exc)

    def _initialise_easy(self) -> None:
        if not self._easy_enabled or easyocr is None:
            return
        if self.primary_engine != "easyocr" and self.secondary_engine != "easyocr":
            return
        try:
            self._easy_reader = easyocr.Reader(self._easy_langs, gpu=self._easy_use_gpu)  # type: ignore[arg-type]
            self.logger.info(
                "EasyOCR initialised (langs=%s, gpu=%s).",
                ",".join(self._easy_langs),
                self._easy_use_gpu,
            )
            if self._easy_use_gpu:
                loguru_logger.success("EasyOCR (GPU) initialized successfully.")
        except Exception as exc:  # pragma: no cover - EasyOCR runtime errors
            self.logger.warning("Failed to initialise EasyOCR: %s", exc)
            self._easy_reader = None

    def _determine_paddle_gpu(self, requested: bool) -> bool:
        has_cuda = False
        gpu_count = 0
        if paddle is not None:
            try:
                has_cuda = bool(paddle.device.is_compiled_with_cuda())
                gpu_count = paddle.device.cuda.device_count() if has_cuda else 0
                has_cuda = has_cuda and gpu_count > 0
            except Exception:  # pragma: no cover - defensive
                has_cuda = False
                gpu_count = 0
        else:
            self.logger.info("Paddle not installed with GPU support; using CPU.")
        self.logger.info("🧠 GPU available: %s (%d GPUs detected)", has_cuda, gpu_count)
        if not requested:
            return False
        if has_cuda:
            self.logger.info("CUDA detected; enabling PaddleOCR GPU execution on %d GPU(s).", gpu_count)
            return True
        if requested and not has_cuda:
            self.logger.info("CUDA not available; PaddleOCR will run on CPU.")
        return False

    def _determine_easy_gpu(self, requested: bool) -> bool:
        if not requested:
            return False
        if torch is None:
            self.logger.info("PyTorch not installed with CUDA support; EasyOCR will run on CPU.")
            return False
        try:
            if torch.cuda.is_available():  # type: ignore[union-attr]
                gpu_count = torch.cuda.device_count()  # type: ignore[union-attr]
                self.logger.info("CUDA detected; enabling EasyOCR GPU execution on %d GPU(s).", gpu_count)
                return True
        except Exception:  # pragma: no cover - defensive
            pass
        self.logger.info("CUDA not available; EasyOCR will run on CPU.")
        return False

    @staticmethod
    def _pil_to_np(image: Image.Image) -> np.ndarray:
        array = np.array(image.convert("RGB"))
        return array[:, :, ::-1]

    @staticmethod
    def _crop(image: Image.Image, bbox: Sequence[int]) -> Optional[Image.Image]:
        coords = list(bbox)
        if len(coords) != 4:
            return None
        left, top, right, bottom = coords
        width, height = image.size
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))

    def _load_document_images(self, path: str) -> List[Image.Image]:
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            if convert_from_path is None:
                raise RuntimeError(
                    "pdf2image is required for PDF OCR but is not installed"
                )
            pages = convert_from_path(path)
            return [page.convert("RGB") for page in pages]

        with Image.open(path) as img:
            frames: List[Image.Image] = []
            n_frames = getattr(img, "n_frames", 1)
            for frame in range(n_frames):
                try:
                    img.seek(frame)
                except EOFError:
                    break
                frames.append(img.convert("RGB"))
            return [frame.copy() for frame in frames] or [img.convert("RGB")]

    def _preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Enhance image for OCR: upscale (if small), denoise, and sharpen.
        """
        try:
            # 0. Phase 9: Auto-Enhancement (User Request: "cientos de gigas")
            # If configured, we apply PIL-based enhancements BEFORE converting to OpenCV BGR
            pre_conf = self.config.preprocessing
            if pre_conf.get("auto_enhance", False):
                pil_image = enhance_image(
                    pil_image,
                    contrast=float(pre_conf.get("contrast", 1.0)),
                    brightness=float(pre_conf.get("brightness", 1.0)),
                    sharpness=float(pre_conf.get("sharpness", 1.0)),
                    apply_clahe=bool(pre_conf.get("apply_clahe", False))
                )

            # Convert PIL to BGR (OpenCV format)
            img_np = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()

            # 1. Upscale if too small (width < 1500px)
            height, width = img_np.shape[:2]
            if width < 1500:
                scale = 1500 / width
                img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # 2. Denoise (Fast Non-Local Means) - moderate strength
            img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)

            # 3. Sharpening (Unsharp Masking style)
            gaussian = cv2.GaussianBlur(img_np, (0, 0), 3.0)
            img_np = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)
            
            # 4. Deskew (Straighten)
            img_np = self._deskew(img_np)

            # Convert back to PIL RGB
            return Image.fromarray(img_np[:, :, ::-1])
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {e}")
            return pil_image

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detect text orientation and rotate image to straighten it.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle for cv2.minAreaRect idiosyncrasies
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Only rotate if significant skew detected (> 0.5 deg) to avoid blurring straight docs
            if abs(angle) > 0.5 and abs(angle) < 89: # Skip if it looks like 90deg rotation (handled by orientation cls)
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                self.logger.info(f"📐 Fixed skew: {angle:.2f}°")
                return rotated
            return img
        except Exception as e:
            self.logger.debug(f"Skew correction skipped: {e}")
            return img

    def _analyze_image_content(self, pil_image: Image.Image) -> str:
        """
        Analyze image to determine content type (table, text, noise).
        Returns: 'table', 'text', or 'noise'
        """
        try:
            img_np = np.array(pil_image.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 1. Detect Lines (Tables)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                # If we see substantial horizontal/vertical lines, it's likely a table/form
                return "table"
                
            return "text"
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {e}")
            return "text"


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_heading(line: str) -> bool:
    if len(line) <= 2:
        return False
    if len(line) < 100 and (line.isupper() or line.endswith(":")):
        return True
    if len(line.split()) <= 6 and line == line.title():
        return True
    return False


__all__ = ["OCRManager", "OCRConfig", "ocr_text_to_markdown", "get_paddle_ocr"]
