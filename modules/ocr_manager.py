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
from .engines import SuryaOCREngine, OCREngine
from .image_utils import detect_handwriting_probability, enhance_image, deskew_image, denoise_image
from .paddle_singleton import get_ppstructure_v3_instance

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

# Windows-specific DLL handling for PyTorch/PaddleOCR
if os.name == "nt":
    try:
        import sys
        # Identify the probable location of the torch DLLs
        possible_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        if possible_torch_lib.exists():
            os.add_dll_directory(str(possible_torch_lib))
            
        # Add CUDA bin directory for cuDNN (PaddleOCR fix for error 126)
        possible_cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")
        if possible_cuda_bin.exists():
            os.add_dll_directory(str(possible_cuda_bin))
            # Some Paddle components bypass os.add_dll_directory; we also update PATH
            os.environ["PATH"] = str(possible_cuda_bin) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# Selective backends for OCR engines are loaded lazily via get_paddle_ocr() singleton.
# However, we need top-level imports for device detection methods below.
try:
    from paddleocr import PaddleOCR  # type: ignore
except (ImportError, OSError):
    PaddleOCR = None

try:
    import paddle  # type: ignore
except (ImportError, OSError):
    paddle = None

try:
    import easyocr  # type: ignore
except (ImportError, OSError):
    easyocr = None

try:
    import torch  # type: ignore
except (ImportError, OSError):
    torch = None

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:
    convert_from_path = None


DEFAULT_LANGUAGES: Tuple[str, ...] = ("spa", "eng")
_LOGGER = logging.getLogger(__name__)

_PADDLE_SINGLETON: Optional["PaddleOCR"] = None
_PADDLE_CONFIG: Dict[str, Any] | None = None


def get_paddle_ocr(lang: str, use_gpu: bool, gpu_id: int = 0, **kwargs: Any) -> Optional["PaddleOCR"]:
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

    if requested_cuda:
        params["use_gpu"] = True
        params["gpu_id"] = gpu_id
        # Note: In some versions of PaddleOCR, this might be use_cuda instead of use_gpu
        # but the internal logic usually handles it.

    # merge any other custom kwargs
    if kwargs:
        params.update(kwargs)

    # For multi-GPU, we can't use a single global singleton easily if they need different GPUs
    # However, if it's the SAME GPU_ID, we can reuse.
    # We'll use a dictionary of singletons keyed by (lang, gpu_id)
    global _PADDLE_SINGLETON_MAP
    if "_PADDLE_SINGLETON_MAP" not in globals():
        globals()["_PADDLE_SINGLETON_MAP"] = {}

    key = (lang, gpu_id if requested_cuda else -1)
    
    if key not in globals()["_PADDLE_SINGLETON_MAP"]:
        try:
            instance = PaddleOCR(**params)
            globals()["_PADDLE_SINGLETON_MAP"][key] = instance
            _LOGGER.info(f"PaddleOCR loaded successfully for {key}.")
        except Exception as exc:
            _LOGGER.error(f"Failed to initialise PaddleOCR for {key}: {exc}")
            return None
    
    return globals()["_PADDLE_SINGLETON_MAP"][key]



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
        gpu_id: int = 0,
    ) -> None:
        self.config = config or OCRConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = self.config.enabled
        self.gpu_id = gpu_id

        self.engine_configs: Dict[str, Dict[str, object]] = {
            key.lower(): value for key, value in (self.config.engine_configs or {}).items()
        }
        self.primary_engine = self.config.primary_engine.lower()
        self.secondary_engine = self.config.secondary_engine.lower()
        self.languages = tuple(self.config.languages) if self.config.languages else DEFAULT_LANGUAGES
        self.poppler_path = self.engine_configs.get("poppler_path")

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
            self.logger.error("OCR initialization failed: PaddleOCR is %s, EasyOCR is %s", 
                               type(self._paddle_ocr), type(self._easy_reader))
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

            #  Preprocess image for better accuracy
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
            # PPStructureV3 returns a list of dictionaries (one for each detected block)
            # We extract text from all blocks to provide a full page representation.
            results = self._paddle_ocr(array)
        except Exception as exc:  # pragma: no cover - Paddle runtime errors
            self.logger.error("PaddleOCR (PPStructureV3) failed: %s", exc)
            return "", 0.0

        texts: List[str] = []
        confidences: List[float] = []

        if not results:
            return "", 0.0

        for block in results:
            # Structural blocks have a 'res' key containing detection/recognition results
            res = block.get("res")
            if not res or not isinstance(res, list):
                continue
            
            # Format depends on block type, but for text/table it's usually a list of lines
            for item in res:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                # item format: [box, (text, score)]
                data = item[1]
                if not isinstance(data, (list, tuple)) or not data:
                    continue
                
                text = (str(data[0]) or "").strip()
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
        if not self._paddle_enabled:
            return
        if self.primary_engine != "paddleocr" and self.secondary_engine != "paddleocr":
            return
        try:
            # Strictly use PPStructureV3 from the singleton as requested.
            # No legacy hacks, no version detection, no direct use_gpu/gpu_id arguments.
            self._paddle_ocr = get_ppstructure_v3_instance()
            if self._paddle_ocr is not None:
                self.logger.info("PaddleOCR (PPStructureV3) initialised via singleton.")
                if self._paddle_use_gpu:
                    loguru_logger.success("PaddleOCR (GPU) initialized successfully.")
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
                    self.logger.info(" Surya OCR Engine initialized.")
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
            # easyocr.Reader doesn't always support gpu_id in all versions.
            # It uses the current torch device.
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
        self.logger.info(" GPU available: %s (%d GPUs detected)", has_cuda, gpu_count)
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
        print(f"DEBUG: _load_document_images called for {path}")
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            if convert_from_path is None:
                raise RuntimeError(
                    "pdf2image is required for PDF OCR but is not installed"
                )
            
            # Use poppler_path if provided in config
            kwargs = {}
            if self.poppler_path:
                print(f"DEBUG: Using poppler_path: {self.poppler_path}")
                self.logger.info(f"PDF OCR: Using poppler_path: {self.poppler_path}")
                
                # Cross-platform check
                pdfinfo_bin = "pdfinfo.exe" if os.name == 'nt' else "pdfinfo"
                pdfinfo_path = os.path.join(self.poppler_path, pdfinfo_bin)
                
                if os.path.exists(pdfinfo_path):
                    print(f"DEBUG: Verified {pdfinfo_bin} at {pdfinfo_path}")
                    self.logger.info(f"PDF OCR: Verified {pdfinfo_bin} at {pdfinfo_path}")
                    kwargs["poppler_path"] = self.poppler_path
                    
                    # Fix for Windows: Ensure poppler bin is in PATH for DLL loading
                    if os.name == 'nt':
                        import os
                        current_path = os.environ.get("PATH", "")
                        if self.poppler_path not in current_path:
                            os.environ["PATH"] = self.poppler_path + os.pathsep + current_path
                else:
                    self.logger.error(f"PDF OCR: pdfinfo.exe NOT FOUND at {pdfinfo_path}")
            else:
                self.logger.warning("PDF OCR: No poppler_path provided in config")
                
            pages = convert_from_path(path, **kwargs)
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
            # 0. Phase 9: Auto-Enhancement
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
            
            # 2. Denoise (Centralized)
            img_np = denoise_image(img_np)

            # 3. Sharpening (Unsharp Masking style)
            gaussian = cv2.GaussianBlur(img_np, (0, 0), 3.0)
            img_np = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)
            
            # 4. Deskew (Centralized)
            img_np, angle = deskew_image(img_np)
            if abs(angle) > 0.5:
                self.logger.info(f"📐 Fixed skew: {angle:.2f}°")

            # Convert back to PIL RGB
            return Image.fromarray(img_np[:, :, ::-1])
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {e}")
            return pil_image


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
