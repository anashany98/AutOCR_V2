from .base import OCREngine
from .surya_wrapper import SuryaOCREngine
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Import only for type checkers; runtime import is lazy to avoid importing torch/transformers unless needed.
    from .paddle_vl_wrapper import PaddleVLOCHEngine  # noqa: F401

__all__ = ["OCREngine", "SuryaOCREngine", "PaddleVLOCHEngine"]


def __getattr__(name: str):  # pragma: no cover - import-time utility
    if name == "PaddleVLOCHEngine":
        from .paddle_vl_wrapper import PaddleVLOCHEngine as _PaddleVLOCHEngine

        return _PaddleVLOCHEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
