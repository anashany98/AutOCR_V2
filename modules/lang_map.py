"""
Language mapping helpers for OCR engines.

The project configuration uses legacy codes (``spa``, ``eng``) while engines
expect ISO-639-1 identifiers.  This module centralises the conversion logic so
all components remain consistent.
"""

from __future__ import annotations

from typing import Iterable, List


LANG_MAP = {
    "spa": "es",
    "es": "es",
    "esp": "es",
    "spanish": "es",
    "eng": "en",
    "en": "en",
    "english": "en",
}


def map_code(code: str) -> str:
    """Return the engine-compatible code for ``code``."""
    normalised = (code or "").strip().lower()
    return LANG_MAP.get(normalised, normalised or "en")


def map_codes(codes: Iterable[str] | None) -> List[str]:
    """
    Convert a sequence of language codes to engine-compatible identifiers.
    """
    if not codes:
        return ["en"]
    return [map_code(code) for code in codes]


__all__ = ["LANG_MAP", "map_code", "map_codes"]
