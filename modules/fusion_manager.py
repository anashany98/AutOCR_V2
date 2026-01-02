"""
Fusion logic for combining OCR results from multiple engines.

Supports confidence voting, cascading and Levenshtein-based similarity merges.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

try:
    from rapidfuzz import fuzz  # type: ignore

    def levenshtein_ratio(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100.0
except ImportError:  # pragma: no cover - fallback for missing dependency
    from difflib import SequenceMatcher

    logging.getLogger(__name__).warning(
        "rapidfuzz not installed; falling back to difflib ratio."
    )

    def levenshtein_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


@dataclass
class FusionConfig:
    """Configuration parameters for :class:FusionManager."""

    strategy: str = "confidence_vote"
    min_confidence_primary: float = 0.6
    confidence_margin: float = 0.05
    min_similarity: float = 0.82
    priority: Sequence[str] = field(default_factory=lambda: ("paddleocr", "easyocr"))


DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
AMOUNT_PATTERN = re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b")
CIF_NIF_PATTERN = re.compile(r"\b[A-Z]\d{7}[A-Z]\b", re.IGNORECASE)
INVOICE_PATTERN = re.compile(r"\b(?:invoice|factura)[- ]?\d+\b", re.IGNORECASE)


class FusionManager:
    """
    Combine primary and secondary OCR results using configurable strategies.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        text_primary: Optional[str],
        conf_primary: Optional[float],
        text_secondary: Optional[str],
        conf_secondary: Optional[float],
        heuristics: Optional[dict] = None,
    ) -> Tuple[str, float]:
        """Fuse OCR outputs from two engines."""

        primary_text = (text_primary or "").strip()
        secondary_text = (text_secondary or "").strip()
        primary_conf = float(conf_primary or 0.0)
        secondary_conf = float(conf_secondary or 0.0)

        if primary_text and not secondary_text:
            return primary_text, primary_conf
        if secondary_text and not primary_text:
            return secondary_text, secondary_conf
        if not primary_text and not secondary_text:
            return "", 0.0

        strategy = self.config.strategy

        if strategy == "cascade":
            if primary_conf >= self.config.min_confidence_primary:
                return primary_text, primary_conf
            return secondary_text, secondary_conf

        if strategy == "levenshtein":
            meta = heuristics or {}
            primary_engine = str(meta.get("primary_engine", "")).lower()
            secondary_engine = str(meta.get("secondary_engine", "")).lower()
            return self._levenshtein_choice(
                primary_text,
                primary_conf,
                secondary_text,
                secondary_conf,
                primary_engine,
                secondary_engine,
            )

        if strategy == "confidence_vote":
            if primary_conf >= self.config.min_confidence_primary:
                return primary_text, primary_conf

            if secondary_conf >= primary_conf + self.config.confidence_margin:
                return secondary_text, secondary_conf

            choice = self._heuristic_choice(
                primary_text, secondary_text, primary_conf, secondary_conf
            )
            if choice == "primary":
                return primary_text, max(primary_conf, secondary_conf)
            if choice == "secondary":
                return secondary_text, max(primary_conf, secondary_conf)

        return self._prefer_highest_confidence(
            primary_text,
            primary_conf,
            secondary_text,
            secondary_conf,
        )

    # ------------------------------------------------------------------ #
    # Internal logic
    # ------------------------------------------------------------------ #

    def _levenshtein_choice(
        self,
        primary_text: str,
        primary_conf: float,
        secondary_text: str,
        secondary_conf: float,
        primary_engine: str,
        secondary_engine: str,
    ) -> Tuple[str, float]:
        similarity = levenshtein_ratio(primary_text, secondary_text)
        if similarity >= self.config.min_similarity:
            avg_conf = (primary_conf + secondary_conf) / 2 if (primary_conf or secondary_conf) else similarity
            chosen = primary_text if len(primary_text) >= len(secondary_text) else secondary_text
            return chosen, float(avg_conf)

        for engine in self.config.priority:
            name = str(engine).lower()
            if name == primary_engine and primary_text:
                return primary_text, primary_conf
            if name == secondary_engine and secondary_text:
                return secondary_text, secondary_conf

        return self._prefer_highest_confidence(primary_text, primary_conf, secondary_text, secondary_conf)

    @staticmethod
    def _heuristic_choice(
        primary_text: str,
        secondary_text: str,
        primary_conf: float,
        secondary_conf: float,
    ) -> str:
        primary_matches = _score_patterns(primary_text)
        secondary_matches = _score_patterns(secondary_text)

        if primary_matches > secondary_matches:
            return "primary"
        if secondary_matches > primary_matches:
            return "secondary"

        if abs(primary_conf - secondary_conf) <= 0.05:
            primary_score = _alphanumeric_word_score(primary_text)
            secondary_score = _alphanumeric_word_score(secondary_text)
            if primary_score > secondary_score:
                return "primary"
            if secondary_score > primary_score:
                return "secondary"

        return "undecided"

    @staticmethod
    def _prefer_highest_confidence(
        primary_text: str,
        primary_conf: float,
        secondary_text: str,
        secondary_conf: float,
    ) -> Tuple[str, float]:
        if primary_conf >= secondary_conf:
            return primary_text, primary_conf
        return secondary_text, secondary_conf


def _score_patterns(text: str) -> int:
    score = 0
    if DATE_PATTERN.search(text):
        score += 1
    if AMOUNT_PATTERN.search(text):
        score += 1
    if CIF_NIF_PATTERN.search(text):
        score += 1
    if INVOICE_PATTERN.search(text):
        score += 1
    return score


def _alphanumeric_word_score(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return sum(len(word) for word in words)


__all__ = ["FusionManager", "FusionConfig"]
