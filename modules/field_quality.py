"""
Field extraction quality evaluation utilities.

This module computes reproducible quality metrics for the core extracted fields:
- date
- total
- supplier (vendor alias)
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modules.normalizer import DataNormalizer
from modules.smart_extractor import FieldExtractor


LEGAL_SUFFIXES = {
    "sl",
    "slu",
    "sa",
    "sau",
    "ltd",
    "limited",
    "inc",
    "corp",
    "corporation",
    "llc",
    "gmbh",
}


def _parse_amount(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d.,-]", "", text)
    if not cleaned:
        return None

    comma_count = cleaned.count(",")
    dot_count = cleaned.count(".")

    if comma_count > 0 and dot_count > 0:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif comma_count > 0 and dot_count == 0:
        if re.search(r",\d{2}$", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif dot_count > 1:
        cleaned = cleaned.replace(".", "")

    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    if not text:
        return None

    # Fast path (already ISO date)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text

    # DMY / MDY with separators.
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
    if m:
        p1, p2, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year = 2000 + year
        if p1 > 12 and p2 <= 12:
            day, month = p1, p2
        elif p2 > 12 and p1 <= 12:
            day, month = p2, p1
        else:
            # Default to D/M in this project context.
            day, month = p1, p2
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # Spanish long form: 2 de enero de 2026
    m = re.fullmatch(r"(\d{1,2})\s+de\s+([a-zA-Z]+)\s+de\s+(\d{4})", text, flags=re.IGNORECASE)
    if m:
        months = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
        }
        day = int(m.group(1))
        month = months.get(m.group(2).strip().lower())
        year = int(m.group(3))
        if month:
            try:
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return None

    return None


def _canonical_supplier(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized).lower()
    tokens = [tok for tok in normalized.split() if tok and tok not in LEGAL_SUFFIXES]
    if not tokens:
        return None
    return " ".join(tokens)


def _is_match_date(expected: Any, predicted: Any) -> bool:
    exp = _parse_date(expected)
    pred = _parse_date(predicted)
    return bool(exp and pred and exp == pred)


def _is_match_total(expected: Any, predicted: Any, abs_tol: float, rel_tol: float) -> bool:
    exp = _parse_amount(expected)
    pred = _parse_amount(predicted)
    if exp is None or pred is None:
        return False
    return math.isclose(exp, pred, abs_tol=abs_tol, rel_tol=rel_tol)


def _is_match_supplier(expected: Any, predicted: Any) -> bool:
    exp = _canonical_supplier(expected)
    pred = _canonical_supplier(predicted)
    return bool(exp and pred and exp == pred)


@dataclass
class _FieldCounter:
    expected_present: int = 0
    predicted_present: int = 0
    correct: int = 0
    sample_correct: int = 0

    def to_metrics(self, sample_count: int) -> Dict[str, Any]:
        precision = (self.correct / self.predicted_present) if self.predicted_present else 0.0
        recall = (self.correct / self.expected_present) if self.expected_present else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = (self.sample_correct / sample_count) if sample_count else 0.0
        return {
            "expected_present": self.expected_present,
            "predicted_present": self.predicted_present,
            "correct": self.correct,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "sample_accuracy": round(accuracy, 4),
        }


def load_reference_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load reference dataset from JSONL or JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Invalid JSONL row at line {i + 1}")
            rows.append(item)
        return rows

    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            return [row for row in payload["samples"] if isinstance(row, dict)]
        raise ValueError("Unsupported JSON dataset format. Use list[] or {\"samples\":[]}.")

    raise ValueError(f"Unsupported dataset extension: {p.suffix}")


def _extract_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(sample.get("expected"), dict):
        exp = dict(sample["expected"])
    else:
        exp = {}

    # Fallback aliases.
    if "supplier" not in exp and "vendor" in exp:
        exp["supplier"] = exp.get("vendor")
    if "vendor" not in exp and "supplier" in exp:
        exp["vendor"] = exp.get("supplier")
    if "date" not in exp and "expected_date" in sample:
        exp["date"] = sample.get("expected_date")
    if "total" not in exp and "expected_total" in sample:
        exp["total"] = sample.get("expected_total")
    if "supplier" not in exp and "expected_supplier" in sample:
        exp["supplier"] = sample.get("expected_supplier")

    return exp


def _extract_predicted_fields(text: str, extractor: FieldExtractor, normalizer: DataNormalizer) -> Dict[str, Any]:
    raw = extractor.extract_fields(text or "")
    normalized = normalizer.normalize(raw or {})
    supplier = None
    if isinstance(normalized.get("supplier"), dict):
        supplier = normalized["supplier"].get("value")
    elif isinstance(normalized.get("vendor"), dict):
        supplier = normalized["vendor"].get("value")
    return {
        "date": (normalized.get("date") or {}).get("value") if isinstance(normalized.get("date"), dict) else None,
        "total": (normalized.get("total") or {}).get("value") if isinstance(normalized.get("total"), dict) else None,
        "supplier": supplier,
        "raw_fields": raw,
        "normalized_fields": normalized,
    }


def evaluate_field_quality(
    dataset: Iterable[Dict[str, Any]],
    *,
    config: Optional[Dict[str, Any]] = None,
    abs_tol_total: float = 0.01,
    rel_tol_total: float = 0.001,
    max_failures: int = 25,
) -> Dict[str, Any]:
    """Evaluate core extraction quality against a reference dataset."""
    extractor = FieldExtractor(config or {})
    normalizer = DataNormalizer(config or {})

    counters = {
        "date": _FieldCounter(),
        "total": _FieldCounter(),
        "supplier": _FieldCounter(),
    }
    failures: List[Dict[str, Any]] = []
    sample_count = 0
    all_fields_exact = 0

    for idx, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            continue
        text = str(sample.get("text") or "")
        expected = _extract_expected(sample)
        predicted = _extract_predicted_fields(text, extractor, normalizer)
        sample_count += 1

        sample_ok = True
        for field in ("date", "total", "supplier"):
            exp = expected.get(field)
            pred = predicted.get(field)
            exp_present = exp not in (None, "")
            pred_present = pred not in (None, "")

            if exp_present:
                counters[field].expected_present += 1
            if pred_present:
                counters[field].predicted_present += 1

            if field == "date":
                match = _is_match_date(exp, pred) if exp_present and pred_present else (not exp_present and not pred_present)
            elif field == "total":
                match = (
                    _is_match_total(exp, pred, abs_tol=abs_tol_total, rel_tol=rel_tol_total)
                    if exp_present and pred_present
                    else (not exp_present and not pred_present)
                )
            else:
                match = _is_match_supplier(exp, pred) if exp_present and pred_present else (not exp_present and not pred_present)

            if exp_present and pred_present and match:
                counters[field].correct += 1
            if match:
                counters[field].sample_correct += 1
            else:
                sample_ok = False

        if sample_ok:
            all_fields_exact += 1
        elif len(failures) < max_failures:
            failures.append(
                {
                    "id": sample.get("id", f"sample-{idx + 1}"),
                    "expected": {
                        "date": expected.get("date"),
                        "total": expected.get("total"),
                        "supplier": expected.get("supplier"),
                    },
                    "predicted": {
                        "date": predicted.get("date"),
                        "total": predicted.get("total"),
                        "supplier": predicted.get("supplier"),
                    },
                }
            )

    metrics = {
        field: counters[field].to_metrics(sample_count)
        for field in ("date", "total", "supplier")
    }
    exact_rate = (all_fields_exact / sample_count) if sample_count else 0.0

    return {
        "samples": sample_count,
        "overall": {
            "all_fields_exact": all_fields_exact,
            "all_fields_exact_rate": round(exact_rate, 4),
        },
        "field_metrics": metrics,
        "failures": failures,
    }

