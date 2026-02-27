"""
Smart field extraction module for invoice/document processing.

Extracts normalized core fields from OCR text:
- date (ISO format)
- total (numeric amount + currency)
- vendor / supplier (company name)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import get_schema_for_type
from .vendor_matcher import VendorMatcher

logger = logging.getLogger(__name__)


class FieldExtractor:
    """Intelligent extraction of structured fields from OCR text."""

    DATE_PATTERNS = [
        (r"\b\d{4}-\d{2}-\d{2}\b", "iso"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", "dmy_or_mdy"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2}\b", "dmy_or_mdy_2digit_year"),
        (r"\b\d{1,2}\s+de\s+[a-zA-Z]+\s+de\s+\d{4}\b", "es_long"),
    ]

    AMOUNT_PATTERNS = [
        (r"(?i)(?:eur|usd|gbp|[€$£])\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})", "currency_prefix"),
        (r"(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*(?i:eur|usd|gbp|[€$£])", "currency_suffix"),
        (r"\b(\d+[.,]\d{2})\b", "plain"),
    ]

    CURRENCY_HINTS = {
        "EUR": ("EUR", "€", "EURO"),
        "USD": ("USD", "$", "DOLAR", "DOLLAR"),
        "GBP": ("GBP", "£"),
    }

    SPANISH_MONTHS = {
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

    DATE_KEYWORDS = ("fecha", "date", "emision", "issued")
    TOTAL_KEYWORDS = ("total", "importe total", "total a pagar", "amount due", "grand total")
    NEGATIVE_TOTAL_KEYWORDS = ("subtotal", "base imponible", "iva", "vat", "tax", "descuento", "discount")
    VENDOR_KEYWORDS = ("proveedor", "vendor", "empresa", "company", "razon social")
    VENDOR_STOPWORDS = (
        "factura",
        "invoice",
        "ticket",
        "fecha",
        "date",
        "total",
        "subtotal",
        "iva",
        "vat",
        "cif",
        "nif",
        "www.",
        "http://",
        "https://",
        "@",
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vendor_matcher = VendorMatcher(self.config)
        self._llm_client: Optional[Any] = None
        self._llm_client_checked = False

    @property
    def llm_client(self) -> Optional[Any]:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, value: Optional[Any]) -> None:
        # Manual assignment is treated as an explicit override.
        self._llm_client = value
        self._llm_client_checked = True

    def _get_llm_client(self) -> Optional[Any]:
        if self._llm_client_checked:
            return self._llm_client
        self._llm_client_checked = True
        try:
            from web_app.services import get_llm_client

            self._llm_client = get_llm_client()
        except Exception:
            self._llm_client = None
        return self._llm_client

    def extract_fields(self, text: str, blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Extract structured fields from OCR text."""
        text = (text or "").strip()
        if not text:
            return {}

        fields: Dict[str, Dict[str, Any]] = {}

        date_result = self._extract_date(text, blocks)
        if date_result:
            fields["date"] = date_result

        total_result = self._extract_total(text, blocks)
        if total_result:
            fields["total"] = total_result

        vendor_result = self._extract_vendor(text, blocks)
        if vendor_result:
            fields["vendor"] = vendor_result
            # Keep alias used by verification UI.
            fields["supplier"] = dict(vendor_result)

        self._maybe_fill_with_llm(text, fields)
        self._attach_validation_summary(fields)
        return fields

    def _extract_date(self, text: str, blocks: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_conf = -1.0
        lines = text.splitlines() or [text]
        total_lines = max(1, len(lines))

        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            for pattern, mode in self.DATE_PATTERNS:
                for match in re.finditer(pattern, line_clean, flags=re.IGNORECASE):
                    raw = match.group(0).strip()
                    parsed = self._parse_date(raw, mode, line_clean)
                    if not parsed:
                        continue

                    conf = 0.45
                    ll = line_clean.lower()
                    if any(k in ll for k in self.DATE_KEYWORDS):
                        conf += 0.35
                    if line_idx < total_lines * 0.35:
                        conf += 0.2

                    conf = min(conf, 1.0)
                    if conf > best_conf:
                        best_conf = conf
                        best = {
                            "value": parsed.strftime("%Y-%m-%d"),
                            "confidence": round(conf, 2),
                            "raw": raw,
                            "source": "regex",
                        }
        return best

    def _parse_date(self, date_str: str, mode: str, line_context: str) -> Optional[datetime]:
        try:
            if mode == "iso":
                return datetime.strptime(date_str, "%Y-%m-%d")

            if mode == "es_long":
                m = re.search(r"(\d{1,2})\s+de\s+([a-zA-Z]+)\s+de\s+(\d{4})", date_str, flags=re.IGNORECASE)
                if not m:
                    return None
                day = int(m.group(1))
                month_name = m.group(2).strip().lower()
                year = int(m.group(3))
                month = self.SPANISH_MONTHS.get(month_name)
                if not month:
                    return None
                return datetime(year, month, day)

            # Ambiguous slash/hyphen format: prefer D/M in this project.
            parts = re.split(r"[/-]", date_str)
            if len(parts) != 3:
                return None
            d1, d2, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 100:
                y = 2000 + y

            # Disambiguation.
            if d1 > 12 and d2 <= 12:
                day, month = d1, d2
            elif d2 > 12 and d1 <= 12:
                day, month = d2, d1
            else:
                ctx = (line_context or "").lower()
                if "us" in ctx or "mm/dd" in ctx:
                    month, day = d1, d2
                else:
                    day, month = d1, d2

            return datetime(y, month, day)
        except Exception:
            return None

    def _extract_total(
        self, text: str, blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_conf = -1.0
        lines = text.splitlines() or [text]
        total_lines = max(1, len(lines))
        fallback_max_amount: Optional[float] = None
        fallback_match: Optional[Dict[str, Any]] = None

        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue

            line_lower = line_clean.lower()
            has_total_kw = any(k in line_lower for k in self.TOTAL_KEYWORDS)
            has_negative_kw = any(k in line_lower for k in self.NEGATIVE_TOTAL_KEYWORDS)

            for pattern, fmt in self.AMOUNT_PATTERNS:
                for match in re.finditer(pattern, line_clean):
                    raw_amount = match.group(1)
                    parsed = self._parse_amount(raw_amount)
                    if parsed is None:
                        continue
                    currency = self._detect_currency(line_clean)

                    conf = 0.35
                    if has_total_kw:
                        conf += 0.45
                    if has_negative_kw:
                        conf -= 0.3
                    if line_idx > total_lines * 0.45:
                        conf += 0.15
                    if fmt != "plain":
                        conf += 0.05
                    conf = max(0.0, min(conf, 1.0))

                    candidate = {
                        "value": round(parsed, 2),
                        "confidence": round(conf, 2),
                        "raw": match.group(0).strip(),
                        "currency": currency,
                        "source": "regex",
                    }

                    if has_total_kw and conf > best_conf:
                        best_conf = conf
                        best = candidate

                    if fallback_max_amount is None or parsed > fallback_max_amount:
                        fallback_max_amount = parsed
                        fallback_match = candidate

        if best:
            return best
        if fallback_match:
            fallback_match["confidence"] = round(max(float(fallback_match.get("confidence", 0.0)), 0.45), 2)
            return fallback_match
        return None

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        # Keep only digits and separators.
        cleaned = re.sub(r"[^\d.,]", "", amount_str or "")
        if not cleaned:
            return None

        comma_count = cleaned.count(",")
        dot_count = cleaned.count(".")

        # Heuristics for decimal separator.
        if comma_count > 0 and dot_count > 0:
            if cleaned.rfind(",") > cleaned.rfind("."):
                # EU style 1.234,56
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # US style 1,234.56
                cleaned = cleaned.replace(",", "")
        elif comma_count > 0 and dot_count == 0:
            # Could be 1234,56 or 1,234
            if re.search(r",\d{2}$", cleaned):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        else:
            # only dots or plain digits
            if dot_count > 1:
                # 1.234.567 -> remove thousand separators
                cleaned = cleaned.replace(".", "")

        try:
            return float(cleaned)
        except Exception:
            return None

    def _detect_currency(self, text: str) -> str:
        up = (text or "").upper()
        for code, hints in self.CURRENCY_HINTS.items():
            if any(h.upper() in up for h in hints):
                return code
        return "EUR"

    def _extract_vendor(
        self, text: str, blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        if not lines:
            return None

        inspect_count = max(2, int(len(lines) * 0.3))
        candidates = lines[:inspect_count]

        for idx, line in enumerate(candidates):
            ll = line.lower()
            if len(line) < 4:
                continue
            if any(stop in ll for stop in self.VENDOR_STOPWORDS):
                continue
            if re.fullmatch(r"[\d\s\-./]+", line):
                continue
            if len(re.findall(r"[A-Za-z]", line)) < 3:
                continue

            score = 0.45
            if idx == 0:
                score += 0.2
            if any(k in ll for k in self.VENDOR_KEYWORDS):
                score += 0.2
            if line.isupper():
                score += 0.15

            normalized = self.vendor_matcher.normalize(line) or line
            return {
                "value": normalized.strip(),
                "confidence": round(min(score, 1.0), 2),
                "raw": line,
                "source": "heuristic",
            }

        # Soft fallback: first substantial line.
        for line in candidates:
            ll = line.lower()
            if any(stop in ll for stop in self.VENDOR_STOPWORDS):
                continue
            if len(line) >= 6 and not re.fullmatch(r"[\d\s\-./]+", line):
                normalized = self.vendor_matcher.normalize(line) or line
                return {
                    "value": normalized.strip(),
                    "confidence": 0.4,
                    "raw": line,
                    "source": "fallback",
                }

        return None

    def _maybe_fill_with_llm(self, text: str, fields: Dict[str, Dict[str, Any]]) -> None:
        missing: List[str] = []
        if not fields.get("date") or float(fields["date"].get("confidence", 0.0)) < 0.6:
            missing.append("date")
        if not fields.get("total") or float(fields["total"].get("confidence", 0.0)) < 0.6:
            missing.append("total")
        if not fields.get("vendor") or float(fields["vendor"].get("confidence", 0.0)) < 0.6:
            missing.append("vendor")
        if not missing:
            return

        llm_client = self._get_llm_client()
        if not llm_client or not getattr(llm_client, "enabled", False):
            return
        if not hasattr(llm_client, "smart_extract"):
            return

        try:
            llm_res = llm_client.smart_extract(text, missing)
            if not isinstance(llm_res, dict) or not llm_res.get("success"):
                return
            payload = llm_res.get("analysis")
            if isinstance(payload, str):
                payload = json.loads(payload or "{}")
            if not isinstance(payload, dict):
                return
        except Exception as exc:
            logger.warning("LLM smart_extract failed: %s", exc)
            return

        date_val = payload.get("date")
        if date_val and "date" in missing:
            parsed_date = self._parse_date(str(date_val), "iso", "")
            if not parsed_date:
                parsed_date = self._parse_date(str(date_val), "dmy_or_mdy", "")
            if parsed_date:
                fields["date"] = {
                    "value": parsed_date.strftime("%Y-%m-%d"),
                    "confidence": 0.9,
                    "raw": str(date_val),
                    "source": "llm",
                }

        total_val = payload.get("total")
        if total_val and "total" in missing:
            parsed_amount = self._parse_amount(str(total_val))
            if parsed_amount is not None:
                fields["total"] = {
                    "value": round(parsed_amount, 2),
                    "confidence": 0.9,
                    "raw": str(total_val),
                    "currency": self._detect_currency(str(total_val)),
                    "source": "llm",
                }

        vendor_val = payload.get("vendor") or payload.get("supplier")
        if vendor_val and "vendor" in missing:
            vendor_name = self.vendor_matcher.normalize(str(vendor_val)) or str(vendor_val)
            vendor_field = {
                "value": vendor_name.strip(),
                "confidence": 0.9,
                "raw": str(vendor_val),
                "source": "llm",
            }
            fields["vendor"] = vendor_field
            fields["supplier"] = dict(vendor_field)

    def _attach_validation_summary(self, fields: Dict[str, Dict[str, Any]]) -> None:
        doc_type = str(fields.get("type", {}).get("value", "Invoice") or "Invoice")
        schema_class = get_schema_for_type(doc_type)

        try:
            payload = {
                "doc_type": doc_type,
                "vendor_name": fields.get("vendor", {}).get("value", "Unknown"),
                "total_amount": float(fields.get("total", {}).get("value", 0.0) or 0.0),
                "date": fields.get("date", {}).get("value"),
                "vlm_validated": bool(fields.get("vlm_validated", False)),
            }
            if hasattr(schema_class, "model_fields"):
                model_fields = getattr(schema_class, "model_fields", {})
                if "base_amount" in model_fields:
                    payload["base_amount"] = float(fields.get("base_amount", {}).get("value", 0.0) or 0.0)
                if "vat_amount" in model_fields:
                    payload["vat_amount"] = float(fields.get("vat_amount", {}).get("value", 0.0) or 0.0)

            validated = schema_class(**payload)
            # pydantic v2
            fields["validated_data"] = validated.model_dump() if hasattr(validated, "model_dump") else dict(validated)
            fields["validation_status"] = "valid"
        except Exception as exc:
            fields["validation_status"] = "invalid"
            fields["validation_error"] = str(exc)


__all__ = ["FieldExtractor"]
