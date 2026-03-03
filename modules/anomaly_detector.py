"""
Anomaly detection for extracted invoice/document fields.

This module keeps backward compatibility with the legacy pipeline API while
also exposing tenant-level anomaly checks for dashboards/alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Structured anomaly payload for tenant-level checks."""

    anomaly_type: str
    severity: str
    description: str
    document_id: str
    vendor_name: str
    amount: float
    currency: str
    details: Dict[str, Any]


class AnomalyDetector:
    """
    Detect anomalies in extracted document data.

    Backward-compatible API used by the processor:
    - ``AnomalyDetector(config)``
    - ``detect(fields, document_history=None) -> List[str]``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._db = None

    # ------------------------------------------------------------------ #
    # Legacy API (used by postbatch_processor)
    # ------------------------------------------------------------------ #

    def detect(self, fields: Dict[str, Dict[str, Any]], document_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Legacy anomaly detection contract for extracted fields.

        Returns simple anomaly codes for compatibility with existing callers.
        """
        anomalies: List[str] = []

        if not fields.get("date"):
            anomalies.append("missing_date")
        if not fields.get("total"):
            anomalies.append("missing_total")
        if not fields.get("vendor"):
            anomalies.append("missing_vendor")

        if fields.get("date"):
            anomalies.extend(self._detect_date_anomalies(fields["date"]))

        if fields.get("total") and document_history:
            anomalies.extend(self._detect_amount_anomalies(fields["total"], document_history))

        return anomalies

    def _detect_date_anomalies(self, date_field: Dict[str, Any]) -> List[str]:
        anomalies: List[str] = []
        try:
            date_value = datetime.strptime(str(date_field.get("value", "")), "%Y-%m-%d")
            today = datetime.now()

            if date_value > today + timedelta(days=7):
                anomalies.append("future_date")

            if date_value.weekday() >= 5:
                anomalies.append("weekend_date")

            if date_value < today - timedelta(days=365 * 5):
                anomalies.append("very_old_date")
        except ValueError:
            anomalies.append("invalid_date_format")
        return anomalies

    def _detect_amount_anomalies(self, total_field: Dict[str, Any], document_history: List[Dict[str, Any]]) -> List[str]:
        anomalies: List[str] = []
        try:
            current_amount = float(total_field.get("value", 0))

            historical_amounts: List[float] = []
            for doc in document_history:
                try:
                    historical_amounts.append(float(doc.get("fields", {}).get("total", {}).get("value")))
                except Exception:
                    continue

            if len(historical_amounts) < 5:
                return anomalies

            avg = mean(historical_amounts)
            std = stdev(historical_amounts) if len(historical_amounts) > 1 else 0.0

            if std > 0 and abs(current_amount - avg) > 3 * std:
                anomalies.append("unusual_amount")

            if avg > 0 and current_amount > avg * 10:
                anomalies.append("very_large_amount")

            if current_amount == round(current_amount, -2) and current_amount >= 1000:
                anomalies.append("suspicious_round_number")
        except (ValueError, TypeError, ZeroDivisionError):
            anomalies.append("invalid_amount")
        return anomalies

    # ------------------------------------------------------------------ #
    # Extended API (tenant-level checks)
    # ------------------------------------------------------------------ #

    def _get_db(self):
        """Best-effort DB manager acquisition for optional advanced checks."""
        if self._db is not None:
            return self._db
        try:
            from modules.db_manager import DBManager

            if hasattr(DBManager, "get_instance"):
                self._db = DBManager.get_instance(self.config)
            else:
                self._db = DBManager(self.config)
        except Exception as exc:
            logger.error("Unable to initialize DB manager for anomaly checks: %s", exc)
            self._db = None
        return self._db

    @staticmethod
    def _fetch_all(db, query: str, params: tuple):
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            cursor.execute(query, params)
            return cursor.fetchall() or []

    @staticmethod
    def _fetch_one(db, query: str, params: tuple):
        with db.get_connection() as conn:
            cursor = db.get_cursor(conn)
            cursor.execute(query, params)
            return cursor.fetchone()

    def check_all_anomalies(self, tenant_id: str) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        anomalies.extend(self.check_amount_outliers(tenant_id))
        anomalies.extend(self.check_new_vendors(tenant_id))
        anomalies.extend(self.check_duplicate_amounts(tenant_id))
        return anomalies

    def check_amount_outliers(self, tenant_id: str) -> List[Anomaly]:
        db = self._get_db()
        if db is None:
            return []

        ph = getattr(db, "placeholder", "%s")
        query = f"""
            SELECT
                d.id,
                d.vendor_name,
                d.total_amount,
                d.currency,
                v.avg_amount,
                v.std_amount
            FROM documents d
            LEFT JOIN vendor_statistics v ON v.vendor_name = d.vendor_name
            WHERE d.tenant_id = {ph}
              AND d.total_amount IS NOT NULL
              AND d.status = 'completed'
        """

        anomalies: List[Anomaly] = []
        try:
            results = self._fetch_all(db, query, (tenant_id,))
            for row in results:
                doc_id = str(row[0] if isinstance(row, (tuple, list)) else row["id"])
                vendor_name = row[1] if isinstance(row, (tuple, list)) else row.get("vendor_name")
                amount = float((row[2] if isinstance(row, (tuple, list)) else row.get("total_amount")) or 0)
                currency = (row[3] if isinstance(row, (tuple, list)) else row.get("currency")) or "EUR"
                avg_amount = float((row[4] if isinstance(row, (tuple, list)) else row.get("avg_amount")) or 0)
                std_amount = float((row[5] if isinstance(row, (tuple, list)) else row.get("std_amount")) or 0)
                if avg_amount <= 0 or std_amount <= 0:
                    continue

                z_score = abs(amount - avg_amount) / std_amount
                if z_score <= 2:
                    continue

                anomalies.append(
                    Anomaly(
                        anomaly_type="amount_outlier",
                        severity="high" if z_score > 3 else "medium",
                        description=f"Importe fuera de rango: {amount:.2f} {currency}",
                        document_id=doc_id,
                        vendor_name=vendor_name or "Sin proveedor",
                        amount=amount,
                        currency=currency,
                        details={"avg_amount": avg_amount, "std_amount": std_amount, "z_score": z_score},
                    )
                )
        except Exception as exc:
            logger.error("Error checking amount outliers: %s", exc)
        return anomalies

    def check_new_vendors(self, tenant_id: str) -> List[Anomaly]:
        db = self._get_db()
        if db is None:
            return []

        ph = getattr(db, "placeholder", "%s")
        query = f"""
            SELECT d.id, d.vendor_name, d.total_amount, d.currency, v.id, v.is_verified
            FROM documents d
            LEFT JOIN vendors v ON v.name = d.vendor_name
            WHERE d.tenant_id = {ph}
              AND d.vendor_name IS NOT NULL
        """
        count_query = f"SELECT COUNT(*) FROM documents WHERE tenant_id = {ph} AND vendor_name = {ph}"

        anomalies: List[Anomaly] = []
        try:
            for row in self._fetch_all(db, query, (tenant_id,)):
                doc_id = str(row[0] if isinstance(row, (tuple, list)) else row["id"])
                vendor_name = row[1] if isinstance(row, (tuple, list)) else row.get("vendor_name")
                amount = float((row[2] if isinstance(row, (tuple, list)) else row.get("total_amount")) or 0)
                currency = (row[3] if isinstance(row, (tuple, list)) else row.get("currency")) or "EUR"
                vendor_id = row[4] if isinstance(row, (tuple, list)) else row.get("id")
                is_verified = row[5] if isinstance(row, (tuple, list)) else row.get("is_verified")
                if vendor_id and is_verified:
                    continue
                if not vendor_name:
                    continue

                count_row = self._fetch_one(db, count_query, (tenant_id, vendor_name))
                vendor_count = int((count_row[0] if isinstance(count_row, (tuple, list)) else count_row.get("count", 0)) or 0)
                if vendor_count > 2:
                    continue

                anomalies.append(
                    Anomaly(
                        anomaly_type="new_vendor",
                        severity="medium",
                        description=f"Proveedor nuevo no homologado: {vendor_name}",
                        document_id=doc_id,
                        vendor_name=vendor_name,
                        amount=amount,
                        currency=currency,
                        details={"vendor_count": vendor_count, "vendor_id": vendor_id, "is_verified": bool(is_verified)},
                    )
                )
        except Exception as exc:
            logger.error("Error checking new vendors: %s", exc)
        return anomalies

    def check_duplicate_amounts(self, tenant_id: str) -> List[Anomaly]:
        db = self._get_db()
        if db is None:
            return []

        ph = getattr(db, "placeholder", "%s")
        query = f"""
            SELECT
                d1.id,
                d1.vendor_name,
                d1.total_amount,
                d1.currency,
                d1.created_at,
                d2.id,
                d2.created_at
            FROM documents d1
            JOIN documents d2 ON
                d1.tenant_id = d2.tenant_id AND
                d1.vendor_name = d2.vendor_name AND
                d1.total_amount = d2.total_amount AND
                d1.id != d2.id
            WHERE d1.tenant_id = {ph}
              AND d1.total_amount IS NOT NULL
              AND d1.status = 'completed'
        """

        anomalies: List[Anomaly] = []
        reported_pairs = set()
        try:
            for row in self._fetch_all(db, query, (tenant_id,)):
                doc1_id = str(row[0] if isinstance(row, (tuple, list)) else row["id"])
                vendor_name = (row[1] if isinstance(row, (tuple, list)) else row.get("vendor_name")) or "Sin proveedor"
                amount = float((row[2] if isinstance(row, (tuple, list)) else row.get("total_amount")) or 0)
                currency = (row[3] if isinstance(row, (tuple, list)) else row.get("currency")) or "EUR"
                doc1_date = row[4] if isinstance(row, (tuple, list)) else row.get("created_at")
                doc2_id = str(row[5] if isinstance(row, (tuple, list)) else row.get("id_1"))
                doc2_date = row[6] if isinstance(row, (tuple, list)) else row.get("created_at_1")

                pair_key = tuple(sorted([doc1_id, doc2_id]))
                if pair_key in reported_pairs:
                    continue
                reported_pairs.add(pair_key)

                days_apart = 0
                if isinstance(doc1_date, datetime) and isinstance(doc2_date, datetime):
                    days_apart = abs((doc1_date - doc2_date).days)
                if days_apart > 7:
                    continue

                anomalies.append(
                    Anomaly(
                        anomaly_type="duplicate_amount",
                        severity="high",
                        description=f"Posible duplicado: {amount:.2f} {currency}",
                        document_id=doc1_id,
                        vendor_name=vendor_name,
                        amount=amount,
                        currency=currency,
                        details={"duplicate_doc_id": doc2_id, "days_apart": days_apart},
                    )
                )
        except Exception as exc:
            logger.error("Error checking duplicate amounts: %s", exc)
        return anomalies


_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector


__all__ = ["Anomaly", "AnomalyDetector", "get_anomaly_detector"]
