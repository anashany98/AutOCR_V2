"""
Audit Logger — Structured audit trail for compliance & debugging.

Logs user actions, pipeline events, and admin operations to both the
``audit_logs`` database table and the application logger.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import g, request

logger = logging.getLogger("audit")


class AuditLogger:
    """
    Structured audit logging to database + log files.

    Parameters
    ----------
    db:
        Database manager instance.
    """

    # Pre-defined action categories
    ACTIONS = {
        # Documents
        "doc.upload": "Document uploaded",
        "doc.delete": "Document deleted",
        "doc.view": "Document viewed",
        "doc.download": "Document downloaded",
        "doc.process": "Document processing started",
        "doc.reprocess": "Document reprocessing triggered",
        # Chat / RAG
        "chat.query": "Chat query submitted",
        "chat.session_create": "Chat session created",
        # Admin
        "admin.user_create": "User account created",
        "admin.user_role_change": "User role changed",
        "admin.user_scope_change": "User hotel scope changed",
        "admin.user_delete": "User account deleted",
        "admin.migration_run": "Database migration executed",
        # System
        "system.login": "User logged in",
        "system.logout": "User logged out",
        "system.login_failed": "Login attempt failed",
    }

    def __init__(self, db: Any):
        self.db = db

    def log(
        self,
        action: str,
        *,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """
        Record an audit event.

        Parameters
        ----------
        action:
            Action code (e.g., ``doc.upload``, ``admin.user_create``).
        resource_type / resource_id:
            The resource being acted upon.
        details:
            Additional context (stored as JSONB).
        """
        # Resolve request context if available
        if tenant_id is None:
            ctx = getattr(g, "tenant_context", None)
            if ctx:
                tenant_id = ctx.tenant_id
                user_id = user_id or ctx.user_id

        if ip_address is None:
            try:
                ip_address = request.remote_addr
            except RuntimeError:
                ip_address = None

        # Log to app logger
        desc = self.ACTIONS.get(action, action)
        logger.info(
            "%s | user=%s tenant=%s resource=%s:%s",
            desc,
            user_id or "system",
            (tenant_id or "?")[:8],
            resource_type or "-",
            (resource_id or "-")[:8] if resource_id else "-",
        )

        # Write to database
        self._store(
            action=action,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
        )

    def _store(
        self,
        action: str,
        tenant_id: Optional[str],
        user_id: Optional[str],
        resource_type: Optional[str],
        resource_id: Optional[str],
        details: Optional[Dict[str, Any]],
        ip_address: Optional[str],
    ) -> None:
        """Persist audit entry to database."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO audit_logs (
                        tenant_id, user_id, action,
                        resource_type, resource_id,
                        details, ip_address
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tenant_id,
                        user_id,
                        action,
                        resource_type,
                        resource_id,
                        json.dumps(details) if details else None,
                        ip_address,
                    ),
                )
                conn.commit()
        except Exception as e:
            # Never let audit logging break the main flow
            logger.warning("Failed to store audit log: %s", e)

    def query_logs(
        self,
        tenant_id: str,
        *,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Query audit logs for a tenant."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                sql = """
                    SELECT id, tenant_id, user_id, action,
                           resource_type, resource_id, details,
                           ip_address, created_at
                    FROM audit_logs
                    WHERE tenant_id = %s
                """
                params = [tenant_id]

                if action:
                    sql += " AND action = %s"
                    params.append(action)
                if user_id:
                    sql += " AND user_id = %s"
                    params.append(user_id)

                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                return [
                    {
                        "id": str(r[0]),
                        "tenant_id": str(r[1]) if r[1] else None,
                        "user_id": str(r[2]) if r[2] else None,
                        "action": r[3],
                        "resource_type": r[4],
                        "resource_id": str(r[5]) if r[5] else None,
                        "details": json.loads(r[6]) if r[6] else None,
                        "ip_address": r[7],
                        "created_at": str(r[8]),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error("Failed to query audit logs: %s", e)
            return []


__all__ = ["AuditLogger"]
