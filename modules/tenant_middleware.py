"""
Tenant Isolation Middleware & Helpers.

Provides Flask middleware that injects ``g.tenant_context`` on every request
and utility decorators/functions for enforcing multi-tenant access control
at the application layer.

Usage in routes::

    from modules.tenant_middleware import require_tenant, tenant_ctx

    @app.route("/api/docs")
    @require_tenant
    def list_docs():
        ctx = tenant_ctx()
        docs = db.scoped_query("SELECT * FROM documents", ctx)
        ...
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flask import abort, g, request

from modules.feature_flags import flags

logger = logging.getLogger(__name__)


# ============================================================================
# Tenant Context
# ============================================================================

@dataclass
class TenantContext:
    """
    Immutable context object attached to ``g.tenant_context`` per request.

    Encapsulates the current user's access scope and is passed to all
    database helpers for automatic filtering.
    """

    tenant_id: str
    user_id: Optional[str] = None
    role: str = "CLIENTE"
    hotel_ids: List[str] = field(default_factory=list)
    is_admin: bool = False

    # Default tenant for migrated/legacy data
    DEFAULT_TENANT = "00000000-0000-0000-0000-000000000001"

    @property
    def has_full_access(self) -> bool:
        """ADMIN and DIRECCION see all hotels within their tenant."""
        return self.is_admin or self.role in ("ADMIN", "DIRECCION")

    @property
    def hotel_filter(self) -> Optional[List[str]]:
        """
        Returns the list of hotel IDs to filter by, or None for unrestricted.
        """
        if self.has_full_access:
            return None  # No hotel filter — full tenant access
        return self.hotel_ids if self.hotel_ids else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "hotel_ids": self.hotel_ids,
            "is_admin": self.is_admin,
        }


# ============================================================================
# Middleware
# ============================================================================

def init_tenant_middleware(app):
    """
    Register a ``before_request`` hook that builds the TenantContext
    from the authenticated user or falls back to the default tenant.
    """

    @app.before_request
    def _inject_tenant_context():
        if not flags.ENABLE_MULTI_TENANT:
            # Multi-tenant disabled — determine actual admin status from user
            is_admin = False
            try:
                from flask_login import current_user
                if current_user.is_authenticated:
                    is_admin = getattr(current_user, 'role', 'CLIENTE') == 'ADMIN'
            except Exception:
                pass
            
            g.tenant_context = TenantContext(
                tenant_id=TenantContext.DEFAULT_TENANT,
                is_admin=is_admin,
            )
            return

        # Try to build context from Flask-Login current_user
        try:
            from flask_login import current_user

            if current_user.is_authenticated:
                tenant_id = getattr(current_user, "tenant_id", None) or TenantContext.DEFAULT_TENANT
                hotel_scope = getattr(current_user, "hotel_scope", None) or []
                role = getattr(current_user, "role", "CLIENTE")

                g.tenant_context = TenantContext(
                    tenant_id=str(tenant_id),
                    user_id=str(current_user.id),
                    role=role,
                    hotel_ids=[str(h) for h in hotel_scope],
                    is_admin=(role == "ADMIN"),
                )
                return
        except Exception:
            pass

        # Check for API key / header-based tenant
        header_tenant = request.headers.get("X-Tenant-ID")
        if header_tenant:
            g.tenant_context = TenantContext(
                tenant_id=header_tenant,
                role="CLIENTE",
            )
            return

        # Fallback: default tenant (unauthenticated or legacy)
        g.tenant_context = TenantContext(
            tenant_id=TenantContext.DEFAULT_TENANT,
        )


# ============================================================================
# Decorators
# ============================================================================

def require_tenant(f):
    """
    Route decorator that ensures a valid tenant context exists.
    Aborts with 403 if multi-tenant is enabled but no tenant is resolved.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        ctx = getattr(g, "tenant_context", None)
        if ctx is None:
            logger.warning("No tenant context on request %s", request.path)
            abort(403, description="Tenant context required")
        return f(*args, **kwargs)

    return decorated


def require_role(*roles: str):
    """
    Route decorator that checks the user has one of the specified roles.

    Usage::

        @require_role("ADMIN", "DIRECCION")
        def admin_panel():
            ...
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            ctx = getattr(g, "tenant_context", None)
            if ctx is None or ctx.role not in roles:
                abort(403, description="Insufficient permissions")
            return f(*args, **kwargs)

        return decorated

    return decorator


def require_hotel_access(hotel_id_param: str = "hotel_id"):
    """
    Route decorator that verifies the user can access the specified hotel.

    The hotel ID is extracted from the route kwargs or query params.
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            ctx = getattr(g, "tenant_context", None)
            if ctx is None:
                abort(403)

            # Get hotel_id from route params or query string
            hotel_id = kwargs.get(hotel_id_param) or request.args.get(hotel_id_param)
            if not hotel_id:
                return f(*args, **kwargs)  # No hotel specified — let query scoping handle it

            if ctx.has_full_access:
                return f(*args, **kwargs)

            if str(hotel_id) not in [str(h) for h in ctx.hotel_ids]:
                abort(403, description="No access to this hotel")

            return f(*args, **kwargs)

        return decorated

    return decorator


# ============================================================================
# Query Helpers
# ============================================================================

def tenant_ctx() -> TenantContext:
    """Shorthand to get the current request's tenant context."""
    ctx = getattr(g, "tenant_context", None)
    if ctx is None:
        return TenantContext(tenant_id=TenantContext.DEFAULT_TENANT, is_admin=True)
    return ctx


def apply_tenant_filter(
    sql: str,
    params: list,
    ctx: Optional[TenantContext] = None,
    *,
    table_alias: str = "",
    tenant_col: str = "tenant_id",
    hotel_col: str = "hotel_id",
) -> tuple:
    """
    Append tenant + hotel WHERE clauses to a SQL query.

    Parameters
    ----------
    sql:
        The SQL query (must already have a WHERE or will get one appended).
    params:
        Mutable list of query parameters.
    ctx:
        TenantContext (defaults to current request context).
    table_alias:
        Optional table alias prefix (e.g., ``d.``).
    tenant_col / hotel_col:
        Column names for tenant and hotel filtering.

    Returns
    -------
    (modified_sql, params) tuple.
    """
    if ctx is None:
        ctx = tenant_ctx()

    prefix = f"{table_alias}." if table_alias else ""

    # Always filter by tenant
    if " WHERE " in sql.upper():
        sql += f" AND {prefix}{tenant_col} = %s"
    else:
        sql += f" WHERE {prefix}{tenant_col} = %s"
    params.append(ctx.tenant_id)

    # Filter by hotel if user doesn't have full access
    hotel_filter = ctx.hotel_filter
    if hotel_filter is not None:
        if hotel_filter:
            sql += f" AND {prefix}{hotel_col} = ANY(%s)"
            params.append(hotel_filter)
        else:
            # Empty hotel_ids = no access to anything
            sql += " AND FALSE"

    return sql, params


__all__ = [
    "TenantContext",
    "init_tenant_middleware",
    "require_tenant",
    "require_role",
    "require_hotel_access",
    "tenant_ctx",
    "apply_tenant_filter",
]
