"""
Legacy compatibility shim for `web_app.routes.auth`.

New code lives in `web_app.routes.auth_routes`, but older tests/modules still
patch symbols in this module path.
"""

from modules.auth_manager import AuthManager
from web_app.services import get_db


def authenticate_user(username: str, password: str):
    """Backward-compatible authentication helper."""
    auth = AuthManager(get_db())
    return auth.verify_login(username, password)


__all__ = ["authenticate_user"]
