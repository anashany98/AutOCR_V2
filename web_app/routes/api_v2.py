"""
Legacy compatibility shim for `web_app.routes.api_v2`.

Current API routes are implemented in `api_routes.py` and `chat_v2_routes.py`.
This module keeps old import/patch paths valid.
"""

from typing import Any, Dict


def process_document(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    """Compatibility stub used by legacy tests that monkeypatch this symbol."""
    return {"status": "not_implemented"}


__all__ = ["process_document"]
