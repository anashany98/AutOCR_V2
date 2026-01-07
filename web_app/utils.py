import base64
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Helper to find project root if needed, but usually passed or inferred
# Assuming this file is in web_app/utils.py, root is ../..
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_path(path_value: Optional[str], fallback: Optional[str] = None) -> str:
    value = path_value or fallback
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)

def ensure_within_project(path: Path) -> Path:
    """
    Ensure the path is relative to the project root.
    If absolute and inside root, return relative.
    If absolute and outside root (e.g. temp), return as is (for now) or handle accordingly.
    """
    # Try different implementations to match original behavior or improved one
    try:
        if path.is_absolute():
            return path.relative_to(PROJECT_ROOT)
        return path
    except ValueError:
        return path

def encode_path(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("utf-8")


def decode_path(token: str) -> str:
    return base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")


def safe_json_parse(value: Any, default: Any = None) -> Any:
    """Parse JSON safely with type checking and fallback."""
    if not value or not isinstance(value, str):
        return default if default is not None else []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []
