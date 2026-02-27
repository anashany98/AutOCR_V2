"""
Configuration normalisation helpers.

AutoOCR currently has a few configuration keys that historically lived under
different parents (e.g. `postbatch.hot_folder` vs top-level `hot_folder`).
This module maps those legacy locations into the canonical ones expected by
the runtime.

Notes
- This does not invent new defaults; it only lifts already-present values.
- The function mutates the provided dict in-place and returns it.
"""

from __future__ import annotations

from typing import Any, Dict


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def normalize_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if not _is_dict(config):
        return {}

    postbatch = config.get("postbatch", {})
    if not _is_dict(postbatch):
        postbatch = {}

    # Lift legacy nested sections from `postbatch.*` to top-level.
    for key in ("hot_folder", "email_importer"):
        if key not in config and _is_dict(postbatch.get(key)):
            config[key] = postbatch.get(key)

    # Vision: some configs placed CLIP/FAISS settings under `llm.vision`.
    # Only lift it if it looks like a VisionManager config (not an LLM routing profile).
    if "vision" not in config:
        llm = config.get("llm", {})
        llm_vision = llm.get("vision") if _is_dict(llm) else None
        if _is_dict(llm_vision):
            vision_like_keys = {
                "index_path",
                "embeddings_dir",
                "gallery_dir",
                "model",
                "model_name",
                "pretrained",
                "auto_tagging",
            }
            if any(key in llm_vision for key in vision_like_keys):
                config["vision"] = llm_vision

    # OCR output: sometimes mis-nested under `llm.output`.
    ocr_pipeline = config.get("ocr_pipeline", {})
    if _is_dict(ocr_pipeline) and "output" not in ocr_pipeline:
        llm = config.get("llm", {})
        llm_output = llm.get("output") if _is_dict(llm) else None
        if _is_dict(llm_output) and (
            "formats" in llm_output or "save_markdown_in_db" in llm_output
        ):
            ocr_pipeline["output"] = llm_output

    return config


__all__ = ["normalize_config"]

