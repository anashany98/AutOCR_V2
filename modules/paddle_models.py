"""Utility helpers for managing PaddleOCR model assets.

This module centralises download/extraction of PP-OCRv4 inference models so
runtime components can request model directories without embedding brittle
scripts elsewhere in the codebase.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict

from loguru import logger


# URLs published by PaddleOCR for PP-OCRv4 multilingual (Latin) models compatible with Paddle 3.0.0
PP_OCRV4_SPECS = {
    "latin": {
        "det": {
            "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar",
            "folder": "PP-OCRv4_mobile_det_infer",
        },
        "rec": {
            "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv3_mobile_rec_infer.tar",
            "folder": "latin_PP-OCRv3_mobile_rec_infer",
        },
        "cls": {
            "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar",
            "folder": "PP-LCNet_x0_25_textline_ori_infer",
        },
    },
}


def ensure_ppocrv4_models(base_dir: str, profile: str = "latin") -> Dict[str, str]:
    """Download (if necessary) and return PP-OCRv4 model directories.

    Parameters
    ----------
    base_dir:
        Directory where Paddle models should be stored.
    profile:
        Model profile key (currently only ``latin`` is defined).

    Returns
    -------
    dict
        Mapping compatible with PaddleOCR constructor: ``det_model_dir``,
        ``rec_model_dir`` and ``cls_model_dir``.
    """

    profile_key = profile.lower()
    if profile_key not in PP_OCRV4_SPECS:
        raise ValueError(f"Unsupported PP-OCRv4 profile '{profile}'")

    target_root = Path(base_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    variant_root = target_root / "ppocrv4" / profile_key
    variant_root.mkdir(parents=True, exist_ok=True)

    specs = PP_OCRV4_SPECS[profile_key]
    model_dirs: Dict[str, str] = {}

    for key, spec in specs.items():
        folder = variant_root / spec["folder"]
        if not _model_ready(folder):
            _download_and_extract(spec["url"], variant_root)
        model_dirs[f"{key}_model_dir"] = str(folder)

    return model_dirs


def _model_ready(path: Path) -> bool:
    if not path.exists():
        return False
    for marker in ("inference.pdmodel", "inference.pdiparams"):
        if not (path / marker).exists():
            return False
    return True


def _download_and_extract(url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("⬇️ Downloading Paddle model %s", url)
    # On Windows, we must close the file handle before another process/helper tries to open it by path
    tmp_file = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close() 
    
    try:
        _download_file(url, tmp_path)
        _extract_tar(tmp_path, target_dir)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r") as tar:
        for member in tar.getmembers():
            member_path = destination / member.name
            if not _is_within_directory(destination, member_path):
                raise RuntimeError(f"Blocked unsafe archive entry: {member.name}")
        tar.extractall(destination)


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        target.relative_to(directory)
        return True
    except ValueError:
        return False


__all__ = ["ensure_ppocrv4_models"]
