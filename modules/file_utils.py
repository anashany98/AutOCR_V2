"""
Utility functions for file handling.

This module provides helpers for scanning directories for valid input files,
computing cryptographic hashes for duplicate detection, moving files to
appropriate output folders and ensuring that required directories exist.  All
operations are designed to be safe and idempotent; directories are created
only if missing and moves will overwrite existing files with the same
basename to avoid orphaned copies.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from typing import Iterable, List, Optional


def ensure_directories(*paths: str) -> None:
    """
    Ensure that each provided directory exists, creating it if necessary.
    """
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)


def list_scannable_files(input_folder: str, file_types: Iterable[str]) -> List[str]:
    """
    Return a list of absolute file paths within ``input_folder`` (including
    subdirectories) that match the given extensions. Hidden files and folders
    (starting with a dot) are ignored.
    """
    allowed_exts = {ft.lower() for ft in file_types}
    file_list: List[str] = []

    for root, dirnames, filenames in os.walk(input_folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() in allowed_exts:
                file_list.append(os.path.join(root, name))
    return file_list


def compute_hash(file_path: str, algorithm: str = "md5", block_size: int = 65536) -> str:
    """
    Compute a cryptographic hash of the file's contents.  Supported
    algorithms include ``md5`` and ``sha256``.  Larger block sizes may
    improve performance on very large files.
    """
    if algorithm.lower() == "md5":
        hasher = hashlib.md5()
    elif algorithm.lower() in ("sha256", "sha-256"):
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def move_file(
    src_path: str,
    dest_folder: str,
    delete_original: bool = False,
    relative_to: Optional[str] = None,
    new_filename: Optional[str] = None,
) -> str:
    """
    Move (or copy) a file to ``dest_folder`` and return the new absolute
    destination path.  If ``delete_original`` is False a copy is made and the
    original remains untouched.  If True the original is removed after the
    move.
    """
    base_folder = dest_folder
    filename = new_filename if new_filename else os.path.basename(src_path)


    if relative_to:
        try:
            rel_path = os.path.relpath(src_path, relative_to)
            rel_dir = os.path.dirname(rel_path)
            base_folder = os.path.join(dest_folder, rel_dir) if rel_dir else dest_folder
        except ValueError:
            base_folder = dest_folder

    ensure_directories(base_folder)
    dest_path = os.path.normpath(os.path.join(base_folder, filename))
    src_path = os.path.normpath(src_path)
    
    if src_path == dest_path:
        return dest_path

    try:
        if delete_original:
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
    except shutil.SameFileError:
        pass
        
    return dest_path
