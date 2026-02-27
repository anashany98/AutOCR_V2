"""
Ephemeral data retention utilities.

AutoOCR produces user-facing *artifacts* (exports, Vision Studio intermediate files)
that should not grow without bounds in a 24/7 deployment. This module provides a
safe, best-effort TTL purge for those ephemeral directories.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PurgeStats:
    base_dir: str
    max_age_days: float
    dry_run: bool = False
    scanned_files: int = 0
    deleted_files: int = 0
    deleted_dirs: int = 0
    bytes_freed: int = 0
    errors: int = 0


def purge_directory(
    base_dir: str,
    *,
    max_age_days: float,
    logger: Optional[object] = None,
    dry_run: bool = False,
    max_deletions: int = 50_000,
) -> PurgeStats:
    """
    Delete files older than ``max_age_days`` under ``base_dir``.

    Notes
    - Fails closed: only touches paths under ``base_dir``.
    - Best-effort: errors are counted and logged but do not abort the purge.
    """

    stats = PurgeStats(base_dir=str(base_dir), max_age_days=float(max_age_days), dry_run=bool(dry_run))
    if max_age_days <= 0:
        return stats

    try:
        root = Path(base_dir).expanduser().resolve()
    except Exception:
        # Invalid path string.
        stats.errors += 1
        return stats

    if not root.exists() or not root.is_dir():
        return stats

    cutoff = time.time() - (max_age_days * 86400.0)
    deletions_left = int(max_deletions)

    # Walk bottom-up so we can remove empty directories after deleting files.
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False, followlinks=False):
        if deletions_left <= 0:
            break

        # Defensive: never operate outside the root directory.
        try:
            current_dir = Path(dirpath).resolve()
            if os.path.commonpath([str(current_dir), str(root)]) != str(root):
                continue
        except Exception:
            stats.errors += 1
            continue

        for name in filenames:
            if deletions_left <= 0:
                break
            if name in {".gitkeep"}:
                continue

            path = current_dir / name
            try:
                st = path.stat()
            except OSError:
                stats.errors += 1
                continue

            stats.scanned_files += 1
            mtime = float(getattr(st, "st_mtime", 0.0) or 0.0)
            if mtime >= cutoff:
                continue

            size = int(getattr(st, "st_size", 0) or 0)
            if logger:
                try:
                    logger.info("Retention: deleting %s (age>%.1fd)", str(path), max_age_days)
                except Exception:
                    pass

            if not dry_run:
                try:
                    path.unlink(missing_ok=True)  # py311+
                except TypeError:
                    # Python <3.8 compat (not expected here).
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        stats.errors += 1
                        continue
                except Exception:
                    stats.errors += 1
                    continue

            stats.deleted_files += 1
            stats.bytes_freed += size
            deletions_left -= 1

        # Try removing empty directories (but never delete the base itself).
        if current_dir == root:
            continue
        try:
            if not os.listdir(str(current_dir)):
                if logger:
                    try:
                        logger.info("Retention: removing empty dir %s", str(current_dir))
                    except Exception:
                        pass
                if not dry_run:
                    os.rmdir(str(current_dir))
                stats.deleted_dirs += 1
        except Exception:
            # Ignore non-empty / permission issues.
            continue

    return stats


__all__ = ["PurgeStats", "purge_directory"]

