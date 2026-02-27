"""
Torch CUDA compatibility helpers.

These helpers avoid selecting CUDA when the installed PyTorch build does not
support the current GPU architecture (for example, RTX 50-series with older
CUDA builds), which would otherwise fail at runtime with "no kernel image".
"""

from __future__ import annotations

from typing import Optional, Tuple


def _safe_arch_list(torch_mod) -> set[str]:
    try:
        archs = torch_mod.cuda.get_arch_list()
        return {str(a) for a in (archs or [])}
    except Exception:
        return set()


def _safe_device_arch(torch_mod) -> Optional[str]:
    try:
        major, minor = torch_mod.cuda.get_device_capability(0)
    except Exception:
        return None
    return f"sm_{major}{minor}"


def torch_cuda_usable(
    torch_mod,
    *,
    smoke_test: bool = False,
) -> Tuple[bool, str]:
    """
    Return (usable, reason) for CUDA execution with the provided torch module.
    """
    if torch_mod is None:
        return False, "torch module not available"

    try:
        if not torch_mod.cuda.is_available():
            return False, "CUDA not available"
    except Exception as exc:
        return False, f"CUDA check failed: {exc}"

    device_arch = _safe_device_arch(torch_mod)
    arch_list = _safe_arch_list(torch_mod)
    if device_arch and arch_list and device_arch not in arch_list:
        return (
            False,
            f"GPU architecture {device_arch} is not in this torch build ({', '.join(sorted(arch_list))})",
        )

    if smoke_test:
        try:
            x = torch_mod.tensor([1.0], device="cuda")
            _ = (x + 1.0).item()
        except Exception as exc:
            return False, f"CUDA runtime test failed: {exc}"

    return True, "ok"


__all__ = ["torch_cuda_usable"]
