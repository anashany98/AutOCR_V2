from __future__ import annotations

from collections import namedtuple

import postbatch_processor as pb


def _gib(value: int) -> int:
    return int(value * pb.BYTES_PER_GIB)


def test_plan_processing_batch_as_found_respects_size_limit(monkeypatch):
    files = ["a.pdf", "b.pdf", "c.pdf"]
    sizes = {
        "a.pdf": _gib(6),
        "b.pdf": _gib(3),
        "c.pdf": _gib(3),
    }
    monkeypatch.setattr(pb, "_safe_file_size", lambda path: sizes[path])

    selected, deferred, selected_bytes, discovered_bytes = pb.plan_processing_batch(
        files,
        max_input_gb_per_run=10,
        processing_order="as_found",
    )

    assert selected == ["a.pdf", "b.pdf"]
    assert deferred == ["c.pdf"]
    assert selected_bytes == _gib(9)
    assert discovered_bytes == _gib(12)


def test_plan_processing_batch_selects_at_least_one_file(monkeypatch):
    files = ["huge.pdf", "small.pdf"]
    sizes = {
        "huge.pdf": _gib(12),
        "small.pdf": _gib(1),
    }
    monkeypatch.setattr(pb, "_safe_file_size", lambda path: sizes[path])

    selected, deferred, selected_bytes, discovered_bytes = pb.plan_processing_batch(
        files,
        max_input_gb_per_run=10,
        processing_order="as_found",
    )

    assert selected == ["huge.pdf"]
    assert deferred == ["small.pdf"]
    assert selected_bytes == _gib(12)
    assert discovered_bytes == _gib(13)


def test_plan_processing_batch_small_first_prioritizes_throughput(monkeypatch):
    files = ["a.pdf", "b.pdf", "c.pdf"]
    sizes = {
        "a.pdf": _gib(9),
        "b.pdf": _gib(2),
        "c.pdf": _gib(2),
    }
    monkeypatch.setattr(pb, "_safe_file_size", lambda path: sizes[path])

    selected, deferred, selected_bytes, discovered_bytes = pb.plan_processing_batch(
        files,
        max_input_gb_per_run=10,
        processing_order="small_first",
    )

    assert selected == ["b.pdf", "c.pdf"]
    assert deferred == ["a.pdf"]
    assert selected_bytes == _gib(4)
    assert discovered_bytes == _gib(13)


def test_plan_processing_batch_respects_max_files_per_run(monkeypatch):
    files = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    sizes = {
        "a.pdf": _gib(1),
        "b.pdf": _gib(1),
        "c.pdf": _gib(1),
        "d.pdf": _gib(1),
    }
    monkeypatch.setattr(pb, "_safe_file_size", lambda path: sizes[path])

    selected, deferred, selected_bytes, discovered_bytes = pb.plan_processing_batch(
        files,
        max_input_gb_per_run=10,
        max_files_per_run=2,
        processing_order="as_found",
    )

    assert selected == ["a.pdf", "b.pdf"]
    assert deferred == ["c.pdf", "d.pdf"]
    assert selected_bytes == _gib(2)
    assert discovered_bytes == _gib(4)


def test_resolve_max_input_bytes_per_run_uses_min_of_fixed_and_ratio(monkeypatch):
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        pb.shutil,
        "disk_usage",
        lambda _path: usage(total=_gib(300), used=_gib(200), free=_gib(100)),
    )

    effective, info = pb.resolve_max_input_bytes_per_run(
        "ignored",
        max_input_gb_per_run=60,
        max_input_free_disk_ratio=0.5,  # 50 GiB from 100 GiB free
    )
    assert effective == _gib(50)
    assert info["fixed_bytes"] == _gib(60)
    assert info["ratio_bytes"] == _gib(50)


def test_check_disk_headroom_considers_min_free_and_required(monkeypatch):
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        pb.shutil,
        "disk_usage",
        lambda _path: usage(total=_gib(100), used=_gib(80), free=_gib(20)),
    )

    ok, details = pb.check_disk_headroom(
        "ignored",
        min_free_gb=10,
        required_bytes=_gib(15),
    )
    assert ok is True
    assert details["free_bytes"] == _gib(20)
    assert details["required_min_bytes"] == _gib(15)

    ok, details = pb.check_disk_headroom(
        "ignored",
        min_free_gb=10,
        required_bytes=_gib(25),
    )
    assert ok is False
    assert details["required_min_bytes"] == _gib(25)
