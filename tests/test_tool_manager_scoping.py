"""Security regression tests for ToolManager multi-tenant scoping."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from modules.db_manager import DBManager
from modules.tool_manager import ToolManager


def _make_sqlite_db(tmp_path) -> DBManager:
    db_path = tmp_path / "test.db"
    config = {
        "database": {
            "engine": "sqlite",
            "pool_size": 2,
            "sqlite": {"path": str(db_path)},
        }
    }
    return DBManager(config)


def _read_export_ids(csv_path: Path) -> list[int]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(row["id"]) for row in reader if row.get("id")]


def test_tool_export_respects_hotel_scope(tmp_path):
    db = _make_sqlite_db(tmp_path)
    tm = ToolManager(db, str(tmp_path), allowed_doc_roots=[str(tmp_path)])

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    t0 = dt.datetime.now()
    doc1 = db.insert_document(
        filename="a.pdf",
        path=str(docs_dir / "a.pdf"),
        md5_hash="h1",
        timestamp=t0,
        duration=0.0,
        status="done",
        owner_id=10,
        hotel_id=1,
    )
    doc2 = db.insert_document(
        filename="b.pdf",
        path=str(docs_dir / "b.pdf"),
        md5_hash="h2",
        timestamp=t0,
        duration=0.0,
        status="done",
        owner_id=11,
        hotel_id=2,
    )

    user_ctx = {"role": "GESTOR", "hotel_scope": [1], "user_id": "99"}
    msg = tm.execute_tool("export_search_results_to_csv", {}, user_context=user_ctx)
    marker = "/data/exports/"
    assert marker in msg
    rel = msg.split(marker, 1)[1].strip()
    export_path = (tmp_path / "data" / "exports" / rel).resolve()
    assert export_path.exists(), f"export file not found at {export_path}"

    ids = _read_export_ids(export_path)
    assert doc1 in ids
    assert doc2 not in ids


def test_tool_update_document_type_scoped(tmp_path):
    db = _make_sqlite_db(tmp_path)
    tm = ToolManager(db, str(tmp_path), allowed_doc_roots=[str(tmp_path)])

    t0 = dt.datetime.now()
    doc1 = db.insert_document(
        filename="a.pdf",
        path=str(tmp_path / "a.pdf"),
        md5_hash="h1",
        timestamp=t0,
        duration=0.0,
        status="done",
        owner_id=10,
        hotel_id=1,
    )
    doc2 = db.insert_document(
        filename="b.pdf",
        path=str(tmp_path / "b.pdf"),
        md5_hash="h2",
        timestamp=t0,
        duration=0.0,
        status="done",
        owner_id=11,
        hotel_id=2,
    )

    user_ctx = {"role": "GESTOR", "hotel_scope": [1], "user_id": "99"}

    ok = tm.execute_tool(
        "update_document_type",
        {"doc_id": doc1, "new_type": "Invoice"},
        user_context=user_ctx,
    )
    assert "actualizado" in ok.lower()

    denied = tm.execute_tool(
        "update_document_type",
        {"doc_id": doc2, "new_type": "Invoice"},
        user_context=user_ctx,
    )
    assert "denegado" in denied.lower()

    # Verify doc2 not modified.
    row = db.execute("SELECT type FROM documents WHERE id = ?", (doc2,)).fetchone()
    doc2_type = row[0] if isinstance(row, (tuple, list)) else row["type"]
    assert doc2_type is None


def test_client_cannot_execute_document_mutation_tools(tmp_path):
    db = _make_sqlite_db(tmp_path)
    tm = ToolManager(db, str(tmp_path), allowed_doc_roots=[str(tmp_path)])

    t0 = dt.datetime.now()
    doc1 = db.insert_document(
        filename="a.pdf",
        path=str(tmp_path / "a.pdf"),
        md5_hash="h1",
        timestamp=t0,
        duration=0.0,
        status="done",
        owner_id=10,
        hotel_id=1,
    )

    user_ctx = {"role": "CLIENTE", "hotel_scope": [1], "user_id": "10"}
    denied = tm.execute_tool(
        "update_document_type",
        {"doc_id": doc1, "new_type": "Invoice"},
        user_context=user_ctx,
    )
    assert "permisos" in denied.lower() or "denegado" in denied.lower()

    row = db.execute("SELECT type FROM documents WHERE id = ?", (doc1,)).fetchone()
    doc_type = row[0] if isinstance(row, (tuple, list)) else row["type"]
    assert doc_type is None
