from __future__ import annotations

import datetime as dt

from modules.auth_manager import AuthManager
from modules.db_manager import DBManager
from web_app.routes import api_routes
from web_app.routes import main_routes


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


def test_public_register_cannot_escalate_role_and_requires_verified_email(tmp_path):
    db = _make_sqlite_db(tmp_path)
    auth = AuthManager(db)

    ok, token = auth.create_user_with_email(
        "alice",
        "alice@example.com",
        "secret123",
        role="personal",
        allow_elevated_role=False,
    )
    assert ok is True
    assert isinstance(token, str) and token

    row = db.execute(
        "SELECT role, is_verified FROM users WHERE username = ?",
        ("alice",),
    ).fetchone()
    role = row[0] if isinstance(row, (tuple, list)) else row["role"]
    verified = row[1] if isinstance(row, (tuple, list)) else row["is_verified"]
    assert role == "CLIENTE"
    assert int(verified or 0) == 0

    assert auth.verify_login("alice", "secret123") is None
    assert auth.last_error == "email_not_verified"

    ok_verify, _ = auth.verify_email(token)
    assert ok_verify is True
    assert auth.verify_login("alice", "secret123") is not None


def test_user_role_and_scope_updates_work_in_sqlite(tmp_path):
    db = _make_sqlite_db(tmp_path)
    auth = AuthManager(db)

    ok, _ = auth.create_user("bob", "pass123")
    assert ok is True
    bob = auth.get_user_by_username("bob")
    assert bob is not None

    ok_role, _ = auth.update_user_role(bob.id, "GESTOR")
    assert ok_role is True

    ok_scope, _ = auth.update_user_hotel_scope(bob.id, [1, 2, 3])
    assert ok_scope is True

    updated = auth.get_user(bob.id)
    assert updated is not None
    assert updated.role == "GESTOR"
    assert updated.hotel_scope == ["1", "2", "3"]


def test_recent_logs_and_create_hotel_do_not_break(tmp_path):
    db = _make_sqlite_db(tmp_path)

    hid = db.create_hotel("Hotel A", "HA", "desc")
    assert isinstance(hid, int) and hid > 0

    row = db.execute(
        "SELECT COUNT(*) FROM hotels WHERE code = ?",
        ("HA",),
    ).fetchone()
    count = row[0] if isinstance(row, (tuple, list)) else row["COUNT(*)"]
    assert int(count) == 1

    db.log_audit(user_id=1, action="test_action", resource_type="test", resource_id="1", details={"ok": True})
    logs = db.get_recent_logs(10)
    assert isinstance(logs, list)
    assert len(logs) >= 1


def test_get_document_path_works_with_legacy_path_column(tmp_path):
    db = _make_sqlite_db(tmp_path)
    doc_id = db.insert_document(
        filename="doc.pdf",
        path=str(tmp_path / "doc.pdf"),
        md5_hash="abc123",
        timestamp=dt.datetime.now(),
        duration=0.5,
        status="OK",
    )

    path = db.get_document_path(doc_id)
    assert path is not None
    assert path.endswith("doc.pdf")


def test_api_documents_schema_prefers_migrated_columns(tmp_path):
    db = _make_sqlite_db(tmp_path)

    db.execute("ALTER TABLE documents ADD COLUMN file_path TEXT", commit=True)
    db.execute("ALTER TABLE documents ADD COLUMN created_at TEXT", commit=True)

    schema = api_routes._documents_schema(db)
    assert schema["path_col"] == "file_path"
    assert schema["created_col"] == "created_at"
    assert schema["type_col"] == "doc_type"


def test_api_find_document_by_path_matches_relative_and_absolute(tmp_path):
    db = _make_sqlite_db(tmp_path)
    rel_only_path = "data/tests/path_match_rel.png"
    abs_from_rel = str((api_routes.PROJECT_ROOT / rel_only_path).resolve())
    rel_from_abs = "data/tests/path_match_abs.png"
    abs_only_path = str((api_routes.PROJECT_ROOT / rel_from_abs).resolve())

    doc_rel_id = db.insert_document(
        filename="rel.png",
        path=rel_only_path,
        md5_hash="rel_hash",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
    )
    doc_abs_id = db.insert_document(
        filename="abs.png",
        path=abs_only_path,
        md5_hash="abs_hash",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
    )

    schema = api_routes._documents_schema(db)
    path_col = schema["path_col"]

    row_from_abs = api_routes._find_document_by_path(
        db,
        path_col,
        abs_from_rel,
        f"id, {path_col} AS path",
    )
    assert row_from_abs is not None
    row_abs_id = row_from_abs[0] if isinstance(row_from_abs, (tuple, list)) else row_from_abs["id"]
    assert int(row_abs_id) == int(doc_rel_id)

    row_from_rel = api_routes._find_document_by_path(
        db,
        path_col,
        rel_from_abs,
        f"id, {path_col} AS path",
    )
    assert row_from_rel is not None
    row_rel_id = row_from_rel[0] if isinstance(row_from_rel, (tuple, list)) else row_from_rel["id"]
    assert int(row_rel_id) == int(doc_abs_id)


def test_main_filter_accessible_doc_ids_enforces_hotel_and_owner_scope(tmp_path):
    db = _make_sqlite_db(tmp_path)

    doc_h1_owner7 = db.insert_document(
        filename="h1_owner7.png",
        path="data/tests/h1_owner7.png",
        md5_hash="h1o7",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
        owner_id=7,
        hotel_id=1,
    )
    doc_h2_owner7 = db.insert_document(
        filename="h2_owner7.png",
        path="data/tests/h2_owner7.png",
        md5_hash="h2o7",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
        owner_id=7,
        hotel_id=2,
    )
    doc_h1_owner9 = db.insert_document(
        filename="h1_owner9.png",
        path="data/tests/h1_owner9.png",
        md5_hash="h1o9",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
        owner_id=9,
        hotel_id=1,
    )

    class _User:
        def __init__(self, uid, role, scope):
            self.id = uid
            self.role = role
            self.hotel_scope = scope

    gestor = _User(uid=100, role="GESTOR", scope=[1])
    allowed_for_gestor = main_routes._filter_accessible_doc_ids(
        db, [doc_h1_owner7, doc_h2_owner7, doc_h1_owner9], user=gestor
    )
    assert allowed_for_gestor == [doc_h1_owner7, doc_h1_owner9]

    cliente = _User(uid=7, role="CLIENTE", scope=[1, 2])
    allowed_for_cliente = main_routes._filter_accessible_doc_ids(
        db, [doc_h1_owner7, doc_h2_owner7, doc_h1_owner9], user=cliente
    )
    assert allowed_for_cliente == [doc_h1_owner7, doc_h2_owner7]


def test_main_find_document_by_path_matches_absolute_relative(tmp_path):
    db = _make_sqlite_db(tmp_path)
    rel_only_path = "data/tests/main_rel.png"
    abs_from_rel = str((main_routes.PROJECT_ROOT / rel_only_path).resolve())
    rel_from_abs = "data/tests/main_abs.png"
    abs_only_path = str((main_routes.PROJECT_ROOT / rel_from_abs).resolve())

    rel_id = db.insert_document(
        filename="main_rel.png",
        path=rel_only_path,
        md5_hash="main_rel",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
    )
    abs_id = db.insert_document(
        filename="main_abs.png",
        path=abs_only_path,
        md5_hash="main_abs",
        timestamp=dt.datetime.now(),
        duration=0.1,
        status="OK",
    )

    schema = main_routes._documents_schema(db)
    path_col = schema["path_col"]

    row_from_abs = main_routes._find_document_by_path(
        db, path_col, abs_from_rel, f"id, {path_col} AS path"
    )
    assert row_from_abs is not None
    row_abs_id = row_from_abs[0] if isinstance(row_from_abs, (tuple, list)) else row_from_abs["id"]
    assert int(row_abs_id) == int(rel_id)

    row_from_rel = main_routes._find_document_by_path(
        db, path_col, rel_from_abs, f"id, {path_col} AS path"
    )
    assert row_from_rel is not None
    row_rel_id = row_from_rel[0] if isinstance(row_from_rel, (tuple, list)) else row_from_rel["id"]
    assert int(row_rel_id) == int(abs_id)
