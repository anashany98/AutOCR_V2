"""Security regression tests for chat history and RAG scoping helpers.

These tests avoid spinning up the full Flask app. They validate the two core
building blocks that prevent cross-tenant leaks:
- chat_history must be retrievable only for the requesting user_id
- RAG search filtering must fail closed when metadata is missing
"""

from __future__ import annotations

import datetime
import numpy as np

from modules.db_manager import DBManager
from modules.rag_manager import RAGManager
from web_app.routes import chat_routes


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


def test_chat_history_scoped_by_user_id(tmp_path):
    db = _make_sqlite_db(tmp_path)
    session_id = "same-session"

    # User 1 messages
    db.insert_chat_message(session_id, "user", "u1-m1", user_id="1")
    db.insert_chat_message(session_id, "assistant", "u1-m2", user_id="1")
    db.insert_chat_message(session_id, "assistant", "u1-m3", user_id="1")

    # User 2 message in the same session_id (must never show for user 1)
    db.insert_chat_message(session_id, "user", "u2-m1", user_id="2")

    history_u1 = db.get_chat_history(session_id, limit=2, user_id="1")
    assert [m["content"] for m in history_u1] == ["u1-m2", "u1-m3"]

    history_u2 = db.get_chat_history(session_id, limit=50, user_id="2")
    assert [m["content"] for m in history_u2] == ["u2-m1"]


def test_chat_task_registry_enforces_owner_scope(tmp_path):
    db = _make_sqlite_db(tmp_path)

    ok = db.register_chat_task(
        task_id="task-123",
        user_id="1",
        session_id="s1",
        backend="huey",
    )
    assert ok is True

    own_task = db.get_chat_task("task-123", user_id="1")
    assert own_task is not None
    assert own_task["task_id"] == "task-123"
    assert own_task["user_id"] == "1"
    assert own_task["status"] == "processing"

    other_user_task = db.get_chat_task("task-123", user_id="2")
    assert other_user_task is None


def test_chat_task_status_roundtrip(tmp_path):
    db = _make_sqlite_db(tmp_path)
    db.register_chat_task(task_id="task-abc", user_id="5", session_id="s5", backend="celery")

    updated = db.update_chat_task_status(
        "task-abc",
        "completed",
        result={"answer": "ok", "results": []},
    )
    assert updated is True

    row = db.get_chat_task("task-abc", user_id="5")
    assert row is not None
    assert row["status"] == "completed"
    assert row["result"] == {"answer": "ok", "results": []}


def test_chat_task_counter_metrics_and_alerts(tmp_path):
    db = _make_sqlite_db(tmp_path)
    db.register_chat_task(task_id="processing-1", user_id="u1", session_id="s1", backend="huey")
    db.register_chat_task(task_id="processing-2", user_id="u1", session_id="s1", backend="huey")
    db.register_chat_task(task_id="completed-1", user_id="u2", session_id="s2", backend="huey")
    db.update_chat_task_status("completed-1", "completed", result={"answer": "ok"})

    assert db.count_chat_tasks(statuses=("processing",)) == 2
    assert db.count_chat_tasks(user_id="u1", statuses=("processing",)) == 2
    assert db.count_chat_tasks(user_id="u2", statuses=("processing",)) == 0

    db.insert_chat_metric(user_id="u1", endpoint="/api/chat", status_code=200, duration_ms=120.0, queue_depth=3)
    db.insert_chat_metric(user_id="u1", endpoint="/api/chat", status_code=500, duration_ms=220.0, queue_depth=5)
    db.insert_chat_metric(user_id="u2", endpoint="/api/chat/stream", status_code=200, duration_ms=80.0, queue_depth=2)

    summary = db.get_chat_metrics_summary(window_minutes=30, endpoint="/api/chat")
    assert summary["total_requests"] == 2
    assert summary["error_count"] == 1
    assert summary["latency_ms"]["p95"] >= 120.0
    assert summary["queue_depth"]["max"] == 5

    ok = db.insert_chat_alert(
        level="WARNING",
        code="chat:p95_high",
        message="p95 threshold exceeded",
        metric_value=220.0,
        threshold_value=150.0,
    )
    assert ok is True
    alerts = db.get_recent_chat_alerts(limit=5)
    assert len(alerts) >= 1
    assert alerts[0]["code"] == "chat:p95_high"


def test_chat_ttl_purge_helpers(tmp_path):
    db = _make_sqlite_db(tmp_path)
    session_id = "ttl-session"
    db.insert_chat_message(session_id, "user", "old-msg", user_id="u1")
    db.insert_chat_message(session_id, "assistant", "new-msg", user_id="u1")

    db.register_chat_task(task_id="task-old", user_id="u1", session_id=session_id, backend="huey")
    db.register_chat_task(task_id="task-new", user_id="u1", session_id=session_id, backend="huey")
    db.update_chat_task_status("task-old", "completed", result={"answer": "done"})

    old_ts = (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat()
    db.execute(
        "UPDATE chat_history SET timestamp = ? WHERE content = ?",
        (old_ts, "old-msg"),
        commit=True,
    )
    db.execute(
        "UPDATE chat_tasks SET created_at = ?, status = ? WHERE task_id = ?",
        (old_ts, "completed", "task-old"),
        commit=True,
    )

    deleted_history = db.purge_chat_history_older_than(days=7)
    deleted_tasks = db.purge_chat_tasks_older_than(days=7, only_terminal=True)
    assert deleted_history >= 1
    assert deleted_tasks >= 1

    remaining = db.get_chat_history(session_id, limit=50, user_id="u1")
    assert [m["content"] for m in remaining] == ["new-msg"]
    assert db.get_chat_task("task-old", user_id="u1") is None
    assert db.get_chat_task("task-new", user_id="u1") is not None


def test_chat_admission_and_backpressure_helpers():
    chat_routes._USER_ACTIVE_REQUESTS.clear()
    chat_routes._USER_RATE_WINDOWS.clear()

    tuning = {
        "requests_per_minute": 2,
        "max_concurrent_requests_per_user": 1,
        "max_async_tasks_per_user": 2,
        "max_async_tasks_global": 5,
        "async_recent_window_seconds": 900,
    }

    ok, err = chat_routes._admit_user_request("u1", tuning)
    assert ok is True
    assert err is None

    ok, err = chat_routes._admit_user_request("u1", tuning)
    assert ok is False
    assert "concurrent" in err["error"].lower()

    chat_routes._release_user_request("u1")
    ok, err = chat_routes._admit_user_request("u1", tuning)
    assert ok is True
    assert err is None
    chat_routes._release_user_request("u1")

    class _FakeDB:
        def __init__(self, by_user: int, global_count: int):
            self.by_user = by_user
            self.global_count = global_count

        def count_chat_tasks(self, **kwargs):
            if kwargs.get("user_id"):
                return self.by_user
            return self.global_count

    allowed, payload = chat_routes._check_async_backpressure(_FakeDB(1, 2), "u1", tuning)
    assert allowed is True
    assert payload["pending_user_tasks"] == 1

    allowed, payload = chat_routes._check_async_backpressure(_FakeDB(3, 3), "u1", tuning)
    assert allowed is False
    assert "user" in payload["error"].lower()

    allowed, payload = chat_routes._check_async_backpressure(_FakeDB(1, 7), "u1", tuning)
    assert allowed is False
    assert "saturated" in payload["error"].lower()


class _DummyModel:
    def encode(self, texts):  # noqa: D401 - simple stub
        # Return a fixed 2D vector per item.
        return np.array([[0.0, 0.0] for _ in texts], dtype=np.float32)


class _DummyIndex:
    def __init__(self, d, i):
        self._d = d
        self._i = i
        self.ntotal = int(self._i.max()) + 1 if self._i.size else 0

    def search(self, vec, k):  # noqa: D401 - simple stub
        return self._d, self._i


def test_rag_search_filters_fail_closed_on_missing_metadata():
    rag = RAGManager.__new__(RAGManager)
    rag.ensure_loaded = lambda: None  # type: ignore
    rag.model = _DummyModel()

    D = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    I = np.array([[0, 1, 2, 3]], dtype=np.int64)
    rag.index = _DummyIndex(D, I)

    rag.metadata = [
        {"doc_id": 1, "filename": "a", "owner_id": None, "hotel_id": 1, "text": "x"},
        {"doc_id": 2, "filename": "b", "owner_id": 5, "hotel_id": None, "text": "y"},
        {"doc_id": 3, "filename": "c", "owner_id": 5, "hotel_id": 2, "text": "z"},
        {"doc_id": 4, "filename": "d", "owner_id": 7, "hotel_id": 2, "text": "w"},
    ]
    rag.db_manager = None

    results = rag.search("q", k=10, owner_id=5, hotel_ids=[2])
    assert [r["doc_id"] for r in results] == [3]


def test_rag_hybrid_merge_and_rerank():
    rag = RAGManager.__new__(RAGManager)
    rag.ensure_loaded = lambda: None  # type: ignore
    rag.model = _DummyModel()
    rag.db_manager = object()

    rag._vector_search = lambda *args, **kwargs: [  # type: ignore
        {"doc_id": 1, "filename": "a", "text": "alpha", "score": 0.9},
        {"doc_id": 2, "filename": "b", "text": "beta gamma", "score": 0.6},
    ]
    rag._keyword_search = lambda *args, **kwargs: [  # type: ignore
        {"doc_id": 2, "filename": "b", "text": "beta gamma", "score": 0.95},
    ]

    results = rag.search(
        "beta",
        k=2,
        db_manager=object(),
        owner_id=None,
        hotel_ids=None,
        hybrid=True,
        rerank=True,
    )
    assert len(results) == 2
    assert results[0]["doc_id"] == 2


def test_rag_rebuild_incremental_faiss_indexes_only_new_docs():
    rag = RAGManager.__new__(RAGManager)
    rag.ensure_loaded = lambda: None  # type: ignore
    rag.model = object()
    rag.index = object()
    rag.metadata = []

    meta_docs = [
        {"doc_id": 1, "filename": "a.pdf", "md5_hash": "h1", "owner_id": 1, "hotel_id": 1, "text_len": 100},
        {"doc_id": 2, "filename": "b.pdf", "md5_hash": "h2", "owner_id": 1, "hotel_id": 1, "text_len": 120},
    ]
    full_docs = [
        {"doc_id": 2, "filename": "b.pdf", "md5_hash": "h2", "owner_id": 1, "hotel_id": 1, "text": "nuevo"},
    ]

    doc1_meta_hash = rag._fingerprint_metadata(
        filename="a.pdf",
        owner_id=1,
        hotel_id=1,
        md5_hash="h1",
        text_len=100,
    )

    class _FakeDB:
        engine_type = "sqlite"
        config = {}

        def get_rag_index_state(self):
            return {1: {"metadata_hash": doc1_meta_hash}}

        def upsert_rag_index_state_entries(self, entries):
            self.entries = list(entries)
            return len(self.entries)

        def delete_rag_index_state_not_in(self, doc_ids):
            self.cleaned = list(doc_ids)
            return 0

    db = _FakeDB()

    calls = {"added": [], "saved": 0}

    rag._fetch_documents = lambda _db, **kwargs: full_docs if kwargs.get("include_text") and kwargs.get("doc_ids") else (meta_docs if not kwargs.get("include_text") else full_docs)  # type: ignore
    rag.add_document = lambda **kwargs: calls["added"].append(kwargs["doc_id"])  # type: ignore
    rag.save_index = lambda: calls.__setitem__("saved", calls["saved"] + 1)  # type: ignore
    rag._create_new_index = lambda: None  # type: ignore

    rag.rebuild(db, incremental=True)

    assert calls["added"] == [2]
    assert calls["saved"] == 1
