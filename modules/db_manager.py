"""
Database manager for AutOCR.

Provides a thin abstraction over SQLite (default) or SQL Server databases to
store processed documents, OCR output, logs and batch metrics.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import queue
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - standard library
    import sqlite3
except ImportError:
    sqlite3 = None

try:  # pragma: no cover - optional dependency
    import psycopg2
    from psycopg2 import extras, pool
except ImportError:
    psycopg2 = None

logger = logging.getLogger(__name__)

class DBManager:
    """Unified interface for interacting with SQLite or PostgreSQL."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        root_config: Dict[str, Any] = config if isinstance(config, dict) else {}
        self.config = root_config.get("database", {})
        
        # Auto-detect Production Environment via DATABASE_URL or specific env vars
        db_url = os.environ.get("DATABASE_URL")
        if db_url and "postgres" in db_url:
            self.engine_type = "postgresql"
        elif os.environ.get("POSTGRES_HOST"):
            self.engine_type = "postgresql"
        else:
            self.engine_type = self.config.get("engine", "sqlite").lower()
        self._lock = threading.RLock()
        self.conn = None
        
        # Abstraction of SQL placeholders
        self.placeholder = "?" if self.engine_type == "sqlite" else "%s"

        if self.engine_type == "postgresql":
            if psycopg2 is None:
                raise RuntimeError("psycopg2 is not installed; cannot connect to PostgreSQL")
            self.pg_conf = self.config.get("postgresql", {})
            try:
                # Initialize ThreadedConnectionPool for Postgres
                pool_args = {
                    "minconn": 2,
                    "maxconn": int(os.getenv("DB_POOL_SIZE", self.config.get("pool_size", 10))),
                }
                
                if db_url:
                    pool_args["dsn"] = db_url
                else:
                    pool_args.update({
                        "host": os.getenv("DB_HOST", self.pg_conf.get("host", "localhost")),
                        "port": int(os.getenv("DB_PORT", self.pg_conf.get("port", 5432))),
                        "user": os.getenv("DB_USER", self.pg_conf.get("user", "postgres")),
                        "password": os.getenv("DB_PASSWORD", self.pg_conf.get("password", "123")),
                        "dbname": os.getenv("DB_NAME", self.pg_conf.get("dbname", "autocr"))
                    })
                
                self._pool = pool.ThreadedConnectionPool(**pool_args)
            except Exception as e:
                 raise RuntimeError(f"Failed to initialize PostgreSQL pool: {e}")

        elif self.engine_type == "sqlite":
            if sqlite3 is None:
                raise RuntimeError("sqlite3 is not available in this environment")
            self.db_path = self.config.get("sqlite", {}).get("path", "data/digitalizerai.db")
            # For SQLite, we use a queue of connections as a simple pool
            pool_size = self.config.get("pool_size", 5)
            self._sqlite_pool = queue.Queue(maxsize=pool_size)
            for _ in range(pool_size):
                self._sqlite_pool.put(self._create_connection())

        # SQLite/local tests rely on auto schema bootstrap.
        # For PostgreSQL, keep it opt-in to avoid interfering with SQL migrations.
        auto_init_default = self.engine_type == "sqlite"
        auto_init_schema = bool(self.config.get("auto_init_schema", auto_init_default))
        if auto_init_schema:
            with self.get_connection() as conn:
                self.initialize_schema(conn)
                self.upgrade_schema(conn)

    def _create_connection(self):
        """Create a new raw database connection."""
        if self.engine_type == "postgresql":
            conn = psycopg2.connect(
                host=self.pg_conf.get("host", "localhost"),
                port=self.pg_conf.get("port", 5432),
                user=self.pg_conf.get("user", "postgres"),
                password=self.pg_conf.get("password", ""),
                dbname=self.pg_conf.get("dbname", "autocr")
            )
            return conn
        else:
            # SQLite concurrency tuning: reduce "database is locked" during parallel ingestion.
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass
            return conn

    @contextmanager
    def get_connection(self):
        """Context manager to borrow a connection from the pool."""
        if self.engine_type == "postgresql":
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                self._pool.putconn(conn)
        else:
            conn = self._sqlite_pool.get()
            try:
                yield conn
            finally:
                self._sqlite_pool.put(conn)

    def upgrade_schema(self, conn=None):
        """Handle migrations/column additions."""
        if conn is None:
            with self.get_connection() as c:
                self.upgrade_schema(c)
            return

        self._upgrade_schema_internal(conn)

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full document details including OCR data."""
        queries = [
            """
            SELECT d.id, d.filename, d.path, d.type, d.status, d.datetime,
                   d.tags, o.text, o.markdown_text, o.structured_data, o.blocks_json,
                   d.hotel_id, d.doc_type, d.visibility, d.financial_level,
                   d.duration, d.workflow_state, o.confidence
            FROM documents d
            LEFT JOIN ocr_texts o ON d.id = o.id_doc
            WHERE d.id = ?
            """,
            """
            SELECT d.id, d.filename, d.file_path, d.doc_type, d.status, d.created_at,
                   d.tags, o.text, o.markdown_text, o.structured_data, o.blocks_json,
                   d.hotel_id, d.doc_type, d.visibility, d.financial_level,
                   d.duration, d.workflow_state, o.confidence
            FROM documents d
            LEFT JOIN ocr_texts o ON d.id = o.id_doc
            WHERE d.id = ?
            """,
        ]

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            row = None
            for query in queries:
                try:
                    cursor.execute(query.replace("?", self.placeholder), (doc_id,))
                    row = cursor.fetchone()
                    if row:
                        break
                except Exception:
                    continue

            if not row:
                return None

            def parse_json(val, default):
                if val in (None, ""):
                    return default
                if isinstance(val, (dict, list)):
                    return val
                try:
                    return json.loads(val)
                except Exception:
                    return default

            return {
                "id": row[0],
                "filename": row[1],
                "path": row[2],
                "type": row[3],
                "status": row[4],
                "date": row[5],
                "tags": parse_json(row[6], []),
                "text": row[7],
                "markdown": row[8],
                "structured_data": parse_json(row[9], {}),
                "blocks": parse_json(row[10], []),
                "hotel_id": row[11],
                "doc_type": row[12],
                "visibility": row[13],
                "financial_level": row[14],
                "duration": row[15] if len(row) > 15 else 0.0,
                "workflow_state": row[16] if len(row) > 16 else "new",
                "confidence": row[17] if len(row) > 17 else 0.0,
                "data": parse_json(row[9], {"total": 0.0, "supplier": "", "date": ""}),
            }
             
    def _upgrade_schema_internal(self, conn):
        for column, definition in (
            ("markdown_text", "TEXT"),
            ("language", "TEXT"),
            ("confidence", "REAL"),
            ("blocks_json", "TEXT"),
            ("tables_json", "TEXT"),
            ("structured_data", "TEXT"),  # JSON: fields, anomalies
        ):
            self._ensure_column("ocr_texts", column, definition, conn)
        
        # Add workflow state and error_message to documents table
        self._ensure_column("documents", "workflow_state", "TEXT DEFAULT 'new'", conn)
        self._ensure_column("documents", "error_message", "TEXT", conn)
        
        # Phase 3/4: Auth & Multi-tenancy
        # Keep both `type` and `doc_type` for backwards compatibility.
        self._ensure_column("documents", "type", "TEXT", conn)
        self._ensure_column("documents", "owner_id", "INTEGER", conn)
        self._ensure_column("documents", "hotel_id", "INTEGER", conn)
        self._ensure_column("documents", "doc_type", "TEXT DEFAULT 'other'", conn)
        self._ensure_column("documents", "visibility", "TEXT DEFAULT 'private'", conn)
        self._ensure_column("documents", "financial_level", "TEXT DEFAULT 'none'", conn)
        
        # Create hotels table
        cursor = self.get_cursor(conn)
        sql_sqlite_hotels = """
            CREATE TABLE IF NOT EXISTS hotels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT UNIQUE,
                description TEXT
            )
        """
        sql_pg_hotels = """
            CREATE TABLE IF NOT EXISTS hotels (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                code TEXT UNIQUE,
                description TEXT
            )
        """
        cursor.execute(sql_sqlite_hotels if self.engine_type == "sqlite" else sql_pg_hotels)
        
        # Create users table if not exists
        cursor = self.get_cursor(conn)
        sql_sqlite_users = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'client',
                client_id TEXT,
                created_at TEXT
            )
        """
        sql_pg_users = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'client',
                client_id TEXT,
                created_at TEXT
            )
        """
        cursor.execute(sql_sqlite_users if self.engine_type == "sqlite" else sql_pg_users)

        # Phase 4: Users Scope & Roles
        # Ensure the users table exists before adding columns.
        self._ensure_column("users", "hotel_scope", "TEXT", conn) # JSON list [1, 2, 3]

        # Phase 5: Auth Enhancements (Email, Verification, Recovery)
        email_def = "TEXT UNIQUE" if self.engine_type != "sqlite" else "TEXT"
        self._ensure_column("users", "email", email_def, conn)
        if self.engine_type == "sqlite":
            # SQLite cannot add a UNIQUE column via ALTER TABLE; use a unique index instead.
            try:
                cursor = self.get_cursor(conn)
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)")
            except Exception:
                logger.debug("Failed to create users(email) unique index.", exc_info=True)
        self._ensure_column("users", "is_verified", "INTEGER DEFAULT 0", conn)
        self._ensure_column("users", "verification_token", "TEXT", conn)
        self._ensure_column("users", "reset_token", "TEXT", conn)
        self._ensure_column("users", "token_expiry", "TEXT", conn)

        # Phase 5: Audit Logging
        cursor = self.get_cursor(conn)
        if self.engine_type == "sqlite":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,
                    timestamp TEXT
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    action TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,
                    timestamp TEXT
                )
            """)

        # Phase 4: Product Catalog (ERP)
        cursor = self.get_cursor(conn)
        if self.engine_type == "sqlite":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    price REAL,
                    stock INTEGER DEFAULT 0,
                    image_url TEXT,
                    embedding TEXT,
                    attributes TEXT,
                    category TEXT,
                    tags TEXT
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    sku TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    price REAL,
                    stock INTEGER DEFAULT 0,
                    image_url TEXT,
                    embedding TEXT,
                    attributes TEXT,
                    category TEXT,
                    tags TEXT
                )
            """)

        # Phase 4.1: Product Attributes (Migration)
        self._ensure_column("products", "attributes", "TEXT", conn) # JSON
        self._ensure_column("products", "category", "TEXT", conn)
        self._ensure_column("products", "tags", "TEXT", conn) # JSON list

    def get_cursor(self, conn=None):
        """Get a cursor from the provided connection or raise error if no connection."""
        if conn is None:
             raise RuntimeError("Use 'with db.get_connection() as conn:' pattern instead of get_cursor()")
             
        if self.engine_type == "postgresql":
            return conn.cursor(cursor_factory=extras.DictCursor)
        return conn.cursor()

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #

    def initialize_schema(self, conn=None) -> None:
        """Create the database schema if it does not already exist."""
        if conn is None:
            with self.get_connection() as c:
                self._initialize_schema_internal(c)
        else:
            self._initialize_schema_internal(conn)

    def _initialize_schema_internal(self, conn) -> None:
        cursor = self.get_cursor(conn)

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    md5_hash TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    duration REAL NOT NULL,
                    status TEXT NOT NULL,
                    type TEXT,
                    tags TEXT,
                    workflow_state TEXT DEFAULT 'new',
                    error_message TEXT
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    md5_hash TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    duration REAL NOT NULL,
                    status TEXT NOT NULL,
                    type TEXT,
                    tags TEXT,
                    workflow_state TEXT DEFAULT 'new',
                    error_message TEXT
                )
                """
            )
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_datetime ON documents(datetime)")

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_doc INTEGER,
                    name TEXT,
                    normalized_name TEXT,
                    quantity REAL,
                    unit TEXT,
                    price_unit REAL,
                    total REAL,
                    FOREIGN KEY(id_doc) REFERENCES documents(id)
                )
                """
            )
            # Phase II: Design History
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS design_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_path TEXT,
                    mode TEXT,
                    style TEXT,
                    results_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

            # OCR text storage (SQLite)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_doc INTEGER NOT NULL,
                    text TEXT,
                    markdown_text TEXT,
                    language TEXT,
                    confidence REAL,
                    blocks_json TEXT,
                    tables_json TEXT,
                    structured_data TEXT,
                    FOREIGN KEY(id_doc) REFERENCES documents(id)
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_texts (
                    id SERIAL PRIMARY KEY,
                    id_doc INTEGER NOT NULL REFERENCES documents(id),
                    text TEXT,
                    markdown_text TEXT,
                    language TEXT,
                    confidence REAL,
                    blocks_json TEXT,
                    tables_json TEXT,
                    structured_data TEXT
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_doc ON ocr_texts(id_doc)")

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT,
                    level TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id SERIAL PRIMARY KEY,
                    datetime TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT,
                    level TEXT NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_datetime ON logs(datetime)")

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT NOT NULL,
                    ok_docs INTEGER NOT NULL,
                    failed_docs INTEGER NOT NULL,
                    avg_time REAL NOT NULL,
                    reliability_pct REAL NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    datetime TEXT NOT NULL,
                    ok_docs INTEGER NOT NULL,
                    failed_docs INTEGER NOT NULL,
                    avg_time REAL NOT NULL,
                    reliability_pct REAL NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_datetime ON metrics(datetime)")

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_search USING fts5(
                    doc_id UNINDEXED,
                    filename,
                    text,
                    tokenize='porter'
                );
                """
            )
        else:
            # PostgreSQL Full Text Search approach (simplest: GIN index on text)
            # Production would use tsvector, but let's keep it simple for now
            if self.config.get("postgresql", {}).get("use_pgvector", False):
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS document_embeddings (
                            id SERIAL PRIMARY KEY,
                            doc_id INTEGER NOT NULL REFERENCES documents(id),
                            embedding vector(384),
                            chunk_text TEXT
                        )
                        """
                    )
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON document_embeddings(doc_id)")
                except Exception as e:
                    if self.engine_type == "postgresql":
                        conn.rollback()
                    logger.warning(f"Failed to initialize pgvector: {e}")
            
            # Simple Full Text Search for PostgreSQL
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_fts ON ocr_texts USING gin(to_tsvector('spanish', text))")
            except Exception as e:
                if self.engine_type == "postgresql":
                    conn.rollback()
                logger.warning(f"Failed to create GIN index for FTS: {e}")

        # Chat History Table
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user_session ON chat_history(user_id, session_id)"
        )

        # Chat Task Registry (for secure async status polling per user)
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    backend TEXT,
                    status TEXT NOT NULL DEFAULT 'processing',
                    result_json TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    backend TEXT,
                    status TEXT NOT NULL DEFAULT 'processing',
                    result_json TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_tasks_user ON chat_tasks(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_tasks_created ON chat_tasks(created_at)")

        # Chat request telemetry (latency/error/queue depth for observability).
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    endpoint TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    queue_depth INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    endpoint TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    queue_depth INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_metrics_created ON chat_metrics(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_metrics_endpoint ON chat_metrics(endpoint)")

        # Persisted alerts emitted by chat SLO checks.
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL,
                    threshold_value REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_alerts (
                    id SERIAL PRIMARY KEY,
                    level TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL,
                    threshold_value REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_alerts_created ON chat_alerts(created_at)")

        # Incremental RAG indexing state (doc fingerprint -> last indexed hash).
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_index_state (
                    doc_id INTEGER PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    metadata_hash TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_index_state (
                    doc_id INTEGER PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    metadata_hash TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_index_state_updated ON rag_index_state(updated_at)")

        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    zones_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    zones_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        
        # Folders Table
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    parent_id INTEGER,
                    created_at TEXT,
                    FOREIGN KEY(parent_id) REFERENCES folders(id)
                )
                """
            )
        else:
             cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS folders (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id INTEGER REFERENCES folders(id),
                    created_at TEXT
                )
                """
            )

        # Document Versions Table
        if self.engine_type == "sqlite":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    version_number INTEGER NOT NULL,
                    text_content TEXT,
                    markdown_content TEXT,
                    structured_data TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    change_reason TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                )
                """
            )
        if self.engine_type != "sqlite":
             cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_versions (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER NOT NULL REFERENCES documents(id),
                    version_number INTEGER NOT NULL,
                    text_content TEXT,
                    markdown_content TEXT,
                    structured_data TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    change_reason TEXT
                )
                """
            )
        
        # COMMIT table creation before migrations!
        # Otherwise _ensure_column might rollback if it fails the first time.
        conn.commit()

        # Ensure we have folder_id in documents table (Migration)
        # Note: We need to do this OUTSIDE the big create block but inside the transaction preferably
        # However, _ensure_column starts its own transaction if conn is not passed.
        # But here we have 'conn'.
        self._ensure_column("documents", "folder_id", "INTEGER REFERENCES folders(id)" if self.engine_type != "sqlite" else "INTEGER", conn=conn)
        
        conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str, conn=None) -> None:
        """Ensure ``table`` includes ``column`` with the provided definition."""
        if conn is None:
             with self.get_connection() as c:
                 self._ensure_column_internal(table, column, definition, c)
        else:
             self._ensure_column_internal(table, column, definition, conn)

    def _ensure_column_internal(self, table, column, definition, conn):
        cursor = self.get_cursor(conn)
        try:
            cursor.execute(f"SELECT {column} FROM {table} LIMIT 1")
        except Exception:
            try:
                # Since we are in a transaction, we might need to rollback before ALTER if previous SELECT failed
                conn.rollback() # Important for Postgres
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                conn.commit()
                logger.info("Added missing column %s.%s", table, column)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not add column %s.%s: %s", table, column, exc)
                conn.rollback()

    # ------------------------------------------------------------------ #
    # CRUD helpers
    # ------------------------------------------------------------------ #

    def check_duplicate(self, md5_hash: str) -> Optional[int]:
        """Return existing document ID if the hash already exists."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(f"SELECT id FROM documents WHERE md5_hash = {self.placeholder}", (md5_hash,))
            row = cursor.fetchone()
            return int(row[0] if isinstance(row, (tuple, list)) else row["id"]) if row else None

    def get_document_path(self, doc_id: int) -> Optional[str]:
        """Return the stored path for a given document ID."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            queries = [
                f"SELECT path FROM documents WHERE id = {self.placeholder}",
                f"SELECT file_path FROM documents WHERE id = {self.placeholder}",
            ]
            for query in queries:
                try:
                    cursor.execute(query, (doc_id,))
                    row = cursor.fetchone()
                    if not row:
                        return None
                    if isinstance(row, (tuple, list)):
                        return str(row[0])
                    try:
                        return str(row["path"])
                    except Exception:
                        try:
                            return str(row["file_path"])
                        except Exception:
                            return str(row[0])
                except Exception:
                    continue
            return None

    def insert_document(
        self,
        filename: str,
        path: str,
        md5_hash: str,
        timestamp: datetime.datetime,
        duration: float,
        status: str,
        doc_type: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        workflow_state: str = "new",
        error_message: Optional[str] = None,
        owner_id: Optional[int] = None,
        hotel_id: Optional[int] = None,
        visibility: str = "private",
        financial_level: str = "none",
    ) -> int:
        """Insert a document record and return its ID."""
        tags_json = json.dumps(list(tags)) if tags else None
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO documents (
                    filename, path, md5_hash, datetime, duration, status, type, tags, 
                    workflow_state, error_message, owner_id, hotel_id, visibility, financial_level
                ) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, 
                          {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, 
                          {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder},
                          {self.placeholder}, {self.placeholder})
            """
            params = (filename, path, md5_hash, timestamp.isoformat(), float(duration), status, doc_type, tags_json, 
                      workflow_state, error_message, owner_id, hotel_id, visibility, financial_level)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                row = cursor.fetchone()
                doc_id = row["id"] if isinstance(row, dict) else row[0]
            else:
                cursor.execute(sql, params)
                doc_id = cursor.lastrowid
                
            conn.commit()
            return int(doc_id)

    def insert_ocr_text(
        self,
        id_doc: int,
        text: str,
        markdown_text: Optional[str] = None,
        language: Optional[str] = None,
        confidence: Optional[float] = None,
        blocks: Optional[Iterable[Dict[str, Any]]] = None,
        tables: Optional[Iterable[Dict[str, Any]]] = None,
        structured_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert OCR text and associated metadata."""
        blocks_json = json.dumps(list(blocks), ensure_ascii=False) if blocks else None
        tables_json = json.dumps(list(tables), ensure_ascii=False) if tables else None
        structured_json = json.dumps(structured_data, ensure_ascii=False) if structured_data else None

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO ocr_texts (
                    id_doc, text, markdown_text, language, confidence, blocks_json, tables_json, structured_data
                ) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, 
                          {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            params = (id_doc, text, markdown_text, language, confidence, blocks_json, tables_json, structured_json)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                ocr_id = cursor.fetchone()["id"]
            else:
                cursor.execute(sql, params)
                ocr_id = cursor.lastrowid

            # Auto-index into FTS (SQLite only for now)
            if text and self.engine_type == "sqlite":
                try:
                    cursor.execute(f"SELECT filename FROM documents WHERE id = ?", (id_doc,))
                    doc_row = cursor.fetchone()
                    fname = doc_row[0] if doc_row else ""
                    cursor.execute(
                        "INSERT INTO documents_search (doc_id, filename, text) VALUES (?, ?, ?)",
                        (id_doc, fname, text),
                    )
                except Exception as e:
                    logger.warning(f"Failed to index document {id_doc} for search: {e}")

            conn.commit()
            return int(ocr_id)

    def insert_log(self, event: str, detail: Optional[str], level: str) -> int:
        """Insert a structured log entry."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"INSERT INTO logs (datetime, event, detail, level) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"
            params = (datetime.datetime.now().isoformat(), event, detail, level)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                log_id = cursor.fetchone()["id"]
            else:
                cursor.execute(sql, params)
                log_id = cursor.lastrowid
                
            conn.commit()
            return int(log_id)

    def log_audit(self, user_id: Optional[int], action: str, resource_type: str, resource_id: Optional[str], details: Any = None) -> int:
        """Log a business or security event for auditing."""
        details_str = json.dumps(details, ensure_ascii=False) if details else None
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            params = (
                user_id,
                action,
                resource_type,
                resource_id,
                details_str,
                datetime.datetime.now().isoformat(),
            )
            last_error = None

            # Support both schema variants:
            # - legacy/local: audit_logs.timestamp
            # - migrated/postgres: audit_logs.created_at
            for time_col in ("timestamp", "created_at"):
                sql = f"""
                    INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details, {time_col})
                    VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
                """
                try:
                    if self.engine_type == "postgresql":
                        cursor.execute(sql + " RETURNING id", params)
                        row = cursor.fetchone()
                        audit_id = row["id"] if isinstance(row, dict) else row[0]
                    else:
                        cursor.execute(sql, params)
                        audit_id = cursor.lastrowid
                    conn.commit()
                    return int(audit_id)
                except Exception as exc:
                    last_error = exc
                    conn.rollback()

            logger.error("Failed to write audit log: %s", last_error)
            return 0

    def get_recent_logs(self, limit: int = 100) -> list:
        """Get recent log entries for monitoring."""
        try:
            with self.get_connection() as conn:
                cursor = self.get_cursor(conn)
                queries = [
                    (
                        f"SELECT created_at AS created_at, action, details, resource_type "
                        f"FROM audit_logs ORDER BY created_at DESC LIMIT {self.placeholder}"
                    ),
                    (
                        f"SELECT timestamp AS created_at, action, details, resource_type "
                        f"FROM audit_logs ORDER BY timestamp DESC LIMIT {self.placeholder}"
                    ),
                ]
                for query in queries:
                    try:
                        cursor.execute(query, (limit,))
                        return cursor.fetchall()
                    except Exception:
                        continue
                return []
        except Exception:
            return []

    def insert_metrics(
        self,
        timestamp: datetime.datetime,
        ok_docs: int,
        failed_docs: int,
        avg_time: float,
        reliability_pct: float,
    ) -> int:
        """Insert aggregated batch metrics."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO metrics (datetime, ok_docs, failed_docs, avg_time, reliability_pct)
                VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            params = (timestamp.isoformat(), ok_docs, failed_docs, avg_time, reliability_pct)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                m_id = cursor.fetchone()["id"]
            else:
                cursor.execute(sql, params)
                m_id = cursor.lastrowid
                
            conn.commit()
            return int(m_id)

    def search_documents(
        self,
        query_text: str,
        limit: int = 50,
        *,
        hotel_ids: Optional[Iterable[int]] = None,
        owner_id: Optional[int] = None,
    ) -> list:
        """Perform a full-text search, optionally scoped by hotel_ids/owner_id."""
        if not query_text.strip():
            return []

        hotel_ids_list = list(hotel_ids) if hotel_ids else []

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            if self.engine_type == "sqlite":
                try:
                    sql = """
                        SELECT documents_search.doc_id,
                               documents_search.filename,
                               snippet(documents_search, 2, '<b>', '</b>', '...', 20) as snippet,
                               rank
                        FROM documents_search
                    """
                    params = [query_text]
                    where = f"WHERE documents_search MATCH {self.placeholder}"

                    if hotel_ids_list or owner_id is not None:
                        sql += " JOIN documents d ON d.id = documents_search.doc_id"
                        if hotel_ids_list:
                            placeholders = ",".join([self.placeholder] * len(hotel_ids_list))
                            where += f" AND d.hotel_id IN ({placeholders})"
                            params.extend(hotel_ids_list)
                        if owner_id is not None:
                            where += f" AND d.owner_id = {self.placeholder}"
                            params.append(owner_id)

                    sql += f"""
                        {where}
                        ORDER BY rank
                        LIMIT {self.placeholder}
                    """
                    params.append(limit)
                    cursor.execute(sql, tuple(params))
                    return cursor.fetchall()
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    return []
            else:
                # Basic PostgreSQL ILIKE search as fallback for full FTS implementation
                sql = f"""
                    SELECT d.id as doc_id,
                           d.filename,
                           LEFT(COALESCE(o.text, ''), 250) as snippet
                    FROM documents d
                    LEFT JOIN ocr_texts o ON d.id = o.id_doc
                    WHERE (d.filename ILIKE {self.placeholder}
                       OR o.text ILIKE {self.placeholder})
                """
                params = [f"%{query_text}%", f"%{query_text}%"]

                if hotel_ids_list:
                    placeholders = ",".join([self.placeholder] * len(hotel_ids_list))
                    sql += f" AND d.hotel_id IN ({placeholders})"
                    params.extend(hotel_ids_list)
                if owner_id is not None:
                    sql += f" AND d.owner_id = {self.placeholder}"
                    params.append(owner_id)

                sql += f" ORDER BY d.datetime DESC LIMIT {self.placeholder}"
                params.append(limit)

                cursor.execute(sql, tuple(params))
                return cursor.fetchall()

    def update_document_metadata(self, doc_id: int, text: str, markdown: str, doc_type: str, status: str) -> bool:
        """Update document content and metadata."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"UPDATE documents SET type = {self.placeholder}, status = {self.placeholder} WHERE id = {self.placeholder}",
                    (doc_type, status, doc_id)
                )
                cursor.execute(
                    f"UPDATE ocr_texts SET text = {self.placeholder}, markdown_text = {self.placeholder} WHERE id_doc = {self.placeholder}",
                    (text, markdown, doc_id)
                )
                if self.engine_type == "sqlite":
                    cursor.execute(
                        f"UPDATE documents_search SET text = {self.placeholder} WHERE doc_id = {self.placeholder}",
                        (text, doc_id)
                    )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to update document {doc_id}: {e}")
                return False

    def update_document_state(self, doc_id: int, workflow_state: str) -> bool:
        """Update the workflow state (new, pending, verified) of a document."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"UPDATE documents SET workflow_state = {self.placeholder} WHERE id = {self.placeholder}",
                    (workflow_state, doc_id)
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to update document state {doc_id}: {e}")
                return False

    def update_document_status(self, doc_id: int, status: str) -> bool:
        """Update the processing status field of a document."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"UPDATE documents SET status = {self.placeholder} WHERE id = {self.placeholder}",
                    (status, doc_id),
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to update document status {doc_id}: {e}")
                return False

    def update_document_type(self, doc_id: int, doc_type: str) -> bool:
        """Update the document type (Invoice, Contract, etc.)."""
        try:
            self.execute(
                "UPDATE documents SET type = ? WHERE id = ?",
                (doc_type, doc_id),
                commit=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update document type {doc_id}: {e}")
            return False

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all related data (OCR, FTS, search, embeddings)."""
        # Start by getting the file path to delete it later
        path_str = self.get_document_path(doc_id)

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                # ORDER MATTERS: Delete children first
                # 1. Embeddings (if exist)
                if self.engine_type == "postgresql":
                    try:
                        # Check table existence to avoid transaction abort
                        cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'document_embeddings'")
                        if cursor.fetchone():
                            cursor.execute("DELETE FROM document_embeddings WHERE doc_id = %s", (doc_id,))
                    except Exception:
                        conn.rollback() # Recover from potential failed check
                        cursor = self.get_cursor(conn)
                
                # 2. Search Index (SQLite only)
                if self.engine_type == "sqlite":
                    cursor.execute("DELETE FROM documents_search WHERE doc_id = ?", (doc_id,))
                
                # 3. OCR Texts
                final_q_ocr = "DELETE FROM ocr_texts WHERE id_doc = ?".replace('?', self.placeholder)
                cursor.execute(final_q_ocr, (doc_id,))
                
                # 4. Document Record
                final_q_doc = "DELETE FROM documents WHERE id = ?".replace('?', self.placeholder)
                cursor.execute(final_q_doc, (doc_id,))
                
                conn.commit()
                
                # 5. Physical File Cleanup (Post-commit)
                if path_str:
                    try:
                        abs_path = Path(path_str)
                        if not abs_path.is_absolute():
                            abs_path = Path(os.getcwd()) / path_str
                        if abs_path.exists():
                            os.remove(abs_path)
                    except Exception as e:
                        logger.error(f"Failed to delete file {path_str}: {e}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                conn.rollback()
                return False




    def insert_template(self, name: str, description: str, zones_json: str) -> int:
        """Insert a new template."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO templates (name, description, zones_json, created_at)
                VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            params = (name, description, zones_json, datetime.datetime.now().isoformat())
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                t_id = cursor.fetchone()["id"]
            else:
                cursor.execute(sql, params)
                t_id = cursor.lastrowid
                
            conn.commit()
            return int(t_id)

    def get_templates(self) -> list:
        """Retrieve all templates."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute("SELECT * FROM templates ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_template(self, template_id: int) -> bool:
        """Delete a template by ID."""
        try:
            self.execute("DELETE FROM templates WHERE id = ?", (template_id,), commit=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # FOLDER MANAGEMENT (New Features)
    # -------------------------------------------------------------------------
    def create_folder(self, name: str, parent_id: Optional[int] = None) -> int:
        query = f"INSERT INTO folders (name, parent_id, created_at) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder})" \
            if self.engine_type == "sqlite" else \
            f"INSERT INTO folders (name, parent_id, created_at) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}) RETURNING id"
        
        from datetime import datetime
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(query, (name, parent_id, now))
            conn.commit()
            if self.engine_type == "sqlite":
                return cursor.lastrowid
            else:
                row = cursor.fetchone()
                return row[0] if row else 0

    def get_hotels(self) -> List[Dict[str, Any]]:
        """List all hotels."""
        query = "SELECT id, name, code, description FROM hotels ORDER BY name ASC"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def create_hotel(self, name: str, code: str, description: str = "") -> int:
        """Create a new hotel."""
        query = f"INSERT INTO hotels (name, code, description) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder})"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            if self.engine_type == "postgresql":
                cursor.execute(query + " RETURNING id", (name, code, description))
                t_id = cursor.fetchone()[0]
            else:
                cursor.execute(query, (name, code, description))
                t_id = cursor.lastrowid
            
            conn.commit()
            return int(t_id)

    def update_hotel(self, hotel_id: int, name: str, code: str, description: str) -> bool:
        """Update hotel details."""
        query = f"UPDATE hotels SET name = {self.placeholder}, code = {self.placeholder}, description = {self.placeholder} WHERE id = {self.placeholder}"
        try:
            with self.get_connection() as conn:
                cursor = self.get_cursor(conn)
                cursor.execute(query, (name, code, description, hotel_id))
                conn.commit()
                return True
        except Exception:
            return False

    def get_folders(self) -> List[Dict[str, Any]]:
        """Get flattened list of folders."""
        query = "SELECT id, name, parent_id, created_at FROM folders ORDER BY name ASC"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(query)
            rows = cursor.fetchall()
            return [
                {"id": r[0], "name": r[1], "parent_id": r[2], "created_at": r[3]}
                for r in rows
            ]

    def move_documents_to_folder(self, doc_ids: List[int], folder_id: Optional[int]) -> bool:
        """Move documents to a specific folder (or root if None)."""
        if not doc_ids: return False
        
        placeholders = ",".join([self.placeholder] * len(doc_ids))
        query = f"UPDATE documents SET folder_id = {self.placeholder} WHERE id IN ({placeholders})"
        params = [folder_id] + doc_ids
        
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(query, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to move docs to folder {folder_id}: {e}")
                conn.rollback()
                return False

    # -------------------------------------------------------------------------
    # VERSION CONTROL (New Features)
    # -------------------------------------------------------------------------
    def create_document_version(self, doc_id: int, reason: str = "Manual Save", user: str = "System") -> bool:
        """Snapshot current document state into versions table."""
        # 1. Get current state
        doc = self.get_document(doc_id)
        if not doc: return False
        
        # 2. Get next version number
        v_query = f"SELECT MAX(version_number) FROM document_versions WHERE doc_id = {self.placeholder}"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(v_query, (doc_id,))
            row = cursor.fetchone()
            current_max = row[0] if row else 0 # Handle potential None if table empty
            next_version = (current_max or 0) + 1
            
            # 3. Insert snapshot
            from datetime import datetime
            now = datetime.now().isoformat()
            
            insert_q = f"""
                INSERT INTO document_versions 
                (doc_id, version_number, text_content, markdown_content, structured_data, created_at, created_by, change_reason)
                VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            
            # Serialize structured_data if it's a dict (get_document returns dict)
            s_data = doc.get("structured_data")
            if isinstance(s_data, dict):
                import json
                s_data = json.dumps(s_data)
                
            params = (
                doc_id, 
                next_version, 
                doc.get("text"), 
                doc.get("markdown"), 
                s_data,
                now,
                user,
                reason
            )
            
            try:
                cursor.execute(insert_q, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to create version for doc {doc_id}: {e}")
                conn.rollback()
                return False

    def get_document_versions(self, doc_id: int) -> List[Dict[str, Any]]:
        query = f"""
            SELECT id, version_number, created_at, created_by, change_reason 
            FROM document_versions 
            WHERE doc_id = {self.placeholder} 
            ORDER BY version_number DESC
        """
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(query, (doc_id,))
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0], 
                    "version": r[1], 
                    "date": r[2], 
                    "user": r[3], 
                    "reason": r[4]
                } for r in rows
            ]

    def restore_version(self, version_id: int) -> bool:
        """Restore a version to the main tables."""
        # Get snapshot
        q_snap = f"SELECT doc_id, text_content, markdown_content, structured_data FROM document_versions WHERE id = {self.placeholder}"
        
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(q_snap, (version_id,))
            row = cursor.fetchone()
            if not row: return False
            
            doc_id, text, markdown, s_data = row
            
            # Update main tables
            # 1. Update ocr_texts
            q_ocr = f"UPDATE ocr_texts SET text = {self.placeholder}, markdown_text = {self.placeholder}, tables_json = {self.placeholder} WHERE id_doc = {self.placeholder}"
            # Note: tables_json is part of structured_data usually, or separate? 
            # In get_document, we merge them. In create_version, we just dumped doc['structured_data'].
            # Let's assume restoration updates the main text fields.
            
            try:
                # We need to be careful mapping fields back.
                # For now, let's just restore text and markdown.
                cursor.execute(q_ocr, (text, markdown, s_data, doc_id))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to restore version {version_id}: {e}")
                conn.rollback()
                return False

    def insert_chat_message(self, session_id: str, role: str, content: str, user_id: Optional[str] = None) -> int:
        """Insert a chat message into history."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO chat_history (session_id, role, content, timestamp, user_id)
                VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            params = (session_id, role, content, datetime.datetime.now().isoformat(), user_id)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                msg_id = cursor.fetchone()["id"]
            else:
                cursor.execute(sql, params)
                msg_id = cursor.lastrowid
                
            conn.commit()
            return int(msg_id)

    def register_chat_task(
        self,
        task_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> bool:
        """Register an async chat task for secure status polling."""
        now = datetime.datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                if self.engine_type == "postgresql":
                    cursor.execute(
                        f"""
                        INSERT INTO chat_tasks (
                            task_id, user_id, session_id, backend, status, result_json, error_text, created_at, updated_at
                        ) VALUES (
                            {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder},
                            {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}
                        )
                        ON CONFLICT (task_id) DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            session_id = EXCLUDED.session_id,
                            backend = EXCLUDED.backend,
                            status = EXCLUDED.status,
                            result_json = EXCLUDED.result_json,
                            error_text = EXCLUDED.error_text,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (task_id, str(user_id), session_id, backend, "processing", None, None, now, now),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO chat_tasks (
                            task_id, user_id, session_id, backend, status, result_json, error_text, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """.replace("?", self.placeholder),
                        (task_id, str(user_id), session_id, backend, "processing", None, None, now, now),
                    )
                conn.commit()
                return True
            except Exception as exc:
                logger.error("Failed to register chat task %s: %s", task_id, exc)
                conn.rollback()
                return False

    def get_chat_task(self, task_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a chat task, optionally enforcing ownership via user_id."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                params = [task_id]
                where_sql = f"WHERE task_id = {self.placeholder}"
                if user_id is not None:
                    where_sql += f" AND user_id = {self.placeholder}"
                    params.append(str(user_id))
                cursor.execute(
                    f"""
                    SELECT task_id, user_id, session_id, backend, status, result_json, error_text, created_at, updated_at
                    FROM chat_tasks
                    {where_sql}
                    LIMIT 1
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                def _field(key: str, idx: int):
                    if isinstance(row, (tuple, list)):
                        return row[idx]
                    try:
                        return row[key]
                    except Exception:
                        return row[idx]

                result_json = _field("result_json", 5)
                parsed_result = None
                if result_json:
                    try:
                        parsed_result = json.loads(result_json)
                    except Exception:
                        parsed_result = None

                return {
                    "task_id": _field("task_id", 0),
                    "user_id": _field("user_id", 1),
                    "session_id": _field("session_id", 2),
                    "backend": _field("backend", 3),
                    "status": _field("status", 4),
                    "result": parsed_result,
                    "error": _field("error_text", 6),
                    "created_at": _field("created_at", 7),
                    "updated_at": _field("updated_at", 8),
                }
            except Exception as exc:
                logger.error("Failed to fetch chat task %s: %s", task_id, exc)
                return None

    def update_chat_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update status/result metadata for a chat task."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"""
                    UPDATE chat_tasks
                    SET status = {self.placeholder},
                        result_json = {self.placeholder},
                        error_text = {self.placeholder},
                        updated_at = {self.placeholder}
                    WHERE task_id = {self.placeholder}
                    """,
                    (
                        str(status),
                        json.dumps(result) if result is not None else None,
                        str(error) if error is not None else None,
                        datetime.datetime.now().isoformat(),
                        task_id,
                    ),
                )
                conn.commit()
                return True
            except Exception as exc:
                logger.error("Failed to update chat task %s: %s", task_id, exc)
                conn.rollback()
                return False

    def get_chat_history(
        self, session_id: str, limit: int = 50, user_id: Optional[str] = None
    ) -> list:
        """Retrieve recent chat history for a specific session, optionally scoped by user_id."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)

            params = [session_id]
            where_sql = f"WHERE session_id = {self.placeholder}"
            if user_id is not None:
                where_sql += f" AND user_id = {self.placeholder}"
                params.append(str(user_id))

            # Fetch last N messages then return them in chronological order.
            sql = f"""
                SELECT role, content, timestamp
                FROM (
                    SELECT id, role, content, timestamp
                    FROM chat_history
                    {where_sql}
                    ORDER BY id DESC
                    LIMIT {self.placeholder}
                ) t
                ORDER BY id ASC
            """
            params.append(int(limit))
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def count_chat_tasks(
        self,
        *,
        user_id: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
        recent_seconds: Optional[int] = None,
    ) -> int:
        """Count chat tasks with optional user/status/time filters."""
        where_clauses: List[str] = []
        params: List[Any] = []

        if user_id is not None:
            where_clauses.append(f"user_id = {self.placeholder}")
            params.append(str(user_id))

        status_list = [str(s) for s in (statuses or []) if str(s).strip()]
        if status_list:
            placeholders = ",".join([self.placeholder] * len(status_list))
            where_clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)

        if recent_seconds is not None and int(recent_seconds) > 0:
            cutoff = (datetime.datetime.now() - datetime.timedelta(seconds=int(recent_seconds))).isoformat()
            where_clauses.append(f"created_at >= {self.placeholder}")
            params.append(cutoff)

        sql = "SELECT COUNT(1) FROM chat_tasks"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(sql, tuple(params))
                row = cursor.fetchone()
            except Exception as exc:
                logger.debug("count_chat_tasks failed: %s", exc, exc_info=True)
                return 0
            if not row:
                return 0
            if isinstance(row, (tuple, list)):
                return int(row[0] or 0)
            try:
                return int(row[0] or 0)
            except Exception:
                return int(row.get("count", 0) or 0)

    def insert_chat_metric(
        self,
        *,
        user_id: Optional[str],
        endpoint: str,
        status_code: int,
        duration_ms: float,
        queue_depth: Optional[int] = None,
    ) -> bool:
        """Insert a chat request telemetry row."""
        now = datetime.datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"""
                    INSERT INTO chat_metrics (
                        user_id, endpoint, status_code, duration_ms, queue_depth, created_at
                    ) VALUES (
                        {self.placeholder}, {self.placeholder}, {self.placeholder},
                        {self.placeholder}, {self.placeholder}, {self.placeholder}
                    )
                    """,
                    (
                        str(user_id) if user_id is not None else None,
                        str(endpoint),
                        int(status_code),
                        float(duration_ms),
                        int(queue_depth) if queue_depth is not None else None,
                        now,
                    ),
                )
                conn.commit()
                return True
            except Exception as exc:
                logger.error("Failed to insert chat metric: %s", exc)
                conn.rollback()
                return False

    def get_chat_metrics_summary(
        self,
        *,
        window_minutes: int = 15,
        endpoint: Optional[str] = None,
        max_rows: int = 5000,
    ) -> Dict[str, Any]:
        """Return latency/error/queue summary for recent chat requests."""
        minutes = max(1, int(window_minutes))
        cutoff = (datetime.datetime.now() - datetime.timedelta(minutes=minutes)).isoformat()

        where = [f"created_at >= {self.placeholder}"]
        params: List[Any] = [cutoff]
        if endpoint:
            where.append(f"endpoint = {self.placeholder}")
            params.append(str(endpoint))

        sql = f"""
            SELECT duration_ms, status_code, queue_depth
            FROM chat_metrics
            WHERE {' AND '.join(where)}
            ORDER BY id DESC
            LIMIT {self.placeholder}
        """
        params.append(max(1, int(max_rows)))

        durations: List[float] = []
        status_codes: List[int] = []
        queue_depths: List[int] = []

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall() or []
            except Exception:
                rows = []

        for row in rows:
            if isinstance(row, (tuple, list)):
                dur, code, qd = row[0], row[1], row[2]
            else:
                try:
                    dur = row["duration_ms"]
                except Exception:
                    dur = row[0]
                try:
                    code = row["status_code"]
                except Exception:
                    code = row[1]
                try:
                    qd = row["queue_depth"]
                except Exception:
                    qd = row[2]
            try:
                durations.append(float(dur))
            except Exception:
                pass
            try:
                status_codes.append(int(code))
            except Exception:
                pass
            try:
                if qd is not None:
                    queue_depths.append(int(qd))
            except Exception:
                pass

        durations_sorted = sorted(durations)

        def _percentile(values: List[float], pct: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return float(values[0])
            rank = (pct / 100.0) * (len(values) - 1)
            low = int(rank)
            high = min(low + 1, len(values) - 1)
            frac = rank - low
            return float(values[low] * (1.0 - frac) + values[high] * frac)

        total = len(durations_sorted)
        error_count = sum(1 for c in status_codes if int(c) >= 400)
        queue_avg = (sum(queue_depths) / len(queue_depths)) if queue_depths else 0.0
        queue_max = max(queue_depths) if queue_depths else 0

        return {
            "window_minutes": minutes,
            "total_requests": total,
            "error_count": error_count,
            "error_rate_pct": (100.0 * error_count / total) if total else 0.0,
            "latency_ms": {
                "avg": (sum(durations_sorted) / total) if total else 0.0,
                "p95": _percentile(durations_sorted, 95.0),
                "p99": _percentile(durations_sorted, 99.0),
                "max": max(durations_sorted) if durations_sorted else 0.0,
            },
            "queue_depth": {
                "avg": queue_avg,
                "max": queue_max,
            },
        }

    def insert_chat_alert(
        self,
        *,
        level: str,
        code: str,
        message: str,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
    ) -> bool:
        """Persist a chat SLO alert record."""
        now = datetime.datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    f"""
                    INSERT INTO chat_alerts (
                        level, code, message, metric_value, threshold_value, created_at
                    ) VALUES (
                        {self.placeholder}, {self.placeholder}, {self.placeholder},
                        {self.placeholder}, {self.placeholder}, {self.placeholder}
                    )
                    """,
                    (
                        str(level).upper(),
                        str(code),
                        str(message),
                        float(metric_value) if metric_value is not None else None,
                        float(threshold_value) if threshold_value is not None else None,
                        now,
                    ),
                )
                conn.commit()
                return True
            except Exception as exc:
                logger.error("Failed to insert chat alert: %s", exc)
                conn.rollback()
                return False

    def get_recent_chat_alerts(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent chat alerts in reverse chronological order."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(
                f"""
                SELECT id, level, code, message, metric_value, threshold_value, created_at
                FROM chat_alerts
                ORDER BY id DESC
                LIMIT {self.placeholder}
                """,
                (max(1, int(limit)),),
            )
            rows = cursor.fetchall() or []
            return [dict(r) for r in rows]

    def purge_chat_history_older_than(self, *, days: float, user_id: Optional[str] = None) -> int:
        """Delete chat_history records older than `days` and return deleted count."""
        max_age = float(days)
        if max_age <= 0:
            return 0
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=max_age)).isoformat()

        where = [f"timestamp < {self.placeholder}"]
        params: List[Any] = [cutoff]
        if user_id is not None:
            where.append(f"user_id = {self.placeholder}")
            params.append(str(user_id))

        sql = f"DELETE FROM chat_history WHERE {' AND '.join(where)}"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(sql, tuple(params))
            deleted = int(getattr(cursor, "rowcount", 0) or 0)
            conn.commit()
            return deleted

    def purge_chat_tasks_older_than(
        self,
        *,
        days: float,
        only_terminal: bool = True,
    ) -> int:
        """Delete old chat_tasks (optionally only terminal statuses)."""
        max_age = float(days)
        if max_age <= 0:
            return 0
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=max_age)).isoformat()

        where = [f"created_at < {self.placeholder}"]
        params: List[Any] = [cutoff]
        if only_terminal:
            terminal = ("completed", "failed", "error")
            placeholders = ",".join([self.placeholder] * len(terminal))
            where.append(f"status IN ({placeholders})")
            params.extend(terminal)

        sql = f"DELETE FROM chat_tasks WHERE {' AND '.join(where)}"
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(sql, tuple(params))
            deleted = int(getattr(cursor, "rowcount", 0) or 0)
            conn.commit()
            return deleted

    def get_rag_index_state(self) -> Dict[int, Dict[str, Any]]:
        """Load RAG index state keyed by doc_id."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                cursor.execute(
                    "SELECT doc_id, content_hash, metadata_hash, updated_at FROM rag_index_state"
                )
                rows = cursor.fetchall() or []
            except Exception as exc:
                logger.debug("get_rag_index_state failed: %s", exc, exc_info=True)
                return {}
            out: Dict[int, Dict[str, Any]] = {}
            for row in rows:
                if isinstance(row, (tuple, list)):
                    doc_id, content_hash, metadata_hash, updated_at = row[0], row[1], row[2], row[3]
                else:
                    doc_id = row["doc_id"]
                    content_hash = row["content_hash"]
                    metadata_hash = row["metadata_hash"]
                    updated_at = row["updated_at"]
                try:
                    doc_id_int = int(doc_id)
                except Exception:
                    continue
                out[doc_id_int] = {
                    "content_hash": str(content_hash or ""),
                    "metadata_hash": str(metadata_hash or ""),
                    "updated_at": updated_at,
                }
            return out

    def upsert_rag_index_state_entries(self, entries: Iterable[Dict[str, Any]]) -> int:
        """Upsert RAG index state rows. Returns number of applied entries."""
        rows = []
        now = datetime.datetime.now().isoformat()
        for item in entries or []:
            try:
                doc_id = int(item.get("doc_id"))
            except Exception:
                continue
            rows.append(
                (
                    doc_id,
                    str(item.get("content_hash") or ""),
                    str(item.get("metadata_hash") or ""),
                    str(item.get("updated_at") or now),
                )
            )
        if not rows:
            return 0

        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            applied = 0
            try:
                if self.engine_type == "postgresql":
                    sql = (
                        f"""
                        INSERT INTO rag_index_state (doc_id, content_hash, metadata_hash, updated_at)
                        VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
                        ON CONFLICT (doc_id) DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            metadata_hash = EXCLUDED.metadata_hash,
                            updated_at = EXCLUDED.updated_at
                        """
                    )
                else:
                    sql = (
                        f"""
                        INSERT INTO rag_index_state (doc_id, content_hash, metadata_hash, updated_at)
                        VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
                        ON CONFLICT(doc_id) DO UPDATE SET
                            content_hash = excluded.content_hash,
                            metadata_hash = excluded.metadata_hash,
                            updated_at = excluded.updated_at
                        """
                    )

                for row in rows:
                    cursor.execute(sql, row)
                    applied += 1
                conn.commit()
            except Exception as exc:
                logger.error("Failed to upsert rag_index_state entries: %s", exc)
                conn.rollback()
                return 0
            return applied

    def delete_rag_index_state_not_in(self, doc_ids: Iterable[int]) -> int:
        """Delete rag_index_state rows whose doc_id is not in `doc_ids`."""
        valid_ids = []
        for d in doc_ids or []:
            try:
                valid_ids.append(int(d))
            except Exception:
                continue
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            try:
                if not valid_ids:
                    cursor.execute("DELETE FROM rag_index_state")
                else:
                    placeholders = ",".join([self.placeholder] * len(valid_ids))
                    cursor.execute(
                        f"DELETE FROM rag_index_state WHERE doc_id NOT IN ({placeholders})",
                        tuple(valid_ids),
                    )
                deleted = int(getattr(cursor, "rowcount", 0) or 0)
                conn.commit()
                return deleted
            except Exception as exc:
                logger.error("Failed to delete stale rag_index_state entries: %s", exc)
                conn.rollback()
                return 0

    def execute(self, query: str, params: tuple = (), commit: bool = False):
        """Helper to execute a query with automatic placeholder replacement and connection management."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            # Standardize query
            final_query = query.replace("?", self.placeholder)
            cursor.execute(final_query, params)
            if commit:
                conn.commit()
            return cursor

    def close(self) -> None:
        """Close all connections in the pool."""
        if self.engine_type == "postgresql":
            self._pool.closeall()
        else:
            while not self._sqlite_pool.empty():
                try:
                    conn = self._sqlite_pool.get_nowait()
                    conn.close()
                except Exception:
                    pass


__all__ = ["DBManager"]
