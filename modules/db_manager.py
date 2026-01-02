"""
Database manager for AutOCR.

Provides a thin abstraction over SQLite (default) or SQL Server databases to
store processed documents, OCR output, logs and batch metrics.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import queue
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

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
        config: Dict[str, Any]
    ) -> None:
        self.config = config.get("database", {})
        self.engine_type = self.config.get("engine", "sqlite").lower()
        self._lock = threading.RLock()
        self.conn = None
        
        # Abstraction of SQL placeholders
        self.placeholder = "?" if self.engine_type == "sqlite" else "%s"

        if self.engine_type == "postgresql":
            if psycopg2 is None:
                raise RuntimeError("psycopg2 is not installed; cannot connect to PostgreSQL")
            self.pg_conf = self.config.get("postgresql", {})
            # Validate connection params by creating one test connection
            try:
                test_conn = self._create_connection()
                test_conn.close()
            except Exception as e:
                 raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")

        elif self.engine_type == "sqlite":
            if sqlite3 is None:
                raise RuntimeError("sqlite3 is not available in this environment")
            self.db_path = self.config.get("sqlite", {}).get("path", "data/digitalizerai.db")

        # Initialize connection pool
        pool_size = self.config.get("pool_size", 5)
        self._pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._pool.put(self._create_connection())

        # Initialize schema using a temporary connection from pool
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
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    @contextmanager
    def get_connection(self):
        """Context manager to borrow a connection from the pool."""
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def upgrade_schema(self, conn=None):
        """Handle migrations/column additions."""
        if conn is None:
            with self.get_connection() as c:
                self.upgrade_schema(c)
            return

        self._upgrade_schema_internal(conn)

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full document details including OCR data."""
        query = """
        SELECT d.id, d.filename, d.path, d.type, d.status, d.datetime,
               d.tags, o.text, o.markdown_text, o.structured_data, o.blocks_json
        FROM documents d
        LEFT JOIN ocr_texts o ON d.id = o.id_doc
        WHERE d.id = ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', self.placeholder), (doc_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            def parse_json(val):
                if not val: return []
                try: return json.loads(val)
                except: return []

            return {
                "id": row[0],
                "filename": row[1],
                "path": row[2],
                "type": row[3],
                "status": row[4],
                "date": row[5],
                "tags": parse_json(row[6]),
                "text": row[7],
                "markdown": row[8],
                "structured_data": parse_json(row[9]) if row[9] else {},
                "blocks": parse_json(row[10]),
                "data": parse_json(row[9]) if row[9] else {"total":0.0, "supplier":"", "date":""}
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
        
        # Add workflow state to documents table
        self._ensure_column("documents", "workflow_state", "TEXT DEFAULT 'new'", conn)

    def get_cursor(self, conn=None):
        """Get a cursor from the provided connection or raise error if no connection."""
        if conn is None:
             raise RuntimeError("Use 'with db.get_connection() as conn:' pattern instead of get_cursor()")
             
        if self.engine_type == "postgresql":
            return conn.cursor(cursor_factory=extras.RealDictCursor)
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
                    workflow_state TEXT DEFAULT 'new'
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
                    workflow_state TEXT DEFAULT 'new'
                )
                """
            )
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_datetime ON documents(datetime)")

        if self.engine_type == "sqlite":
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
                    tables_json TEXT
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
                            embedding vector(512)
                        )
                        """
                    )
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON document_embeddings(doc_id)")
                except Exception as e:
                    logger.warning(f"Failed to initialize pgvector: {e}")

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
            cursor.execute(f"SELECT path FROM documents WHERE id = {self.placeholder}", (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return str(row[0] if isinstance(row, (tuple, list)) else row["path"])

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
    ) -> int:
        """Insert a document record and return its ID."""
        tags_json = json.dumps(list(tags)) if tags else None
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            sql = f"""
                INSERT INTO documents (
                    filename, path, md5_hash, datetime, duration, status, type, tags, workflow_state
                ) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, 
                          {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
            """
            params = (filename, path, md5_hash, timestamp.isoformat(), float(duration), status, doc_type, tags_json, workflow_state)
            
            if self.engine_type == "postgresql":
                sql += " RETURNING id"
                cursor.execute(sql, params)
                doc_id = cursor.fetchone()["id"]
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

    def get_recent_logs(self, limit: int = 100) -> list:
        """Get recent log entries for monitoring."""
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            cursor.execute(
                f"SELECT datetime, event, detail, level FROM logs ORDER BY datetime DESC LIMIT {self.placeholder}",
                (limit,),
            )
            return cursor.fetchall()

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

    def search_documents(self, query_text: str, limit: int = 50) -> list:
        """Perform a full-text search."""
        if not query_text.strip():
            return []
        
        with self.get_connection() as conn:
            cursor = self.get_cursor(conn)
            if self.engine_type == "sqlite":
                try:
                    cursor.execute(
                        """
                        SELECT doc_id, filename, snippet(documents_search, 2, '<b>', '</b>', '...', 20) as snippet, rank
                        FROM documents_search
                        WHERE documents_search MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query_text, limit),
                    )
                    return cursor.fetchall()
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    return []
            else:
                # Basic PostgreSQL ILIKE search as fallback for full FTS implementation
                cursor.execute(
                    f"SELECT id as doc_id, filename, content as snippet FROM ocr_texts WHERE text ILIKE {self.placeholder} LIMIT {self.placeholder}",
                    (f"%{query_text}%", limit)
                )
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



    def close(self) -> None:
        """Close all connections in the pool."""
        # Simple implementation: drain pool and close.
        # In production this might need to handle in-use connections gracefully.
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass


__all__ = ["DBManager"]
