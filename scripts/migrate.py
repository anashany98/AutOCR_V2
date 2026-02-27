"""
Database Migration Runner.

Applies SQL migration files from the ``migrations/`` directory in order.
Tracks applied migrations in a ``_migrations`` table to ensure idempotency.

Usage::

    python -m scripts.migrate          # Apply all pending
    python -m scripts.migrate --dry    # Preview without applying
    python -m scripts.migrate --reset  # Drop all and reapply (⚠ destructive)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def get_pg_connection(config: dict):
    """Create a PostgreSQL connection from config.yaml settings."""
    pg = config.get("postgresql", {})
    return psycopg2.connect(
        host=pg.get("host", os.environ.get("DB_HOST", "localhost")),
        port=pg.get("port", int(os.environ.get("DB_PORT", 5432))),
        dbname=pg.get("dbname", os.environ.get("DB_NAME", "autoocr")),
        user=pg.get("user", os.environ.get("DB_USER", "autoocr")),
        password=pg.get("password", os.environ.get("DB_PASSWORD", "autoocr")),
    )


def ensure_migrations_table(conn):
    """Create the migrations tracking table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMPTZ DEFAULT NOW(),
                checksum TEXT
            )
        """)
    conn.commit()


def get_applied_migrations(conn) -> set:
    """Return set of already-applied migration filenames."""
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM _migrations ORDER BY id")
        return {row[0] for row in cur.fetchall()}


def get_pending_migrations(applied: set) -> list:
    """Return sorted list of migration files not yet applied."""
    if not MIGRATIONS_DIR.exists():
        logger.error("Migrations directory not found: %s", MIGRATIONS_DIR)
        return []

    all_files = sorted(
        f for f in MIGRATIONS_DIR.glob("*.sql")
        if f.name not in applied
    )
    return all_files


def apply_migration(conn, path: Path, dry_run: bool = False):
    """Apply a single migration file."""
    sql = path.read_text(encoding="utf-8")
    filename = path.name

    if dry_run:
        logger.info("  [DRY RUN] Would apply: %s (%d bytes)", filename, len(sql))
        return

    logger.info("  Applying: %s ...", filename)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO _migrations (filename, checksum) VALUES (%s, %s)",
                (filename, str(len(sql))),
            )
        conn.commit()
        logger.info("  ✅ Applied: %s", filename)
    except Exception as e:
        conn.rollback()
        logger.error("  ❌ Failed: %s — %s", filename, e)
        raise


def run_migrations(dry_run: bool = False, reset: bool = False):
    """Main migration runner."""
    # Load config
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    conn = get_pg_connection(config)
    logger.info("Connected to PostgreSQL")

    try:
        if reset:
            logger.warning("⚠ RESET mode — dropping all tables!")
            with conn.cursor() as cur:
                cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            conn.commit()
            logger.info("Schema reset complete")

        ensure_migrations_table(conn)
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied)

        if not pending:
            logger.info("No pending migrations.")
            return

        logger.info("Found %d pending migration(s):", len(pending))
        for p in pending:
            apply_migration(conn, p, dry_run=dry_run)

        if not dry_run:
            logger.info("All migrations applied successfully! ✅")
        else:
            logger.info("Dry run complete — no changes made.")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="AutoOCR Database Migration Runner")
    parser.add_argument("--dry", action="store_true", help="Preview without applying")
    parser.add_argument("--reset", action="store_true", help="⚠ Drop all and reapply")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.reset and not args.force:
        confirm = input("⚠ This will DELETE all data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    run_migrations(dry_run=args.dry, reset=args.reset)


if __name__ == "__main__":
    main()
