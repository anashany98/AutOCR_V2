import sqlite3
import psycopg2
import yaml
import os
import shutil
from pathlib import Path

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clear_all():
    config = load_config()
    
    # 1. Clear PostgreSQL
    print("Clearing PostgreSQL...")
    try:
        p_conf = config.get("database", {}).get("postgresql", {})
        conn = psycopg2.connect(
            host=p_conf.get("host", "localhost"),
            port=p_conf.get("port", 5432),
            user=p_conf.get("user", "postgres"),
            password=p_conf.get("password", "123"),
            dbname=p_conf.get("dbname", "autocr")
        )
        conn.autocommit = True
        cur = conn.cursor()
        tables = ["ocr_texts", "documents", "metrics"]
        for table in tables:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                print(f"  - Truncated {table}")
            except Exception as e:
                print(f"  - Error truncating {table}: {e}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Postgres Error: {e}")

    # 2. Clear SQLite
    print("\nClearing SQLite...")
    try:
        sqlite_path = Path("data/digitalizerai.db")
        if sqlite_path.exists():
            conn = sqlite3.connect(str(sqlite_path))
            cur = conn.cursor()
            tables = ["ocr_texts", "documents", "metrics"]
            for table in tables:
                try:
                    cur.execute(f"DELETE FROM {table}")
                    print(f"  - Cleared {table}")
                except Exception as e:
                    print(f"  - Error clearing {table}: {e}")
            conn.commit()
            cur.execute("VACUUM")
            conn.close()
        else:
            print("  - SQLite file not found.")
    except Exception as e:
        print(f"SQLite Error: {e}")

    # 3. Clear RAG Index
    print("\nClearing RAG Index...")
    rag_dir = Path("data/rag_index")
    if rag_dir.exists():
        try:
            for item in rag_dir.iterdir():
                if item.is_file():
                    item.unlink()
            print("  - Deleted RAG index files.")
        except Exception as e:
            print(f"  - Error clearing RAG index: {e}")
    
    print("\nCleanup complete.")

if __name__ == "__main__":
    clear_all()
