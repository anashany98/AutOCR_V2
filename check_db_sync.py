import sqlite3
import psycopg2
import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_db():
    config = load_config()
    
    # Check SQLite
    try:
        sqlite_path = Path("data/digitalizerai.db")
        if sqlite_path.exists():
            conn = sqlite_path.as_posix()
            s_conn = sqlite3.connect(s_conn) # Error here, fixed below
            pass
    except: pass

    # Refined Check
    print("--- SQLite Status ---")
    try:
        s_conn = sqlite3.connect("data/digitalizerai.db")
        s_cur = s_conn.cursor()
        s_cur.execute("SELECT count(*) FROM documents")
        doc_count = s_cur.fetchone()[0]
        s_cur.execute("SELECT count(*) FROM ocr_texts")
        ocr_count = s_cur.fetchone()[0]
        print(f"Documents: {doc_count}")
        print(f"OCR Texts: {ocr_count}")
        s_conn.close()
    except Exception as e:
        print(f"SQLite Error: {e}")

    print("\n--- Postgres Status ---")
    try:
        p_conf = config.get("database", {}).get("postgresql", {})
        p_conn = psycopg2.connect(
            host=p_conf.get("host", "localhost"),
            port=p_conf.get("port", 5432),
            user=p_conf.get("user", "postgres"),
            password=p_conf.get("password", "123"),
            dbname=p_conf.get("dbname", "autocr")
        )
        p_cur = p_conn.cursor()
        p_cur.execute("SELECT count(*) FROM documents")
        doc_count = p_cur.fetchone()[0]
        p_cur.execute("SELECT count(*) FROM ocr_texts")
        ocr_count = p_cur.fetchone()[0]
        print(f"Documents: {doc_count}")
        print(f"OCR Texts: {ocr_count}")
        
        print("\nRecent Documents (Postgres):")
        p_cur.execute("SELECT filename FROM documents ORDER BY created_at DESC LIMIT 10")
        for row in p_cur.fetchall():
            print(f"- {row[0]}")
            
        p_conn.close()
    except Exception as e:
        print(f"Postgres Error: {e}")

if __name__ == "__main__":
    check_db()
