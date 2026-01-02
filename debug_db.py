
import sqlite3
import os
import yaml

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
db_path = config.get("database", {}).get("path", "data/autocr.db")

print(f"Checking DB at: {db_path}")

if not os.path.exists(db_path):
    print("❌ DB file not found!")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]
    print(f"Total documents in DB: {count}")

    cursor.execute("SELECT id, filename, status, datetime FROM documents ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    print("\nLast 5 documents:")
    for row in rows:
        print(row)
except Exception as e:
    print(f"❌ Error querying DB: {e}")

conn.close()
