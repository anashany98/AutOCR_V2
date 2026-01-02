
import sqlite3
import os

db_path = "data/digitalizerai.db"
print(f"Checking DB at: {db_path} (Absolute: {os.path.abspath(db_path)})")

if not os.path.exists(db_path):
    print("❌ DB file not found!")
else:
    print(f"✅ DB file exists. Size: {os.path.getsize(db_path)} bytes")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[t[0] for t in tables]}")

        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        print(f"Total documents: {count}")

        if count > 0:
            cursor.execute("SELECT id, filename, status, datetime FROM documents ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            print("\nLast 5 documents:")
            for row in rows:
                print(row)
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")
