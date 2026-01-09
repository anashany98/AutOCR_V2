from web_app.services import get_db
import sys

def check_schema():
    print("Checking schema...")
    db = get_db()
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        
        # Check tables
        tables = ["folders", "document_versions"]
        for t in tables:
            if db.engine_type == "sqlite":
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{t}'")
            else:
                cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_name='{t}'")
                
            if cursor.fetchone():
                print(f"[OK] Table '{t}' exists.")
            else:
                print(f"[FAIL] Table '{t}' MISSING.")
                
        # Check column
        try:
            cursor.execute(f"SELECT folder_id FROM documents LIMIT 1")
            print("[OK] Column 'folder_id' exists in 'documents'.")
        except Exception as e:
            print(f"[FAIL] Column 'folder_id' missing or error: {e}")

if __name__ == "__main__":
    try:
        check_schema()
    except Exception as e:
        print(f"Error: {e}")
