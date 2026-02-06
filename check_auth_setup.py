
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.db_manager import DBManager
from web_app.services import load_configuration, get_db

def check_schema():
    print("Checking Database Schema...")
    config = load_configuration()
    # Force DB init
    db = get_db()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check users table
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if cursor.fetchone():
                print("✅ Table 'users' exists.")
            else:
                print("❌ Table 'users' MISSING.")
        except Exception as e:
             print(f"Error checking users table: {e}")

        # Check owner_id in documents
        try:
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'owner_id' in columns:
                 print("✅ Column 'owner_id' exists in documents.")
            else:
                 print("❌ Column 'owner_id' MISSING in documents.")
        except Exception as e:
            print(f"Error checking documents columns: {e}")

def check_imports():
    print("\nChecking Imports...")
    try:
        from web_app.app import app
        print("✅ Flask App imported successfully (Blueprints OK).")
    except Exception as e:
        print(f"❌ Error importing Flask App: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_schema()
    check_imports()
