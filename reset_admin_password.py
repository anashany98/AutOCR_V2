
import sys
import os
from pathlib import Path
from werkzeug.security import generate_password_hash

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Manually setup DB connection to avoid full app context if possible, 
# or use the existing modules.
from modules.db_manager import DBManager

def reset_password(username, new_password):
    db = DBManager(config={"database": {"type": "sqlite", "path": "data/digitalizerai.db"}})
    
    # Check if user exists
    query = "SELECT id FROM users WHERE username = ?"
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query.replace('?', db.placeholder), (username,))
        row = cursor.fetchone()
        
        if not row:
            print(f"❌ User '{username}' not found.")
            return

        # Update password and ensure ADMIN role
        password_hash = generate_password_hash(new_password)
        update_query = "UPDATE users SET password_hash = ?, role = 'ADMIN' WHERE username = ?"
        cursor.execute(update_query.replace('?', db.placeholder), (password_hash, username))
        conn.commit()
        print(f"✅ Password for user '{username}' has been reset to '{new_password}'.")

if __name__ == "__main__":
    reset_password("admin", "admin123")
