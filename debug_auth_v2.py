
import sys
import os
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.db_manager import DBManager
from modules.auth_manager import AuthManager

def diagnose_auth(username, password):
    print(f"🔍 Diagnosing auth for user: {username}")
    
    # Initialize DB with same config as app
    db = DBManager(config={"database": {"type": "sqlite", "path": "data/digitalizerai.db"}})
    auth = AuthManager(db)
    
    # 1. Check if user exists
    user = auth.get_user_by_username(username)
    if not user:
        print(f"❌ User '{username}' does NOT exist in the database.")
        print(f"   DB Path used: {os.path.abspath('data/digitalizerai.db')}")
        return
    else:
        print(f"✅ User '{username}' found. Role: {user.role}")

    # 2. Check Password Hash
    print(f"   Stored Hash: {user.password_hash}")
    
    # 3. Test Verification
    is_valid = auth.verify_password(user, password)
    if is_valid:
        print(f"✅ check_password_hash passed!")
    else:
        print(f"❌ check_password_hash FAILED.")
        
        # Test generation info
        new_hash = generate_password_hash(password)
        print(f"   New Hash would be: {new_hash}")
        
        # Test compatibility
        print(f"   Verifying new hash: {check_password_hash(new_hash, password)}")

    # 4. Force Reset if failed
    if not is_valid:
        print("🔧 Attempting forceful reset...")
        new_hash = generate_password_hash(password)
        query = "UPDATE users SET password_hash = ? WHERE username = ?"
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query.replace('?', db.placeholder), (new_hash, username))
            conn.commit()
        print("✅ Password reset via diagnostic script.")
        
        # Verify again
        user = auth.get_user_by_username(username)
        if auth.verify_password(user, password):
             print("✅ Verification after reset: SUCCESS")
        else:
             print("❌ Verification after reset: STILL FAILING")

if __name__ == "__main__":
    diagnose_auth("admin", "admin123")
