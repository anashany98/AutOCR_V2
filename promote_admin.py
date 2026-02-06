import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.db_manager import DBManager

db = DBManager(config={"database": {"type": "sqlite", "path": "data/digitalizerai.db"}})

with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
    conn.commit()
    print("✅ User 'admin' promoted to role 'admin'!")
