#!/usr/bin/env python3
"""
Script to create an admin user for AutoOCR.
Run: python create_admin.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.db_manager import DBManager
from modules.auth_manager import AuthManager

def main():
    # Initialize DB
    db = DBManager(config={"database": {"type": "sqlite", "path": "data/digitalizerai.db"}})
    auth = AuthManager(db)
    
    # Admin credentials
    username = "admin"
    password = "admin123"  # Change this or prompt user
    
    # Check if exists
    existing = auth.get_user_by_username(username)
    if existing:
        print(f"⚠️ User '{username}' already exists with role: {existing.role}")
        return
    
    # Create admin user
    success, msg = auth.create_user(username, password, role='admin')
    
    if success:
        print(f"✅ Admin user created successfully!")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        print(f"   Role: admin")
    else:
        print(f"❌ Failed to create user: {msg}")

if __name__ == "__main__":
    main()
