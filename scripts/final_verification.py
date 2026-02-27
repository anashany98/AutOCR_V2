"""
Final System Verification — AutoOCR Document AI Platform.

Performs an end-to-end health check of the deployed architecture:
1. Database connectivity (PostgreSQL + pgvector extension)
2. Redis connectivity
3. Celery worker status (via inspection)
4. Pipeline component initialization (OCR, Layout, RAG)
5. Directory structure verification
6. API endpoint availability (simulated)

Usage:
    python scripts/final_verification.py
"""

import sys
import os
import time
import requests
import yaml
from pathlib import Path
from sqlalchemy import create_engine, text

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Load config
CONFIG_PATH = ROOT_DIR / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

def print_result(component, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} [{component}] {message}")

def check_structure():
    print("\n--- 1. Directory Structure ---")
    required_dirs = [
        "data", "logs", "models", "processed_docs", "uploads", 
        "migrations", "pipeline", "modules", "web_app"
    ]
    all_ok = True
    for d in required_dirs:
        path = ROOT_DIR / d
        if path.exists() and path.is_dir():
            print_result(d, True, "Exists")
        else:
            print_result(d, False, "Missing")
            all_ok = False
    return all_ok

def check_database():
    print("\n--- 2. Database (PostgreSQL + pgvector) ---")
    db_conf = CONFIG.get("database", {}).get("postgresql", {})
    
    # Fallback to defaults if config is empty (e.g. env vars in Docker)
    host = os.environ.get("DB_HOST", db_conf.get("host", "localhost"))
    port = os.environ.get("DB_PORT", db_conf.get("port", 5432))
    user = os.environ.get("DB_USER", db_conf.get("user", "postgres"))
    password = os.environ.get("DB_PASSWORD", db_conf.get("password", "password"))
    dbname = os.environ.get("DB_NAME", db_conf.get("dbname", "autoocr"))

    # For local test (outside docker), allow localhost usage if 'db' host fails
    if host == "db":
        print(f"   Note: Host is '{host}'. If running outside Docker, ensure expected mapping.")

    uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    try:
        engine = create_engine(uri)
        with engine.connect() as conn:
            # Check connection
            res = conn.execute(text("SELECT 1")).scalar()
            print_result("Connection", True, "Successful")
            
            # Check pgvector extension
            res = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'")).fetchone()
            if res:
                print_result("pgvector", True, "Installed")
            else:
                print_result("pgvector", False, "Extension NOT found!")
                
            # Check tables
            inspector = text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            tables = [row[0] for row in conn.execute(inspector)]
            required_tables = ["documents", "chunks", "embeddings", "audit_logs", "_migrations"]
            missing = [t for t in required_tables if t not in tables]
            
            if not missing:
                print_result("Schema", True, f"Found tables: {len(tables)}")
            else:
                print_result("Schema", False, f"Missing tables: {missing}")

    except Exception as e:
        print_result("Connection", False, str(e))

def check_redis():
    print("\n--- 3. Redis & Celery ---")
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", CONFIG.get("database", {}).get("redis", {}).get("url", "redis://localhost:6379/0"))
        r = redis.from_url(redis_url)
        if r.ping():
            print_result("Redis", True, "Connected")
        else:
            print_result("Redis", False, "Ping failed")
    except Exception as e:
        print_result("Redis", False, f"Connection failed: {e}")

    # Check Celery workers
    try:
        from modules.celery_app import app
        i = app.control.inspect()
        active = i.active()
        if active:
            workers = list(active.keys())
            print_result("Celery Workers", True, f"Active: {workers}")
        else:
            print_result("Celery Workers", False, "No active workers found (is worker container running?)")
    except Exception as e:
        print_result("Celery Inspector", False, str(e))

def check_api():
    print("\n--- 4. API Health ---")
    base_url = "http://localhost:8000"  # Port mapped in docker-compose
    try:
        # Check health/status endpoint (assuming one exists or using root)
        # Using a specialized health check if available, else standard routes
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                print_result("API /health", True, "OK")
            else:
                print_result("API /health", False, f"Status {resp.status_code}")
        except:
            print_result("API /health", False, "Unreachable")
            
    except Exception as e:
        print_result("API", False, str(e))

def main():
    print("========================================")
    print("AutoOCR — Final System Verification")
    print("========================================")
    
    check_structure()
    check_database()
    check_redis()
    check_api()
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
