import psycopg2
import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
db_conf = config.get("database", {}).get("postgresql", {})

try:
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=db_conf.get("host", "localhost"),
        port=db_conf.get("port", 5432),
        user=db_conf.get("user", "postgres"),
        password=db_conf.get("password", ""),
        dbname=db_conf.get("dbname", "autocr")
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("Checking PostgreSQL version...")
    cursor.execute("SELECT version();")
    print(f"Version: {cursor.fetchone()[0]}")

    print("Attempting to create extension 'vector'...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("SUCCESS: Extension 'vector' created/verified.")
    
except Exception as e:
    print(f"FAILURE: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
