import psycopg2
import yaml
import os

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def diag_failures():
    config = load_config()
    p_conf = config.get("database", {}).get("postgresql", {})
    try:
        conn = psycopg2.connect(
            host=p_conf.get("host", "localhost"),
            port=p_conf.get("port", 5432),
            user=p_conf.get("user", "postgres"),
            password=p_conf.get("password", "123"),
            dbname=p_conf.get("dbname", "autocr")
        )
        cur = conn.cursor()
        
        print("--- FAILURE REASONS ---")
        cur.execute("SELECT error_message, count(*) FROM documents WHERE status = 'FAILED' GROUP BY error_message")
        for row in cur.fetchall():
            print(f"Error: {row[0]} | Count: {row[1]}")
            
        print("\n--- SAMPLE FAILED PATHS AND EXISTENCE ---")
        cur.execute("SELECT id, filename, path FROM documents WHERE status = 'FAILED' LIMIT 10")
        for row in cur.fetchall():
            exists = os.path.exists(row[2])
            print(f"ID: {row[0]} | File: {row[1]} | Path: {row[2]} | Exists: {exists}")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diag_failures()
