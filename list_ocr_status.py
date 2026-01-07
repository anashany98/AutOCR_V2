import psycopg2
import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_ocr_status():
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
        
        print("--- FILES WITH OCR ---")
        cur.execute("SELECT d.filename FROM documents d JOIN ocr_texts o ON d.id = o.id_doc")
        rows = cur.fetchall()
        for r in rows:
            print(f"- {r[0]}")
            
        print("\n--- FILES WITHOUT OCR (Sample 10) ---")
        cur.execute("SELECT d.filename FROM documents d LEFT JOIN ocr_texts o ON d.id = o.id_doc WHERE o.id_doc IS NULL LIMIT 10")
        rows = cur.fetchall()
        for r in rows:
            print(f"- {r[0]}")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_ocr_status()
