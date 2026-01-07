import psycopg2
import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def diag():
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
        
        print("--- STATUS SUMMARY ---")
        cur.execute("SELECT status, count(*) FROM documents GROUP BY status")
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")
            
        print("\n--- SAMPLE OF DOCUMENTS WITHOUT OCR ---")
        cur.execute("""
            SELECT d.id, d.filename, d.status 
            FROM documents d 
            LEFT JOIN ocr_texts o ON d.id = o.id_doc 
            WHERE o.id_doc IS NULL 
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"ID: {row[0]} | File: {row[1]} | Status: {row[2]}")
            
        print("\n--- SAMPLE OF OCR TEXT FOR 'ZAFIRO' DOCS ---")
        cur.execute("""
            SELECT d.filename, o.text 
            FROM documents d 
            JOIN ocr_texts o ON d.id = o.id_doc 
            WHERE d.filename ILIKE '%Zafiro%' OR o.text ILIKE '%Zafiro%'
        """)
        for row in cur.fetchall():
            print(f"File: {row[0]}")
            print(f"Text Snippet: {row[1][:200]}...")
            print("-" * 20)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diag()
