from modules.db_manager import DBManager
import yaml
from pathlib import Path

print("Checking Document Embeddings...")
config_path = Path("config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db = DBManager(config)
try:
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Check for document_embeddings
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'document_embeddings')")
            if cur.fetchone()[0]:
                cur.execute("SELECT COUNT(*) FROM document_embeddings")
                emb_count = cur.fetchone()[0]
                print(f"Total Embeddings: {emb_count}")
                if emb_count > 0:
                    cur.execute("SELECT doc_id, chunk_text FROM document_embeddings LIMIT 1")
                    row = cur.fetchone()
                    print(f"Sample: {row}")
            else:
                print("Table 'document_embeddings' does NOT exist.")

except Exception as e:
    print(f"Error checking DB: {e}")
