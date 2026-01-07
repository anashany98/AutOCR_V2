from modules.db_manager import DBManager
import yaml
from pathlib import Path

print("Checking RAG Status...")
config_path = Path("config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db = DBManager(config)
try:
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            print(f"Total Documents in DB: {doc_count}")
            
            # Check if table exists first (some schemas differ)
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'document_chunks')")
            if cur.fetchone()[0]:
                cur.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = cur.fetchone()[0]
                print(f"Total Text Chunks in DB: {chunk_count}")
            else:
                print("Table 'document_chunks' does NOT exist.")

    index_path = Path("data/rag_index.faiss")
    if index_path.exists():
        print(f"FAISS Index exists. Size: {index_path.stat().st_size} bytes")
    else:
        print("FAISS Index NOT FOUND.")

except Exception as e:
    print(f"Error checking DB: {e}")
