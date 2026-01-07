from modules.db_manager import DBManager
from modules.rag_manager import RAGManager
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Rebuilding RAG Index (Hybrid Mode)...")
    
    # Load config and DB
    # We must replicate the load_configuration logic if it's not importable or use the one I fixed
    # Actually I should verify if I can import load_configuration from modules.app or where it is
    # In web_app/app.py there is load_configuration. In db_manager.py there isn't?
    # Let's check modules/db_manager.py imports again. 
    # It does NOT export load_configuration.
    # So I will use my robust load_config function here too.
    
    import yaml
    def load_config():
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config = load_config()
    PROJECT_ROOT = Path(".").absolute()
    
    # Init DB
    print("Initializing Database...")
    db = DBManager(config)
    
    # Init RAG
    print("Initializing RAG Manager...")
    rag_dir = PROJECT_ROOT / "data" / "rag_index"
    rag = RAGManager(str(rag_dir))
    
    # Rebuild
    print("Starting indexing...")
    rag.rebuild(db)
    print("Rebuild complete. Saving index...")
    rag.save_index()
    print("Done!")

if __name__ == "__main__":
    main()
