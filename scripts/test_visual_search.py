import sys
from pathlib import Path
import os
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.services import load_configuration, get_db
from modules.product_manager import ProductManager
from modules.vision_manager import VisionManager, VisionManagerConfig

def test_visual_search():
    print("Initializing services for Visual Search test...")
    config = load_configuration()
    db = get_db()
    
    # Initialize VisionManager explictly
    try:
        vm = VisionManager(config=VisionManagerConfig(enabled=True))
        vm.ensure_loaded()
        print("VisionManager loaded.")
    except Exception as e:
        print(f"Failed to load VisionHelper: {e}")
        return

    pm = ProductManager(db, vision_manager=vm)

    # Test 1: Text-to-Image (Product) Search
    # "I want a blue velvet sofa" -> CLIP Text Embedding -> Match Product CLIP Embedding
    query_text = "un sofá elegante de terciopelo azul"
    print(f"\n--- Test 1: Searching for '{query_text}' (Hybrid/CLIP) ---")
    
    results = pm.search_products(query_text, k=3)
    
    if not results:
        print("No results found.")
    else:
        for p in results:
            print(f"[Score: {p.get('score', 0):.4f}] {p['name']} - {p['category']}")
            
    # Test 2: Verify All Products Embedding Dimensions
    print("\n--- Test 2: Database Check ---")
    with db.get_connection() as conn:
        cursor = db.get_cursor(conn)
        cursor.execute("SELECT sku, name, embedding FROM products")
        rows = cursor.fetchall()
        for row in rows:
            sku = row[1] if isinstance(row, (list, tuple)) else row['sku']
            name = row[2] if isinstance(row, (list, tuple)) else row['name']
            emb_json = row[7] if isinstance(row, (list, tuple)) else (row[2] if isinstance(row, (list, tuple)) else row['embedding'])
            # wait, fix index for fetchall
            sku = row[0]
            name = row[1]
            emb_json = row[2]
            
            if emb_json:
                emb = json.loads(emb_json)
                dim = len(emb)
                print(f"Product: {sku} ({name}) -> Dimension: {dim}")
                if dim != 512:
                    print(f"  [ERROR] {sku} has 384-dim (MiniLM) instead of 512-dim (CLIP).")
            else:
                print(f"Product: {sku} ({name}) -> NO EMBEDDING")


if __name__ == "__main__":
    test_visual_search()
