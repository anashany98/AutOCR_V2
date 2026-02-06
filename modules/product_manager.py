import json
import logging
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

from modules.db_manager import DBManager
# We will use the same embedding model as RAG
# To avoid cyclical imports or reloading, we might pass the model or load it if needed.
from sentence_transformers import SentenceTransformer

class ProductManager:
    def __init__(self, db_manager: DBManager, model_name: str = "all-MiniLM-L6-v2"):
        self.db = db_manager
        self.logger = logging.getLogger("ProductManager")
        self.model_name = model_name
        self.model = None
        
    def ensure_model(self):
        if not self.model:
            self.logger.info(f"Loading embedding model for Products: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def add_product(self, sku: str, name: str, description: str, price: float, stock: int, image_url: str = ""):
        self.ensure_model()
        
        # Create rich text for embedding
        # We want to find products based on their description and name
        text_to_embed = f"{name}\n{description}\nPrecio: {price}€"
        embedding = self.model.encode([text_to_embed])[0].tolist()
        embedding_json = json.dumps(embedding)
        
        with self.db.get_connection() as conn:
            cursor = self.db.get_cursor(conn)
            
            # Check if exists to update or insert
            cursor.execute(f"SELECT id FROM products WHERE sku = {self.db.placeholder}", (sku,))
            row = cursor.fetchone()
            
            if row:
                # Update
                self.logger.info(f"Updating product {sku}")
                sql = f"""
                    UPDATE products SET 
                        name = {self.db.placeholder}, 
                        description = {self.db.placeholder}, 
                        price = {self.db.placeholder}, 
                        stock = {self.db.placeholder}, 
                        image_url = {self.db.placeholder}, 
                        embedding = {self.db.placeholder}
                    WHERE sku = {self.db.placeholder}
                """
                cursor.execute(sql, (name, description, price, stock, image_url, embedding_json, sku))
            else:
                # Insert
                self.logger.info(f"Inserting product {sku}")
                sql = f"""
                    INSERT INTO products (sku, name, description, price, stock, image_url, embedding)
                    VALUES ({self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder}, 
                            {self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder})
                """
                cursor.execute(sql, (sku, name, description, price, stock, image_url, embedding_json))
            conn.commit()

    def search_products(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search for products."""
        self.ensure_model()
        query_vec = self.model.encode([query])[0]
        
        results = []
        with self.db.get_connection() as conn:
            cursor = self.db.get_cursor(conn)
            cursor.execute("SELECT id, sku, name, description, price, stock, image_url, embedding FROM products")
            rows = cursor.fetchall()
            
            for row in rows:
                p_id = row[0] if isinstance(row, (list, tuple)) else row['id']
                # sku = row[1]
                # name = row[2]
                emb_json = row[7] if isinstance(row, (list, tuple)) else row['embedding']
                
                if not emb_json:
                    continue
                    
                prod_vec = np.array(json.loads(emb_json))
                
                # Cosine similarity
                norm_q = np.linalg.norm(query_vec)
                norm_p = np.linalg.norm(prod_vec)
                if norm_q == 0 or norm_p == 0:
                    score = 0
                else:
                    score = np.dot(query_vec, prod_vec) / (norm_q * norm_p)
                
                if score > 0.3: # Minimum threshold
                    # unpack properly
                    results.append({
                        "id": p_id,
                        "sku": row[1] if isinstance(row, (list, tuple)) else row['sku'],
                        "name": row[2] if isinstance(row, (list, tuple)) else row['name'],
                        "description": row[3] if isinstance(row, (list, tuple)) else row['description'],
                        "price": row[4] if isinstance(row, (list, tuple)) else row['price'],
                        "stock": row[5] if isinstance(row, (list, tuple)) else row['stock'],
                        "image_url": row[6] if isinstance(row, (list, tuple)) else row['image_url'],
                        "score": float(score)
                    })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
