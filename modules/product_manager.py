import json
import logging
import sqlite3
import numpy as np
import os
from typing import List, Dict, Any, Optional

from modules.db_manager import DBManager
# We will use the same embedding model as RAG
# To avoid cyclical imports or reloading, we might pass the model or load it if needed.
try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError):  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

class ProductManager:
    def __init__(self, db_manager: DBManager, vision_manager=None, model_name: str = "all-MiniLM-L6-v2"):
        self.db = db_manager
        self.logger = logging.getLogger("ProductManager")
        self.vision_manager = vision_manager
        self.model_name = model_name
        self.model = None
        
        # If no vision manager, we fallback to text-only model
        if not self.vision_manager:
            self.logger.info("VisionManager not provided. Using text-only mode.")

    def ensure_model(self):
        if self.vision_manager:
            self.vision_manager.ensure_loaded()
        elif not self.model:
            if SentenceTransformer is None:
                self.logger.warning(
                    "sentence-transformers is not installed; product semantic search is disabled."
                )
                return
            self.logger.info(f"Loading embedding model for Products: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as exc:
                self.logger.error(f"Failed to load product embedding model: {exc}")
                self.model = None

    def _get_embedding(self, text: str = None, image_path: str = None) -> List[float]:
        """Get embedding using VisionManager (CLIP) or SentenceTransformer."""
        if self.vision_manager:
            if image_path:
                return self.vision_manager.embed_image(image_path).tolist()
            elif text:
                return self.vision_manager.embed_text(text).tolist()
        else:
            # Text only fallback
            if text and self.model is not None:
                return self.model.encode([text])[0].tolist()
        return []

    def add_product(self, sku: str, name: str, description: str, price: float, stock: int, image_url: str = "", 
                    attributes: Dict[str, Any] = None, category: str = "General", tags: List[str] = None):
        self.ensure_model()
        
        attributes_json = json.dumps(attributes or {})
        tags_json = json.dumps(tags or [])
        
        # Create rich text for embedding: Name + Description + Attributes + Category
        attr_text = ", ".join([f"{k}: {v}" for k, v in (attributes or {}).items()])
        text_to_embed = f"{name}\nCategory: {category}\n{description}\nAttributes: {attr_text}\nPrecio: {price}€"
        
        # Ideal: Use Image embedding if local path exists, else Text
        # For now, we only have URLs or text. If we had a local image path, we'd use it.
        # Let's check if image_url is actually a local path (common in MVP)
        embedding = []
        if image_url and os.path.exists(image_url):
             embedding = self._get_embedding(image_path=image_url)
        
        if not embedding:
             embedding = self._get_embedding(text=text_to_embed)

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
                        embedding = {self.db.placeholder},
                        attributes = {self.db.placeholder},
                        category = {self.db.placeholder},
                        tags = {self.db.placeholder}
                    WHERE sku = {self.db.placeholder}
                """
                cursor.execute(sql, (name, description, price, stock, image_url, embedding_json, 
                                     attributes_json, category, tags_json, sku))
            else:
                # Insert
                self.logger.info(f"Inserting product {sku}")
                sql = f"""
                    INSERT INTO products (sku, name, description, price, stock, image_url, embedding, attributes, category, tags)
                    VALUES ({self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder}, 
                            {self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder},
                            {self.db.placeholder}, {self.db.placeholder}, {self.db.placeholder})
                """
                cursor.execute(sql, (sku, name, description, price, stock, image_url, embedding_json, 
                                     attributes_json, category, tags_json))
            conn.commit()

    def search_products(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search for products."""
        self.ensure_model()
        
        # If query looks like a file path, treat as image search
        if os.path.exists(query):
             query_vec = np.array(self._get_embedding(image_path=query))
        else:
             query_vec = np.array(self._get_embedding(text=query))
        
        if len(query_vec) == 0:
            return []

        return self.search_by_embedding(query_vec, k)

    def search_by_embedding(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search products using a pre-calculated embedding vector."""
        results = []
        with self.db.get_connection() as conn:
            cursor = self.db.get_cursor(conn)
            cursor.execute("SELECT id, sku, name, description, price, stock, image_url, embedding, attributes, category, tags FROM products")
            rows = cursor.fetchall()
            
            for row in rows:
                p_id = row[0] if isinstance(row, (list, tuple)) else row['id']
                emb_json = row[7] if isinstance(row, (list, tuple)) else row['embedding']
                
                if not emb_json:
                    continue
                    
                prod_vec = np.array(json.loads(emb_json))
                
                # Check dim match (CLIP is 512, MiniLM is 384)
                if query_vec.shape != prod_vec.shape:
                    continue

                # Cosine similarity
                norm_q = np.linalg.norm(query_vec)
                norm_p = np.linalg.norm(prod_vec)
                if norm_q == 0 or norm_p == 0:
                    score = 0
                else:
                    score = np.dot(query_vec, prod_vec) / (norm_q * norm_p)
                
                if score > 0.25: # Slightly lower threshold for CLIP cross-modal
                    results.append({
                        "id": p_id,
                        "sku": row[1] if isinstance(row, (list, tuple)) else row['sku'],
                        "name": row[2] if isinstance(row, (list, tuple)) else row['name'],
                        "description": row[3] if isinstance(row, (list, tuple)) else row['description'],
                        "price": row[4] if isinstance(row, (list, tuple)) else row['price'],
                        "stock": row[5] if isinstance(row, (list, tuple)) else row['stock'],
                        "image_url": row[6] if isinstance(row, (list, tuple)) else row['image_url'],
                        "attributes": json.loads(row[8] if isinstance(row, (list, tuple)) else row['attributes'] or '{}'),
                        "category": row[9] if isinstance(row, (list, tuple)) else row['category'],
                        "tags": json.loads(row[10] if isinstance(row, (list, tuple)) else row['tags'] or '[]'),
                        "score": float(score)
                    })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    def sync_with_erp(self, csv_path: str):
        """Sync price and stock with external CSV."""
        if not os.path.exists(csv_path):
            self.logger.error(f"ERP CSV not found: {csv_path}")
            return False
            
        import csv
        updated_count = 0
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                with self.db.get_connection() as conn:
                    cursor = self.db.get_cursor(conn)
                    for row in reader:
                        sku = row.get('sku')
                        price = row.get('price')
                        stock = row.get('stock')
                        
                        if sku and price is not None and stock is not None:
                            sql = f"UPDATE products SET price = {self.db.placeholder}, stock = {self.db.placeholder} WHERE sku = {self.db.placeholder}"
                            cursor.execute(sql, (float(price), int(stock), sku))
                            if cursor.rowcount > 0:
                                updated_count += 1
                    conn.commit()
            self.logger.info(f"ERP Sync complete. Updated {updated_count} products.")
            return True
        except Exception as e:
            self.logger.error(f"ERP Sync failed: {e}")
            return False

    def get_product_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """Fetch a single product by SKU."""
        with self.db.get_connection() as conn:
            cursor = self.db.get_cursor(conn)
            cursor.execute(f"SELECT sku, name, description, price, stock, image_url, attributes, category, tags FROM products WHERE sku = {self.db.placeholder}", (sku,))
            row = cursor.fetchone()
            if row:
                return {
                    "sku": row[0] if isinstance(row, (list, tuple)) else row['sku'],
                    "name": row[1] if isinstance(row, (list, tuple)) else row['name'],
                    "description": row[2] if isinstance(row, (list, tuple)) else row['description'],
                    "price": row[3] if isinstance(row, (list, tuple)) else row['price'],
                    "stock": row[4] if isinstance(row, (list, tuple)) else row['stock'],
                    "image_url": row[5] if isinstance(row, (list, tuple)) else row['image_url'],
                    "attributes": json.loads(row[6] if isinstance(row, (list, tuple)) else row['attributes'] or '{}'),
                    "category": row[7] if isinstance(row, (list, tuple)) else row['category'],
                    "tags": json.loads(row[8] if isinstance(row, (list, tuple)) else row['tags'] or '[]')
                }
        return None
