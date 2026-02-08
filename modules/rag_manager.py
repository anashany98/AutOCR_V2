import os
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError):  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
logger = logging.getLogger(__name__)

class RAGManager:
    """Manages semantic search/RAG functionality."""

    def __init__(self, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "text_index.faiss"
        self.meta_path = self.index_dir / "text_meta.pkl"
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = [] # Maps index ID to doc info
        
        self.db_manager = None # Optional reference
        self.ensure_loaded()

    def set_db_manager(self, db_manager):
        self.db_manager = db_manager

    def ensure_loaded(self):
        """Lazy load model and index."""
        if not SentenceTransformer or not faiss:
            logger.warning("RAG requires sentence-transformers and faiss.")
            return

        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

        if self.index is None:
            if self.index_path.exists() and self.meta_path.exists():
                logger.info("Loading existing RAG index...")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    with open(self.meta_path, "rb") as f:
                        self.metadata = pickle.load(f)
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. creating new.")
                    self._create_new_index()
            else:
                self._create_new_index()

    def _create_new_index(self):
        if self.model is None: return
        dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def save_index(self):
        """Persist index to disk."""
        if not self.index_dir.exists():
            os.makedirs(self.index_dir, exist_ok=True)
            
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add_document(self, doc_id: int, filename: str, text: str, db_manager=None, owner_id: Optional[int] = None, hotel_id: Optional[int] = None):
        """Chunk and index a document."""
        self.ensure_loaded()
        if not self.model:
            return

        # Use db_manager from argument or self
        db = db_manager or self.db_manager

        # Improved chunking
        raw_chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 20]
        chunks = []
        max_chunk_chars = 1500
        
        for rc in raw_chunks:
            if len(rc) <= max_chunk_chars:
                chunks.append(rc)
            else:
                for i in range(0, len(rc), max_chunk_chars):
                    chunks.append(rc[i:i+max_chunk_chars])
        
        if not chunks and text:
             chunks = [text[:max_chunk_chars]]
        if not chunks: return

        embeddings = self.model.encode(chunks)
        
        # Check for pgvector support
        use_pgvector = False
        if db:
            pg_conf = db.config.get("postgresql", {})
            use_pgvector = pg_conf.get("use_pgvector", False)

        if db and db.engine_type == "postgresql" and use_pgvector:
            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                for i, chunk in enumerate(chunks):
                    emb = list(embeddings[i].astype('float32'))
                    cursor.execute(
                        "INSERT INTO document_embeddings (doc_id, embedding, chunk_text) VALUES (%s, %s, %s)",
                        (doc_id, emb, chunk)
                    )
                conn.commit()
        else:
            if not self.index: return
            self.index.add(np.array(embeddings).astype('float32'))
            for chunk in chunks:
                self.metadata.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "owner_id": owner_id,
                    "hotel_id": hotel_id,
                    "text": chunk
                })

    def search(self, query: str, k: int = 5, db_manager=None, owner_id: Optional[int] = None, hotel_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks."""
        self.ensure_loaded()
        if not self.model:
            return []

        db = db_manager or self.db_manager
        
        # Check for pgvector support
        use_pgvector = False
        if db:
            pg_conf = db.config.get("postgresql", {})
            use_pgvector = pg_conf.get("use_pgvector", False)

        if db and db.engine_type == "postgresql" and use_pgvector:
            vec = list(self.model.encode([query])[0].astype('float32'))
            with db.get_connection() as conn:
                cursor = db.get_cursor(conn)
                where_clauses = []
                params = [vec]
                
                if owner_id is not None:
                    where_clauses.append("d.owner_id = %s")
                    params.append(owner_id)
                if hotel_id is not None:
                    where_clauses.append("d.hotel_id = %s")
                    params.append(hotel_id)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                params.append(k)
                
                query_sql = f"""
                    SELECT e.doc_id, e.chunk_text as text, e.embedding <-> %s as score 
                    FROM document_embeddings e
                    JOIN documents d ON e.doc_id = d.id
                    {where_sql}
                    ORDER BY score LIMIT %s
                """
                cursor.execute(query_sql, tuple(params))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({"doc_id": row[0], "text": row[1], "score": float(row[2])})
                return results
        else:
            if not self.index or self.index.ntotal == 0:
                return []
            vec = self.model.encode([query])
            
            # Fetch more candidates to allow for filtering
            search_k = k * 20 if (owner_id is not None or hotel_id is not None) else k
            D, I = self.index.search(np.array(vec).astype('float32'), search_k)
            
            results = []
            count = 0
            for i, idx in enumerate(I[0]):
                if idx == -1 or idx >= len(self.metadata): continue
                item = self.metadata[idx].copy()
                
                # Isolation filtering
                if owner_id is not None:
                    doc_owner = item.get("owner_id")
                    if doc_owner is not None and int(doc_owner) != int(owner_id): continue
                
                if hotel_id is not None:
                    doc_hotel = item.get("hotel_id")
                    if doc_hotel is not None and int(doc_hotel) != int(hotel_id): continue
                
                item["score"] = float(D[0][i])
                results.append(item)
                count += 1
                if count >= k: break
                
            return results

    def rebuild(self, db_manager):
        """Re-index everything from DB."""
        self._create_new_index() # Reset
        
        # Need to fetch all docs
        # Need to fetch all docs
        cursor = db_manager.execute("""
            SELECT d.id, d.filename, o.text, d.owner_id, d.hotel_id
            FROM documents d 
            JOIN ocr_texts o ON d.id = o.id_doc 
            WHERE o.text IS NOT NULL
        """)
        rows = cursor.fetchall()
        
        logger.info(f"Rebuilding RAG index for {len(rows)} documents...")
        for row in rows:
            id_val = row[0] if isinstance(row, (tuple, list)) else row['id']
            fname_val = row[1] if isinstance(row, (tuple, list)) else row['filename']
            text_val = row[2] if isinstance(row, (tuple, list)) else row['text']
            owner_val = row[3] if isinstance(row, (tuple, list)) else row.get('owner_id')
            hotel_val = row[4] if isinstance(row, (tuple, list)) else row.get('hotel_id')
            self.add_document(id_val, fname_val, text_val, db_manager, owner_id=owner_val, hotel_id=hotel_val)
            
        if db_manager.engine_type == "sqlite":
            self.save_index()
        logger.info("RAG index rebuild complete.")
