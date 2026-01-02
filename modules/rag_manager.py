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
except ImportError:
    SentenceTransformer = None

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
        
        self.ensure_loaded()

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

    def add_document(self, doc_id: int, filename: str, text: str):
        """Chunk and index a document."""
        self.ensure_loaded()
        if not self.model or not self.index:
            return

        # Simple chunking by paragraphs or lines
        # For better RAG, we might want overlapping chunks, but let's keep it simple
        chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
        if not chunks:
             # Fallback to single chunk if no paragraphs
             chunks = [text[:1000]] # Limit size
        
        if not chunks: 
            return

        embeddings = self.model.encode(chunks)
        self.index.add(np.array(embeddings).astype('float32'))
        
        for chunk in chunks:
            self.metadata.append({
                "doc_id": doc_id,
                "filename": filename,
                "text": chunk
            })
        
        # Auto-save occasionally? For now, manual save calls or save on every add (slow)
        self.save_index()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks."""
        self.ensure_loaded()
        if not self.model or not self.index or self.index.ntotal == 0:
            return []

        vec = self.model.encode([query])
        D, I = self.index.search(np.array(vec).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(D[0][i])
            results.append(item)
            
        return results

    def rebuild(self, db_manager):
        """Re-index everything from DB."""
        self._create_new_index() # Reset
        
        # Need to fetch all docs
        # This is heavy for large DBs, but fine for MVP
        cursor = db_manager.conn.cursor()
        cursor.execute("""
            SELECT d.id, d.filename, o.text 
            FROM documents d 
            JOIN ocr_texts o ON d.id = o.id_doc 
            WHERE o.text IS NOT NULL
        """)
        rows = cursor.fetchall()
        
        logger.info(f"Rebuilding RAG index for {len(rows)} documents...")
        for row in rows:
            self.add_document(row[0], row[1], row[2])
            
        self.save_index()
        logger.info("RAG index rebuild complete.")
