import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
except ImportError:
    Pipeline = None

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of the AI classifier."""

    def __init__(self, db_manager, model_path: str):
        self.db = db_manager
        self.model_path = Path(model_path)
        
    def train(self) -> Tuple[bool, str]:
        """Train model using verified documents from DB."""
        if Pipeline is None:
            return False, "Scikit-learn not installed."

        try:
            # Fetch data: Assuming verified docs have correct types
            # Use 'completed' or 'verified' status if available.
            # Here we assume whatever 'doc_type' is in the DB is the ground truth
            # for non-Unknown, non-New types.
            
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT o.text, d.doc_type 
                FROM documents d
                JOIN ocr_texts o ON d.id = o.id_doc
                WHERE o.text IS NOT NULL 
                AND length(o.text) > 20
                AND d.doc_type IS NOT NULL
                AND d.doc_type != 'Unknown'
            """)
            rows = cursor.fetchall()
            
            if len(rows) < 5:
                return False, f"Not enough data to train (found {len(rows)}, need 5+)."

            texts = [r[0] for r in rows]
            labels = [r[1] for r in rows]
            
            # Create Pipeline
            # SGDClassifier is fast and supports incremental learning if needed later
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words=None)),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
            ])
            
            logger.info(f"Training on {len(texts)} documents...")
            pipeline.fit(texts, labels)
            
            # Save
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(pipeline, f)
                
            return True, f"Model trained successfully on {len(texts)} documents."
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False, str(e)
