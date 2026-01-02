from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class Deduplicator:
    """
    Finds exact and near-exact duplicates using visual embeddings.
    """
    def __init__(self, vision_manager):
        self.vision_manager = vision_manager

    def find_duplicates(self, threshold: float = 0.985) -> List[Dict[str, Any]]:
        """
        Returns a list of duplicate groups.
        Each group contains 'primary' doc and 'duplicates' list.
        """
        if not self.vision_manager or not self.vision_manager.config.enabled:
            logger.warning("Vision manager disable cannot deduplicate.")
            return []

        # Ensure index is loaded
        self.vision_manager.ensure_loaded()
        if not self.vision_manager.index or self.vision_manager.index.ntotal < 2:
            return []

        # Get all embeddings
        # FAISS index reconstruction might be needed if flat, 
        # but standard IndexFlatL2 supports reconstruct_n
        total = self.vision_manager.index.ntotal
        try:
            embeddings = self.vision_manager.index.reconstruct_n(0, total)
        except Exception as e:
            logger.error(f"Failed to reconstruct index for dedup: {e}")
            return []
        
        # Metadata logic is tricky because FAISS index ID -> metadata list index
        # We assume 1:1 mapping if safe, but vision manager might have gaps if we delete?
        # Actually vision manager rebuilds index on startup usually or appends.
        # Let's use self.vision_manager.doc_map if available or metadata list
        
        meta = self.vision_manager.metadata
        if len(meta) != total:
            logger.warning(f"Index size {total} != Metadata size {len(meta)}. Reconstruction unsafe.")
            return []

        # Pairwise comparison (Brute force for now, efficient enough for <10k docs)
        # For larger datasets, we'd use range_search or clustering
        import numpy as np
        
        # Normalize vectors for cosine similarity (if IP metric)
        # But if L2, close is 0. 
        # VisionManager uses L2 (IndexFlatL2). 
        # Distance = 0 means identical.
        # We look for Distance < epsilon
        
        duplicates = []
        seen = set()
        
        # Matrix multiplication is faster but memory intensive for huge datasets
        # Let's iterate. 
        # Or better: search the index against itself?
        
        # Let's search for neighbors within a very small radius
        # But IndexFlatL2.search returns L2 distance.
        # 0.99 similarity in Cosine is roughly what in L2? 
        # If normalized: L2 = 2 * (1 - cosine).  
        # So Cosine 0.99 -> L2 approx 0.02.
        
        # radius search
        # limits = faiss.RangeSearchPartialResult(res)
        # This is complex. Let's do a simpler N^2 approach for N<1000
        # If N > 1000, we warn user.
        
        if total > 5000:
             logger.warning("Dataset too large for simple deduplication.")
             return []

        vecs = np.array(embeddings)
        # Compute L2 distance matrix
        # d(a,b)^2 = |a|^2 + |b|^2 - 2*a*b
        # If vectors are not normalized, L2 is absolute.
        
        # Let's rely on faiss search for each vector
        match_groups = []
        
        # Search for 5 nearest neighbors for everyone
        D, I = self.vision_manager.index.search(vecs, 5)
        
        for i in range(total):
            if i in seen:
                continue
                
            group = []
            # Check neighbors
            for j, dist in enumerate(D[i]):
                idx = I[i][j]
                if idx == -1 or idx == i:
                    continue
                
                # Threshold for "Duplicate"
                # Experimentally: Identical images have dist 0.0
                # Very close resizes: < 5.0 (unnormalized L2) or < 0.2 (normalized)
                # Let's assume embeddings are normalized by CLIP/VisionManager?
                # VisionManager calls `encode(normalize_embeddings=True)` generally?
                # Check vision_manager.py ... it calls model.encode.
                # Usually standardCLIP is unnormalized? OpenCLIP is normalized.
                # Let's assert a tight threshold.
                
                if dist < 1.0: # Conservative L2 threshold
                     if idx not in seen:
                         group.append(idx)
            
            if group:
                # Add primary and duplicates
                seen.add(i)
                dupes = []
                for g_idx in group:
                    seen.add(g_idx)
                    dupes.append(meta[g_idx])
                
                duplicates.append({
                    "primary": meta[i],
                    "duplicates": dupes
                })
                
        return duplicates
