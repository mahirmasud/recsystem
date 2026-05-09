"""ANN Search - Approximate Nearest Neighbor search for candidate retrieval."""
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
logger = logging.getLogger(__name__)

class ANNSearch:
    """Approximate nearest neighbor search using LSH-like hashing."""
    def __init__(self, config: Dict[str, Any], n_hashes: int = 128):
        self.config = config
        self.n_hashes = n_hashes
        self.hash_tables = []
        self.item_embeddings = None
        self.item_ids = None
        
    def build_index(self, item_embeddings: np.ndarray, item_ids: List[int]):
        """Build hash index for items."""
        logger.info(f"Building ANN index for {len(item_ids)} items")
        self.item_embeddings = item_embeddings
        self.item_ids = np.array(item_ids)
        
        # Create random projection matrices for LSH
        dim = item_embeddings.shape[1]
        self.projection = np.random.randn(dim, self.n_hashes)
        
        # Build hash tables
        self.hash_buckets = {}
        for i, emb in enumerate(item_embeddings):
            hash_code = tuple((emb @ self.projection > 0).astype(int))
            if hash_code not in self.hash_buckets:
                self.hash_buckets[hash_code] = []
            self.hash_buckets[hash_code].append(i)
        
        return self
    
    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[List[int], List[float]]:
        """Find approximate nearest neighbors."""
        if self.item_embeddings is None:
            return [], []
        
        # Get hash code for query
        hash_code = tuple((query_embedding @ self.projection > 0).astype(int))
        
        # Get candidates from same bucket
        candidates = self.hash_buckets.get(hash_code, [])
        
        # If not enough candidates, check neighboring buckets
        if len(candidates) < k:
            for _ in range(5):  # Check up to 5 neighboring buckets
                # Flip one bit to get neighbor
                for i in range(min(3, self.n_hashes)):
                    neighbor_code = list(hash_code)
                    neighbor_code[i] = 1 - neighbor_code[i]
                    neighbor_code = tuple(neighbor_code)
                    candidates.extend(self.hash_buckets.get(neighbor_code, []))
                if len(candidates) >= k:
                    break
        
        if not candidates:
            # Fallback to brute force
            candidates = list(range(len(self.item_embeddings)))
        
        # Compute exact distances for candidates
        candidates = list(set(candidates))[:500]  # Limit for efficiency
        scores = self.item_embeddings[candidates] @ query_embedding.flatten()
        
        # Get top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.item_ids[candidates[i]] for i in top_indices], [float(scores[i]) for i in top_indices]
