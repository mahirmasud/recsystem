"""Embedding Retrieval - Retrieves candidates using embedding similarity."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

class EmbeddingRetrieval:
    """Retrieves candidates based on embedding similarity."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.item_embeddings = {}
        self.category_embeddings = {}
        
    def load_item_embeddings(self, embeddings: Dict[int, np.ndarray]):
        """Load pre-computed item embeddings."""
        self.item_embeddings = embeddings
        logger.info(f"Loaded {len(embeddings)} item embeddings")
        
    def load_category_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Load category-level embeddings."""
        self.category_embeddings = embeddings
        
    def retrieve_by_similarity(self, user_embedding: np.ndarray, 
                                k: int = 100,
                                category_filter: Optional[str] = None) -> List[int]:
        """Retrieve top-k items by embedding similarity."""
        if not self.item_embeddings:
            return []
        
        scores = {}
        for item_id, item_emb in self.item_embeddings.items():
            score = float(np.dot(user_embedding.flatten(), item_emb.flatten()))
            scores[item_id] = score
        
        # Sort and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:k]]
    
    def retrieve_by_category(self, category: str, k: int = 50) -> List[int]:
        """Retrieve items from a specific category."""
        if category not in self.category_embeddings:
            return []
        # Return items whose embeddings are close to category embedding
        cat_emb = self.category_embeddings[category]
        return self.retrieve_by_similarity(cat_emb, k=k)
