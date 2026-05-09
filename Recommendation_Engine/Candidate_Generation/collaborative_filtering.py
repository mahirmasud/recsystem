"""Collaborative Filtering - User-based and Item-based CF."""
import numpy as np
from typing import Dict, Any, Optional, List
import logging
logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """Memory-based collaborative filtering."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_similarity = None
        self.item_similarity = None
        self.user_item_matrix = None
        
    def fit(self, interactions: Dict[int, List[int]], n_users: int, n_items: int):
        """Build user-item interaction matrix."""
        logger.info("Building collaborative filtering model")
        self.user_item_matrix = np.zeros((n_users, n_items))
        for user_id, items in interactions.items():
            if user_id < n_users:
                for item_id in items:
                    if item_id < n_items:
                        self.user_item_matrix[user_id, item_id] = 1
        # Compute similarities
        self.user_similarity = self._cosine_similarity(self.user_item_matrix)
        self.item_similarity = self._cosine_similarity(self.user_item_matrix.T)
        return self
    
    def _cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        normalized = matrix / norm
        return np.dot(normalized, normalized.T)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Get recommendations for a user."""
        if self.user_item_matrix is None:
            return []
        user_vec = self.user_item_matrix[user_id]
        scores = np.dot(self.user_similarity[user_id], self.user_item_matrix)
        scores[user_vec > 0] = 0  # Exclude already interacted
        return list(np.argsort(scores)[-n_recommendations:][::-1])
