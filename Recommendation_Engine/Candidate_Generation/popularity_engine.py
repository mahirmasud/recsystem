"""Popularity Engine - Popularity-based candidate generation."""
import numpy as np
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class PopularityEngine:
    """Generates candidates based on item popularity."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.popularity_scores = {}
        self.trending_items = []
        
    def fit(self, item_interactions: Dict[int, int], time_decay: float = 0.99):
        """Compute popularity scores with time decay."""
        logger.info("Computing popularity scores")
        total = sum(item_interactions.values()) + 1e-8
        for item_id, count in item_interactions.items():
            self.popularity_scores[item_id] = count / total
        
        # Sort by popularity
        sorted_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        self.trending_items = [item_id for item_id, _ in sorted_items]
        return self
    
    def get_candidates(self, n: int = 100, exclude: List[int] = None) -> List[int]:
        """Get top-n popular items."""
        exclude = exclude or []
        candidates = [item for item in self.trending_items if item not in exclude][:n]
        return candidates
    
    def get_score(self, item_id: int) -> float:
        """Get popularity score for an item."""
        return self.popularity_scores.get(item_id, 0.0)
