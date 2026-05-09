"""
Precision@K Metric - Evaluation metric for recommendation precision

Calculates precision at K positions in ranked recommendations.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PrecisionAtK:
    """
    Calculate Precision@K for recommendation evaluation.
    
    Precision@K = (relevant items in top K) / K
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Precision@K metric.
        
        Args:
            k: Number of top items to consider
        """
        self.k = k
        
    def compute(
        self,
        recommendations: List[int],
        relevant_items: List[int]
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs (ranked)
            relevant_items: List of relevant item IDs (ground truth)
            
        Returns:
            Precision@K score
        """
        if len(recommendations) == 0 or len(relevant_items) == 0:
            return 0.0
        
        # Get top K recommendations
        top_k = recommendations[:self.k]
        
        # Count relevant items in top K
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        
        precision = relevant_in_top_k / min(len(top_k), self.k)
        
        return precision
    
    def compute_batch(
        self,
        all_recommendations: List[List[int]],
        all_relevant: List[List[int]]
    ) -> Dict[str, float]:
        """
        Calculate Precision@K for multiple users.
        
        Args:
            all_recommendations: List of recommendation lists
            all_relevant: List of relevant item lists
            
        Returns:
            Dictionary with mean and individual scores
        """
        if len(all_recommendations) != len(all_relevant):
            raise ValueError("Number of recommendations must match number of relevant lists")
        
        scores = []
        for recs, relevant in zip(all_recommendations, all_relevant):
            score = self.compute(recs, relevant)
            scores.append(score)
        
        return {
            'mean_precision': np.mean(scores),
            'std_precision': np.std(scores),
            'min_precision': np.min(scores),
            'max_precision': np.max(scores),
            'individual_scores': scores
        }
    
    def compute_from_scores(
        self,
        items: List[int],
        scores: np.ndarray,
        relevant_items: List[int]
    ) -> float:
        """
        Calculate Precision@K from items and their scores.
        
        Args:
            items: List of all item IDs
            scores: Array of scores for each item
            relevant_items: List of relevant item IDs
            
        Returns:
            Precision@K score
        """
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        sorted_items = [items[i] for i in sorted_indices]
        
        return self.compute(sorted_items, relevant_items)
