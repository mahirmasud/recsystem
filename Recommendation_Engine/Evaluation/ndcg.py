"""
NDCG (Normalized Discounted Cumulative Gain) Metric

Calculates NDCG for evaluating ranking quality with graded relevance.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class NDCGMetric:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG accounts for both relevance grades and position in ranking.
    Supports binary and graded relevance.
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize NDCG metric.
        
        Args:
            k: Number of top items to consider
        """
        self.k = k
        
    def _dcg(self, gains: np.ndarray) -> float:
        """
        Calculate Discounted Cumulative Gain.
        
        DCG = sum(gain_i / log2(i + 1)) for i in 1..n
        
        Args:
            gains: Array of relevance gains at each position
            
        Returns:
            DCG score
        """
        if len(gains) == 0:
            return 0.0
        
        # Create discount factors: 1/log2(i+2) for positions 0, 1, 2...
        discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
        
        return np.sum(gains * discounts)
    
    def _ndcg(
        self,
        predicted_gains: np.ndarray,
        ideal_gains: np.ndarray
    ) -> float:
        """
        Calculate Normalized DCG.
        
        NDCG = DCG / IDCG (Ideal DCG)
        
        Args:
            predicted_gains: Gains from predicted ranking
            ideal_gains: Gains from ideal ranking (sorted descending)
            
        Returns:
            NDCG score between 0 and 1
        """
        dcg = self._dcg(predicted_gains)
        idcg = self._dcg(ideal_gains)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def compute(
        self,
        recommendations: List[int],
        relevant_items: Union[List[int], Dict[int, float]],
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs (ranked)
            relevant_items: Either list of relevant IDs or dict of {item_id: relevance}
            relevance_scores: Optional dict of {item_id: relevance} if relevant_items is a list
            
        Returns:
            NDCG@K score
        """
        if len(recommendations) == 0:
            return 0.0
        
        # Build relevance mapping
        if isinstance(relevant_items, dict):
            relevance_map = relevant_items
        elif relevance_scores is not None:
            relevance_map = relevance_scores
        else:
            # Binary relevance
            relevance_map = {item_id: 1.0 for item_id in relevant_items}
        
        # Get top K recommendations
        top_k_recs = recommendations[:self.k]
        
        # Get gains for predicted ranking
        predicted_gains = np.array([relevance_map.get(item_id, 0.0) for item_id in top_k_recs])
        
        # Get ideal gains (sorted descending)
        all_gains = sorted(relevance_map.values(), reverse=True)[:self.k]
        ideal_gains = np.array(all_gains + [0.0] * (len(predicted_gains) - len(all_gains)))
        
        return self._ndcg(predicted_gains, ideal_gains)
    
    def compute_from_scores(
        self,
        items: List[int],
        predicted_scores: np.ndarray,
        true_relevance: Dict[int, float]
    ) -> float:
        """
        Calculate NDCG@K from predicted scores and true relevance.
        
        Args:
            items: List of all item IDs
            predicted_scores: Array of predicted scores
            true_relevance: Dict of {item_id: true_relevance_score}
            
        Returns:
            NDCG@K score
        """
        # Sort by predicted scores
        sorted_indices = np.argsort(-predicted_scores)
        sorted_items = [items[i] for i in sorted_indices]
        
        return self.compute(sorted_items, true_relevance)
    
    def compute_batch(
        self,
        all_recommendations: List[List[int]],
        all_relevant: List[Union[List[int], Dict[int, float]]],
        all_relevance_scores: Optional[List[Dict[int, float]]] = None
    ) -> Dict[str, float]:
        """
        Calculate NDCG@K for multiple users.
        
        Args:
            all_recommendations: List of recommendation lists
            all_relevant: List of relevant item specifications
            all_relevance_scores: Optional list of relevance score dicts
            
        Returns:
            Dictionary with mean and individual NDCG scores
        """
        if len(all_recommendations) != len(all_relevant):
            raise ValueError("Number of recommendations must match number of relevant lists")
        
        scores = []
        for i, (recs, relevant) in enumerate(zip(all_recommendations, all_relevant)):
            rel_scores = all_relevance_scores[i] if all_relevance_scores else None
            score = self.compute(recs, relevant, rel_scores)
            scores.append(score)
        
        return {
            'mean_ndcg': np.mean(scores),
            'std_ndcg': np.std(scores),
            'min_ndcg': np.min(scores),
            'max_ndcg': np.max(scores),
            'individual_scores': scores
        }
