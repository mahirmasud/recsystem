"""
MAP (Mean Average Precision) Metric - Evaluation metric for ranking quality

Calculates Mean Average Precision considering position of relevant items.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MAPMetric:
    """
    Calculate Mean Average Precision (MAP) for recommendation evaluation.
    
    MAP considers both precision and the position of relevant items.
    """
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize MAP metric.
        
        Args:
            k: Optional cutoff for calculation (None = no cutoff)
        """
        self.k = k
        
    def compute_average_precision(
        self,
        recommendations: List[int],
        relevant_items: List[int]
    ) -> float:
        """
        Calculate Average Precision for a single user.
        
        Args:
            recommendations: List of recommended item IDs (ranked)
            relevant_items: List of relevant item IDs (ground truth)
            
        Returns:
            Average Precision score
        """
        if len(recommendations) == 0 or len(relevant_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        
        # Apply cutoff if specified
        recs_to_eval = recommendations[:self.k] if self.k else recommendations
        
        num_relevant_found = 0
        precision_sum = 0.0
        
        for i, item_id in enumerate(recs_to_eval):
            if item_id in relevant_set:
                num_relevant_found += 1
                # Precision at this position
                precision_at_i = num_relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        if num_relevant_found == 0:
            return 0.0
        
        # Average precision
        ap = precision_sum / min(len(relevant_items), len(recs_to_eval))
        
        return ap
    
    def compute(
        self,
        all_recommendations: List[List[int]],
        all_relevant: List[List[int]]
    ) -> float:
        """
        Calculate Mean Average Precision across multiple users.
        
        Args:
            all_recommendations: List of recommendation lists (one per user)
            all_relevant: List of relevant item lists (one per user)
            
        Returns:
            Mean Average Precision score
        """
        if len(all_recommendations) != len(all_relevant):
            raise ValueError("Number of recommendations must match number of relevant lists")
        
        if len(all_recommendations) == 0:
            return 0.0
        
        ap_scores = []
        for recs, relevant in zip(all_recommendations, all_relevant):
            ap = self.compute_average_precision(recs, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def compute_batch(
        self,
        all_recommendations: List[List[int]],
        all_relevant: List[List[int]]
    ) -> Dict[str, float]:
        """
        Calculate MAP with detailed statistics.
        
        Args:
            all_recommendations: List of recommendation lists
            all_relevant: List of relevant item lists
            
        Returns:
            Dictionary with MAP and individual AP scores
        """
        if len(all_recommendations) != len(all_relevant):
            raise ValueError("Number of recommendations must match number of relevant lists")
        
        ap_scores = []
        for recs, relevant in zip(all_recommendations, all_relevant):
            ap = self.compute_average_precision(recs, relevant)
            ap_scores.append(ap)
        
        return {
            'map': np.mean(ap_scores),
            'std_ap': np.std(ap_scores),
            'min_ap': np.min(ap_scores),
            'max_ap': np.max(ap_scores),
            'individual_scores': ap_scores
        }
