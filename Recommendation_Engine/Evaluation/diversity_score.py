"""
Diversity Score - Evaluation metric for recommendation diversity

Calculates various diversity metrics for recommendations.
"""

import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


class DiversityScorer:
    """
    Calculate diversity metrics for recommendation evaluation.
    
    Metrics:
    - Category diversity (unique categories / K)
    - Intra-list distance
    - Coverage of item space
    """
    
    def __init__(self, category_column: str = 'category_id'):
        """
        Initialize diversity scorer.
        
        Args:
            category_column: Name of category column in items
        """
        self.category_column = category_column
        
    def compute_category_diversity(
        self,
        recommendations: List[Dict[str, Any]],
        k: int = 10
    ) -> float:
        """
        Calculate category diversity in top K recommendations.
        
        Diversity = unique_categories / min(K, total_items)
        
        Args:
            recommendations: List of recommendation dicts with category info
            k: Number of top items to consider
            
        Returns:
            Category diversity score [0, 1]
        """
        if len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        
        # Extract categories
        categories: Set = set()
        for rec in top_k:
            if self.category_column in rec:
                categories.add(rec[self.category_column])
        
        if len(categories) == 0:
            return 0.0
        
        # Normalize by maximum possible diversity
        max_diversity = min(len(top_k), len(categories))
        diversity = len(categories) / max_diversity
        
        return diversity
    
    def compute_intra_list_distance(
        self,
        recommendations: List[Dict[str, Any]],
        item_embeddings: Optional[Dict[int, np.ndarray]] = None
    ) -> float:
        """
        Calculate intra-list distance (average pairwise dissimilarity).
        
        Args:
            recommendations: List of recommendation dicts
            item_embeddings: Optional dict of {item_id: embedding_vector}
            
        Returns:
            Average pairwise distance [0, 1]
        """
        if len(recommendations) <= 1:
            return 0.0
        
        n = len(recommendations)
        total_distance = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                item_i = recommendations[i].get('item_id')
                item_j = recommendations[j].get('item_id')
                
                if item_embeddings and item_i in item_embeddings and item_j in item_embeddings:
                    # Cosine distance from embeddings
                    emb_i = item_embeddings[item_i]
                    emb_j = item_embeddings[item_j]
                    
                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        distance = (1 - cosine_sim) / 2  # Normalize to [0, 1]
                    else:
                        distance = 0.5
                else:
                    # Category-based distance
                    cat_i = recommendations[i].get(self.category_column)
                    cat_j = recommendations[j].get(self.category_column)
                    distance = 0.0 if cat_i == cat_j else 1.0
                
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        return total_distance / pair_count
    
    def compute_coverage(
        self,
        all_recommendations: List[List[int]],
        all_items: Set[int]
    ) -> float:
        """
        Calculate catalog coverage (percentage of items recommended).
        
        Args:
            all_recommendations: List of recommendation lists for all users
            all_items: Set of all available item IDs
            
        Returns:
            Coverage percentage [0, 1]
        """
        if len(all_items) == 0:
            return 0.0
        
        # Get all unique recommended items
        recommended_items: Set[int] = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        coverage = len(recommended_items) / len(all_items)
        
        return coverage
    
    def compute_batch(
        self,
        all_recommendations: List[List[Dict[str, Any]]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate diversity metrics for multiple users.
        
        Args:
            all_recommendations: List of recommendation lists
            k: Number of top items to consider
            
        Returns:
            Dictionary with diversity statistics
        """
        category_diversities = []
        ild_scores = []
        
        for recs in all_recommendations:
            cat_div = self.compute_category_diversity(recs, k)
            ild = self.compute_intra_list_distance(recs)
            
            category_diversities.append(cat_div)
            ild_scores.append(ild)
        
        return {
            'mean_category_diversity': np.mean(category_diversities),
            'std_category_diversity': np.std(category_diversities),
            'mean_ild': np.mean(ild_scores),
            'std_ild': np.std(ild_scores),
            'individual_diversities': category_diversities,
            'individual_ild': ild_scores
        }
