"""
Coverage Metric - Evaluation metric for catalog coverage

Calculates what percentage of the item catalog gets recommended.
"""

import logging
from typing import List, Dict, Any, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CoverageCalculator:
    """
    Calculate coverage metrics for recommendation evaluation.
    
    Metrics:
    - Catalog coverage (% of items ever recommended)
    - User coverage (% of users receiving recommendations)
    - Aggregate diversity (unique items / K * num_users)
    """
    
    def compute_catalog_coverage(
        self,
        all_recommendations: List[List[int]],
        all_items: Set[int]
    ) -> float:
        """
        Calculate catalog coverage.
        
        Coverage = unique_recommended_items / total_items
        
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
    
    def compute_aggregate_diversity(
        self,
        all_recommendations: List[List[int]],
        k: int = 10
    ) -> float:
        """
        Calculate aggregate diversity across all users.
        
        Agg_Diversity = unique_items_in_all_topK / (K * num_users)
        
        Args:
            all_recommendations: List of recommendation lists
            k: Number of top items per user to consider
            
        Returns:
            Aggregate diversity score [0, 1]
        """
        if len(all_recommendations) == 0:
            return 0.0
        
        # Get unique items in top K across all users
        unique_items: Set[int] = set()
        for recs in all_recommendations:
            top_k = recs[:k]
            unique_items.update(top_k)
        
        # Maximum possible unique items
        max_unique = min(k * len(all_recommendations), len(unique_items))
        
        # Normalize
        diversity = len(unique_items) / max_unique if max_unique > 0 else 0.0
        
        return diversity
    
    def compute_user_coverage(
        self,
        users_with_recs: Set[int],
        total_users: Set[int]
    ) -> float:
        """
        Calculate user coverage.
        
        Coverage = users_with_recommendations / total_users
        
        Args:
            users_with_recs: Set of user IDs who received recommendations
            total_users: Set of all user IDs
            
        Returns:
            User coverage percentage [0, 1]
        """
        if len(total_users) == 0:
            return 0.0
        
        coverage = len(users_with_recs) / len(total_users)
        return coverage
    
    def compute_item_distribution(
        self,
        all_recommendations: List[List[int]]
    ) -> Dict[str, float]:
        """
        Analyze the distribution of item recommendations.
        
        Args:
            all_recommendations: List of recommendation lists
            
        Returns:
            Dictionary with distribution statistics
        """
        # Count how many times each item is recommended
        item_counts: Dict[int, int] = {}
        for recs in all_recommendations:
            for item_id in recs:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        if len(item_counts) == 0:
            return {
                'mean_recommendations': 0,
                'std_recommendations': 0,
                'max_recommendations': 0,
                'min_recommendations': 0,
                'gini_coefficient': 0
            }
        
        counts = list(item_counts.values())
        
        return {
            'mean_recommendations': np.mean(counts),
            'std_recommendations': np.std(counts),
            'max_recommendations': max(counts),
            'min_recommendations': min(counts),
            'median_recommendations': np.median(counts),
            'items_recommended_once': sum(1 for c in counts if c == 1),
            'gini_coefficient': self._compute_gini(counts)
        }
    
    def _compute_gini(self, values: List[int]) -> float:
        """
        Compute Gini coefficient for recommendation distribution.
        
        Higher Gini = more unequal (popular items get recommended more)
        
        Args:
            values: List of item recommendation counts
            
        Returns:
            Gini coefficient [0, 1]
        """
        if len(values) == 0 or sum(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini using the formula
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return max(0, min(1, gini))
    
    def compute_batch(
        self,
        all_recommendations: List[List[int]],
        all_items: Set[int],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate all coverage metrics.
        
        Args:
            all_recommendations: List of recommendation lists
            all_items: Set of all available item IDs
            k: Number of top items to consider
            
        Returns:
            Dictionary with all coverage metrics
        """
        return {
            'catalog_coverage': self.compute_catalog_coverage(all_recommendations, all_items),
            'aggregate_diversity': self.compute_aggregate_diversity(all_recommendations, k),
            'distribution': self.compute_item_distribution(all_recommendations)
        }
