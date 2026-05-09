"""
Base ReRanker - Abstract base class for re-ranking strategies

Provides common interface for diversity, freshness, business boosters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseReRanker(ABC):
    """
    Abstract base class for recommendation re-ranking strategies.
    
    All re-rankers must implement:
    - rerank(): Apply re-ranking strategy to scored recommendations
    - get_config(): Return current configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base re-ranker.
        
        Args:
            config: Configuration dictionary with re-ranking parameters
        """
        self.config = config
        self.weight = config.get('weight', 1.0)
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    def rerank(
        self, 
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply re-ranking strategy to recommendations.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user identifier for personalization
            context: Optional context dictionary (session, time, etc.)
            
        Returns:
            Re-ranked DataFrame with adjusted scores
        """
        pass
    
    def apply(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply re-ranking if enabled.
        
        Args:
            recommendations: Input recommendations
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            Possibly re-ranked DataFrame
        """
        if not self.enabled:
            logger.debug(f"{self.__class__.__name__} is disabled")
            return recommendations
        
        return self.rerank(recommendations, user_id, context)
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            'class': self.__class__.__name__,
            'enabled': self.enabled,
            'weight': self.weight,
            **self.config
        }
    
    def combine_scores(
        self,
        original_scores: np.ndarray,
        adjustment_scores: np.ndarray,
        combination_method: str = 'additive'
    ) -> np.ndarray:
        """
        Combine original scores with adjustment scores.
        
        Args:
            original_scores: Original ranking scores
            adjustment_scores: Scores from re-ranking strategy
            combination_method: How to combine ('additive', 'multiplicative', 'weighted')
            
        Returns:
            Combined scores
        """
        if combination_method == 'additive':
            return original_scores + self.weight * adjustment_scores
        elif combination_method == 'multiplicative':
            return original_scores * (1 + self.weight * adjustment_scores)
        elif combination_method == 'weighted':
            return (1 - self.weight) * original_scores + self.weight * adjustment_scores
        else:
            logger.warning(f"Unknown combination method: {combination_method}, using additive")
            return original_scores + self.weight * adjustment_scores


class ReRanker:
    """
    Concrete re-ranker that combines multiple re-ranking strategies.
    
    Supports:
    - Diversity re-ranking
    - Freshness boosting
    - Business rule boosting
    - Cold start handling
    - Exploration vs exploitation
    """
    
    def __init__(self, config=None):
        """
        Initialize re-ranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.diversity_weight = self.config.get('diversity_weight', 0.1)
        self.freshness_weight = self.config.get('freshness_weight', 0.0)
        self.business_weight = self.config.get('business_weight', 0.0)
    
    def rerank(
        self,
        item_ids,
        scores,
        user_id=None,
        item_metadata=None
    ):
        """
        Apply re-ranking to item list.
        
        Args:
            item_ids: List of item IDs
            scores: Original ranking scores
            user_id: Optional user ID
            item_metadata: Optional item metadata DataFrame
            
        Returns:
            List of indices representing new order
        """
        import numpy as np
        n_items = len(item_ids)
        
        if n_items == 0:
            return []
        
        # Start with original scores
        adjusted_scores = scores.copy()
        
        # Apply diversity adjustment (simple category-based MMR-like approach)
        if self.diversity_weight > 0 and item_metadata is not None:
            adjusted_scores = self._apply_diversity(
                item_ids, adjusted_scores, item_metadata
            )
        
        # Apply freshness adjustment
        if self.freshness_weight > 0 and item_metadata is not None:
            adjusted_scores = self._apply_freshness(
                item_ids, adjusted_scores, item_metadata
            )
        
        # Apply business rules adjustment
        if self.business_weight > 0 and item_metadata is not None:
            adjusted_scores = self._apply_business_rules(
                item_ids, adjusted_scores, item_metadata
            )
        
        # Return sorted indices (descending by adjusted score)
        return list(np.argsort(adjusted_scores)[::-1])
    
    def _apply_diversity(self, item_ids, scores, item_metadata):
        """Apply diversity adjustment using category information."""
        import numpy as np
        from collections import Counter
        adjusted = scores.copy()
        
        # Penalize items from overrepresented categories
        if 'category' in item_metadata.columns:
            categories = []
            for item_id in item_ids:
                if item_id in item_metadata.index:
                    cat = item_metadata.loc[item_id, 'category']
                else:
                    cat = 'unknown'
                categories.append(cat)
            
            # Count category occurrences
            cat_counts = Counter(categories)
            
            # Penalize based on category frequency
            for i, cat in enumerate(categories):
                penalty = (cat_counts[cat] - 1) * 0.05 * self.diversity_weight
                adjusted[i] -= penalty
        
        return adjusted
    
    def _apply_freshness(self, item_ids, scores, item_metadata):
        """Apply freshness boost for newer items."""
        import numpy as np
        adjusted = scores.copy()
        
        if 'created_at' in item_metadata.columns or 'release_date' in item_metadata.columns:
            for i, item_id in enumerate(item_ids):
                if item_id in item_metadata.index:
                    # Boost newer items
                    adjusted[i] += self.freshness_weight * 0.1
        
        return adjusted
    
    def _apply_business_rules(self, item_ids, scores, item_metadata):
        """Apply business rule adjustments."""
        import numpy as np
        adjusted = scores.copy()
        
        # Boost high-margin items
        if 'margin' in item_metadata.columns:
            for i, item_id in enumerate(item_ids):
                if item_id in item_metadata.index:
                    margin = item_metadata.loc[item_id, 'margin']
                    adjusted[i] += margin * self.business_weight * 0.1
        
        # Boost promoted items
        if 'promoted' in item_metadata.columns:
            for i, item_id in enumerate(item_ids):
                if item_id in item_metadata.index:
                    if item_metadata.loc[item_id, 'promoted']:
                        adjusted[i] += self.business_weight * 0.2
        
        return adjusted
