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
