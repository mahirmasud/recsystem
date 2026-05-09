"""
Margin Booster - Re-ranking for profit margin optimization

Implements business-aware re-ranking:
- Profit margin boosting
- Revenue optimization
- Strategic product promotion
- Inventory-aware adjustments
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class MarginBooster(BaseReRanker):
    """
    Re-ranker that optimizes for profit margins and business metrics.
    
    Strategies:
    - Direct margin boosting
    - Revenue-weighted scoring
    - Strategic category promotion
    - Inventory level considerations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize margin booster.
        
        Args:
            config: Configuration with margin parameters
        """
        super().__init__(config)
        
        self.margin_column = config.get('margin_column', 'profit_margin')
        self.price_column = config.get('price_column', 'price')
        self.revenue_weight = config.get('revenue_weight', 0.3)
        self.min_margin_threshold = config.get('min_margin_threshold', 0.0)
        self.max_boost_factor = config.get('max_boost_factor', 2.0)
        self.strategy = config.get('strategy', 'margin')  # 'margin', 'revenue', 'hybrid'
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply margin-focused re-ranking.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            Re-ranked DataFrame with margin adjustments
        """
        if len(recommendations) == 0:
            return recommendations
        
        recs = recommendations.copy()
        
        # Calculate business scores based on strategy
        if self.strategy == 'margin':
            business_scores = self._calculate_margin_scores(recs)
        elif self.strategy == 'revenue':
            business_scores = self._calculate_revenue_scores(recs)
        else:  # hybrid
            margin_scores = self._calculate_margin_scores(recs)
            revenue_scores = self._calculate_revenue_scores(recs)
            business_scores = (1 - self.revenue_weight) * margin_scores + self.revenue_weight * revenue_scores
        
        # Add business score to recommendations
        recs['business_score'] = business_scores
        
        # Adjust final scores
        if 'ranking_score' in recs.columns:
            # Normalize business scores
            score_range = recs['ranking_score'].max() - recs['ranking_score'].min()
            if score_range > 0:
                normalized_business = business_scores * score_range * self.weight
            else:
                normalized_business = business_scores * self.weight
            
            recs['rerank_score'] = recs['ranking_score'] + normalized_business
        
        # Sort by adjusted score
        score_col = 'rerank_score' if 'rerank_score' in recs.columns else 'ranking_score'
        if score_col in recs.columns:
            recs = recs.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        logger.info(f"Margin re-ranking applied: {len(recs)} items, strategy={self.strategy}")
        return recs
    
    def _calculate_margin_scores(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Calculate margin-based scores.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of margin scores [0, 1]
        """
        scores = np.zeros(len(recs))
        
        if self.margin_column not in recs.columns:
            logger.debug(f"Margin column '{self.margin_column}' not found")
            return np.ones(len(recs)) * 0.5  # Neutral score
        
        margins = recs[self.margin_column].values
        
        # Filter by minimum margin threshold
        valid_mask = margins >= self.min_margin_threshold
        
        if not valid_mask.any():
            logger.warning("No items meet minimum margin threshold")
            return scores
        
        # Normalize margins to [0, 1]
        valid_margins = margins[valid_mask]
        if valid_margins.max() > valid_margins.min():
            normalized = (valid_margins - valid_margins.min()) / (valid_margins.max() - valid_margins.min())
        else:
            normalized = np.ones(len(valid_margins)) * 0.5
        
        scores[valid_mask] = normalized
        
        # Cap boost factor
        scores = np.clip(scores, 0, self.max_boost_factor)
        
        return scores
    
    def _calculate_revenue_scores(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Calculate revenue-based scores (price * conversion probability).
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of revenue scores [0, 1]
        """
        scores = np.zeros(len(recs))
        
        # Need price column
        if self.price_column not in recs.columns:
            logger.debug(f"Price column '{self.price_column}' not found")
            return scores
        
        prices = recs[self.price_column].values
        
        # Use ranking score as proxy for conversion probability
        if 'ranking_score' in recs.columns:
            conversion_prob = recs['ranking_score'].values
            # Normalize conversion probability
            if conversion_prob.max() > conversion_prob.min():
                conversion_prob = (conversion_prob - conversion_prob.min()) / \
                                  (conversion_prob.max() - conversion_prob.min())
        else:
            conversion_prob = np.ones(len(prices)) * 0.5
        
        # Calculate expected revenue
        expected_revenue = prices * conversion_prob
        
        # Normalize to [0, 1]
        if expected_revenue.max() > expected_revenue.min():
            scores = (expected_revenue - expected_revenue.min()) / \
                     (expected_revenue.max() - expected_revenue.min())
        
        return scores
    
    def apply_strategic_boost(
        self,
        recs: pd.DataFrame,
        strategic_items: List[int],
        boost_factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Apply extra boost to strategic items.
        
        Args:
            recs: Recommendations DataFrame
            strategic_items: List of item IDs to boost
            boost_factor: Multiplicative boost factor
            
        Returns:
            DataFrame with strategic boost applied
        """
        if 'item_id' not in recs.columns:
            logger.warning("Cannot apply strategic boost: item_id column missing")
            return recs
        
        strategic_set = set(strategic_items)
        
        for idx, row in recs.iterrows():
            if row['item_id'] in strategic_set:
                if 'rerank_score' in recs.columns:
                    recs.loc[idx, 'rerank_score'] *= boost_factor
                elif 'ranking_score' in recs.columns:
                    recs.loc[idx, 'ranking_score'] *= boost_factor
        
        return recs
    
    def filter_low_margin(
        self,
        recs: pd.DataFrame,
        min_margin: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter out items below minimum margin threshold.
        
        Args:
            recs: Recommendations DataFrame
            min_margin: Minimum margin threshold
            
        Returns:
            Filtered DataFrame
        """
        threshold = min_margin if min_margin is not None else self.min_margin_threshold
        
        if self.margin_column not in recs.columns:
            return recs
        
        filtered = recs[recs[self.margin_column] >= threshold].copy()
        
        logger.info(f"Filtered {len(recs) - len(filtered)} low-margin items")
        return filtered
