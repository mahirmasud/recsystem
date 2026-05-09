"""
Freshness Booster - Re-ranking for content freshness

Implements freshness strategies:
- Recency-based boosting
- Trending item promotion
- New item exploration
- Time-decay scoring
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class FreshnessBooster(BaseReRanker):
    """
    Re-ranker that promotes fresh and trending content.
    
    Strategies:
    - Recency scoring with exponential decay
    - Trend detection based on velocity
    - New item boost for cold-start
    - Seasonal/trending adjustments
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize freshness booster.
        
        Args:
            config: Configuration with freshness parameters
        """
        super().__init__(config)
        
        self.half_life_days = config.get('half_life_days', 7)
        self.trending_window_days = config.get('trending_window_days', 3)
        self.new_item_boost = config.get('new_item_boost', 0.5)
        self.new_item_threshold_days = config.get('new_item_threshold_days', 14)
        self.date_column = config.get('date_column', 'created_at')
        self.interaction_column = config.get('interaction_column', 'view_count')
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply freshness-focused re-ranking.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context (may include current_time)
            
        Returns:
            Re-ranked DataFrame with freshness adjustments
        """
        if len(recommendations) == 0:
            return recommendations
        
        recs = recommendations.copy()
        current_time = context.get('current_time', datetime.now()) if context else datetime.now()
        
        # Calculate freshness scores
        freshness_scores = self._calculate_freshness_scores(recs, current_time)
        
        # Calculate trending scores
        trending_scores = self._calculate_trending_scores(recs)
        
        # Combine freshness components
        combined_freshness = 0.7 * freshness_scores + 0.3 * trending_scores
        
        # Add to recommendations
        recs['freshness_score'] = freshness_scores
        recs['trending_score'] = trending_scores
        
        # Adjust final scores
        if 'ranking_score' in recs.columns:
            # Normalize freshness to same scale as ranking scores
            score_range = recs['ranking_score'].max() - recs['ranking_score'].min()
            if score_range > 0:
                normalized_freshness = combined_freshness * score_range * self.weight
            else:
                normalized_freshness = combined_freshness * self.weight
            
            recs['rerank_score'] = recs['ranking_score'] + normalized_freshness
        
        # Sort by adjusted score
        score_col = 'rerank_score' if 'rerank_score' in recs.columns else 'ranking_score'
        if score_col in recs.columns:
            recs = recs.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        logger.info(f"Freshness re-ranking applied: {len(recs)} items")
        return recs
    
    def _calculate_freshness_scores(self, recs: pd.DataFrame, current_time: datetime) -> np.ndarray:
        """
        Calculate recency-based freshness scores.
        
        Uses exponential decay: score = 0.5^(days_since_creation / half_life)
        
        Args:
            recs: Recommendations DataFrame
            current_time: Current timestamp for calculation
            
        Returns:
            Array of freshness scores [0, 1]
        """
        scores = np.ones(len(recs))  # Default to 1.0 if no date info
        
        if self.date_column not in recs.columns:
            logger.debug(f"Date column '{self.date_column}' not found, using default freshness")
            return scores
        
        for idx, row in recs.iterrows():
            created_at = row[self.date_column]
            
            # Handle various date formats
            if isinstance(created_at, str):
                try:
                    created_at = pd.to_datetime(created_at)
                except:
                    continue
            elif not isinstance(created_at, (datetime, pd.Timestamp)):
                continue
            
            # Calculate days since creation
            days_since = (current_time - created_at).days
            
            if days_since >= 0:
                # Exponential decay
                scores[idx] = 0.5 ** (days_since / self.half_life_days)
            else:
                # Future dates get max score
                scores[idx] = 1.0
        
        return scores
    
    def _calculate_trending_scores(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Calculate trending scores based on interaction velocity.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of trending scores [0, 1]
        """
        scores = np.zeros(len(recs))
        
        if self.interaction_column not in recs.columns:
            # Use popularity as proxy if available
            if 'popularity_score' in recs.columns:
                pop = recs['popularity_score'].values
                if pop.max() > pop.min():
                    scores = (pop - pop.min()) / (pop.max() - pop.min())
            return scores
        
        interactions = recs[self.interaction_column].values
        
        # Normalize to [0, 1]
        if interactions.max() > interactions.min():
            scores = (interactions - interactions.min()) / (interactions.max() - interactions.min())
        else:
            scores = np.ones(len(recs)) * 0.5
        
        return scores
    
    def boost_new_items(
        self,
        recs: pd.DataFrame,
        threshold_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply extra boost to new items for exploration.
        
        Args:
            recs: Recommendations DataFrame
            threshold_days: Days threshold for "new" items
            
        Returns:
            DataFrame with new item boost applied
        """
        threshold = threshold_days or self.new_item_threshold_days
        
        if self.date_column not in recs.columns:
            return recs
        
        current_time = datetime.now()
        
        for idx, row in recs.iterrows():
            created_at = row[self.date_column]
            
            if isinstance(created_at, str):
                try:
                    created_at = pd.to_datetime(created_at)
                except:
                    continue
            
            days_since = (current_time - created_at).days
            
            if 0 <= days_since <= threshold:
                # Apply new item boost
                if 'rerank_score' in recs.columns:
                    recs.loc[idx, 'rerank_score'] += self.new_item_boost
                elif 'ranking_score' in recs.columns:
                    recs.loc[idx, 'ranking_score'] += self.new_item_boost
        
        return recs
