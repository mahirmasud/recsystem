"""
Cold Start Handler - Re-ranking for new users and items

Implements cold-start mitigation strategies:
- Popularity-based fallback for new users
- Content-based recommendations for new items
- Hybrid approaches for limited history
- Exploration-focused ranking
"""

import logging
from typing import Dict, List, Optional, Any, Set
import numpy as np
import pandas as pd

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class ColdStartHandler(BaseReRanker):
    """
    Re-ranker that handles cold-start scenarios for users and items.
    
    Strategies:
    - New user handling (popularity, trending, demographic)
    - New item handling (content similarity, category matching)
    - Limited history handling (exploration boost)
    - Hybrid cold-start mitigation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cold start handler.
        
        Args:
            config: Configuration with cold-start parameters
        """
        super().__init__(config)
        
        self.popularity_column = config.get('popularity_column', 'popularity_score')
        self.trending_column = config.get('trending_column', 'trending_score')
        self.category_column = config.get('category_column', 'category_id')
        self.created_at_column = config.get('created_at_column', 'created_at')
        
        # New user strategy
        self.new_user_strategy = config.get('new_user_strategy', 'popularity')  # popularity, trending, diverse
        self.new_user_threshold = config.get('new_user_threshold', 5)  # Min interactions to be non-new
        
        # New item strategy  
        self.new_item_days_threshold = config.get('new_item_days_threshold', 30)
        self.new_item_boost = config.get('new_item_boost', 0.3)
        
        # Exploration settings
        self.exploration_ratio = config.get('exploration_ratio', 0.2)
        self.min_history_threshold = config.get('min_history_threshold', 10)
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply cold-start aware re-ranking.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context with user/item metadata
            
        Returns:
            Re-ranked DataFrame with cold-start adjustments
        """
        if len(recommendations) == 0:
            return recommendations
        
        recs = recommendations.copy()
        
        # Determine if this is a cold-start scenario
        is_new_user = self._is_new_user(user_id, context)
        new_items_mask = self._identify_new_items(recs)
        
        # Apply appropriate strategy
        if is_new_user:
            logger.info(f"Applying new user strategy: {self.new_user_strategy}")
            recs = self._handle_new_user(recs, context)
        
        # Boost new items for exploration
        if new_items_mask.any():
            recs = self._boost_new_items(recs, new_items_mask)
        
        # Apply exploration for limited history
        if context and context.get('user_interaction_count', 0) < self.min_history_threshold:
            recs = self._apply_exploration(recs)
        
        logger.info(f"Cold start handling applied: {len(recs)} items, new_user={is_new_user}")
        return recs
    
    def _is_new_user(self, user_id: Optional[int], context: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if user is new (cold start).
        
        Args:
            user_id: User identifier
            context: Context with user history
            
        Returns:
            True if user is considered new
        """
        if user_id is None:
            return True
        
        # Check context for interaction count
        if context:
            interaction_count = context.get('user_interaction_count', 0)
            if interaction_count < self.new_user_threshold:
                return True
        
        # Could check against stored user profiles
        return False
    
    def _identify_new_items(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Identify items that are new (within threshold days).
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Boolean mask of new items
        """
        from datetime import datetime, timedelta
        
        if self.created_at_column not in recs.columns:
            return np.zeros(len(recs), dtype=bool)
        
        cutoff_date = datetime.now() - timedelta(days=self.new_item_days_threshold)
        new_items_mask = pd.to_datetime(recs[self.created_at_column]) > cutoff_date
        
        return new_items_mask
    
    def _handle_new_user(
        self,
        recs: pd.DataFrame,
        context: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Handle recommendations for new users.
        
        Args:
            recs: Recommendations DataFrame
            context: Optional context
            
        Returns:
            Adjusted recommendations
        """
        if self.new_user_strategy == 'popularity':
            return self._rank_by_popularity(recs)
        elif self.new_user_strategy == 'trending':
            return self._rank_by_trending(recs)
        elif self.new_user_strategy == 'diverse':
            return self._rank_diverse(recs)
        else:
            return self._rank_by_popularity(recs)
    
    def _rank_by_popularity(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Rank items by popularity for new users.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Popularity-ranked recommendations
        """
        if self.popularity_column in recs.columns:
            recs = recs.sort_values(self.popularity_column, ascending=False)
        elif 'ranking_score' in recs.columns:
            # Use existing score as fallback
            recs = recs.sort_values('ranking_score', ascending=False)
        
        return recs.reset_index(drop=True)
    
    def _rank_by_trending(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Rank items by trending score for new users.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Trending-ranked recommendations
        """
        if self.trending_column in recs.columns:
            recs = recs.sort_values(self.trending_column, ascending=False)
        elif self.popularity_column in recs.columns:
            recs = recs.sort_values(self.popularity_column, ascending=False)
        
        return recs.reset_index(drop=True)
    
    def _rank_diverse(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Rank items with diversity focus for new users.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Diversity-ranked recommendations
        """
        if self.category_column not in recs.columns:
            return self._rank_by_popularity(recs)
        
        # Select top items from each category
        selected_indices = []
        categories_seen: Set = set()
        
        # Sort by popularity first
        if self.popularity_column in recs.columns:
            sorted_recs = recs.sort_values(self.popularity_column, ascending=False)
        else:
            sorted_recs = recs
        
        for idx, row in sorted_recs.iterrows():
            cat = row[self.category_column]
            if cat not in categories_seen or len(selected_indices) < len(recs) // 2:
                selected_indices.append(idx)
                categories_seen.add(cat)
        
        if selected_indices:
            recs = recs.loc[selected_indices].reset_index(drop=True)
        
        return recs
    
    def _boost_new_items(
        self,
        recs: pd.DataFrame,
        new_items_mask: np.ndarray
    ) -> pd.DataFrame:
        """
        Apply boost to new items for exploration.
        
        Args:
            recs: Recommendations DataFrame
            new_items_mask: Boolean mask of new items
            
        Returns:
            Recommendations with new item boost
        """
        for idx in recs[new_items_mask].index:
            if 'rerank_score' in recs.columns:
                recs.loc[idx, 'rerank_score'] += self.new_item_boost
            elif 'ranking_score' in recs.columns:
                recs.loc[idx, 'ranking_score'] += self.new_item_boost
        
        # Add new item flag
        recs['is_new_item'] = new_items_mask
        
        return recs
    
    def _apply_exploration(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Apply exploration strategy for users with limited history.
        
        Injects some random/diverse items to learn preferences.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Recommendations with exploration
        """
        n_explore = max(1, int(len(recs) * self.exploration_ratio))
        
        # Select items for exploration (lower ranked but diverse)
        if self.category_column in recs.columns and len(recs) > n_explore:
            # Pick diverse categories from lower half
            lower_half = recs.iloc[len(recs)//2:]
            unique_categories = lower_half[self.category_column].unique()
            
            explore_indices = []
            for cat in unique_categories[:n_explore]:
                cat_items = lower_half[lower_half[self.category_column] == cat]
                if len(cat_items) > 0:
                    explore_indices.append(cat_items.index[0])
        else:
            # Random selection
            explore_indices = recs.sample(n=min(n_explore, len(recs))).index.tolist()
        
        # Mark exploration items
        recs['is_exploration'] = False
        recs.loc[explore_indices, 'is_exploration'] = True
        
        logger.info(f"Exploration: {len(explore_indices)} items marked")
        return recs
    
    def get_fallback_recommendations(
        self,
        all_items: pd.DataFrame,
        n: int = 10,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate fallback recommendations when personalization fails.
        
        Args:
            all_items: All available items DataFrame
            n: Number of recommendations
            strategy: Fallback strategy
            
        Returns:
            Fallback recommendations
        """
        strategy = strategy or self.new_user_strategy
        
        if strategy == 'popularity':
            return self._rank_by_popularity(all_items).head(n)
        elif strategy == 'trending':
            return self._rank_by_trending(all_items).head(n)
        else:
            return self._rank_by_popularity(all_items).head(n)
