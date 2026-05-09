"""
Interaction Ranker - Session-aware ranking based on user interactions

Implements ranking strategies that leverage:
- Real-time interaction signals
- Session context
- Short-term preferences
- Interaction recency weighting
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .ranker import BaseRanker

logger = logging.getLogger(__name__)


class InteractionRanker(BaseRanker):
    """
    Ranking model based on user interaction patterns.
    
    Features:
    - Recency-weighted interaction scoring
    - Session-based preference detection
    - Category affinity from recent interactions
    - Diversity-aware interaction balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize interaction ranker.
        
        Args:
            config: Configuration with interaction parameters
        """
        super().__init__(config)
        
        # Interaction parameters
        self.recency_half_life = config.get('recency_half_life', 7)  # days
        self.interaction_weights = config.get('interaction_weights', {
            'view': 1.0,
            'click': 2.0,
            'add_to_cart': 3.0,
            'purchase': 5.0
        })
        self.category_decay = config.get('category_decay', 0.9)
        self.max_interaction_history = config.get('max_interaction_history', 100)
        
        # Feature columns for interaction-based ranking
        self.feature_columns = config.get('feature_columns', [
            'user_id', 'item_id', 'category_id', 
            'interaction_count', 'recency_score', 'affinity_score'
        ])
        
        self.is_trained = True  # This ranker doesn't require training
        
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Fit the interaction ranker (learns interaction patterns).
        
        Args:
            train_df: Training dataframe with historical interactions
            val_df: Optional validation dataframe
            
        Returns:
            Empty dict (no training metrics for this ranker)
        """
        logger.info(f"Fitting interaction ranker on {len(train_df)} interactions")
        
        # Build user interaction profiles
        self.user_profiles = self._build_user_profiles(train_df)
        
        # Build item statistics
        self.item_stats = self._build_item_stats(train_df)
        
        # Build category affinity
        self.category_affinity = self._build_category_affinity(train_df)
        
        return {'interactions_processed': len(train_df)}
    
    def _build_user_profiles(self, interactions: pd.DataFrame) -> Dict[int, Dict]:
        """
        Build user interaction profiles.
        
        Args:
            interactions: Historical interaction data
            
        Returns:
            Dictionary of user_id -> profile
        """
        user_profiles = {}
        
        if 'timestamp' not in interactions.columns:
            interactions = interactions.copy()
            interactions['timestamp'] = datetime.now()
        
        # Sort by timestamp descending
        sorted_interactions = interactions.sort_values('timestamp', ascending=False)
        
        for _, row in sorted_interactions.iterrows():
            user_id = row.get('user_id')
            if user_id is None:
                continue
                
            if user_id not in user_profiles:
                user_profiles[user_id] = {
                    'recent_items': [],
                    'recent_categories': [],
                    'interaction_counts': {},
                    'last_activity': row['timestamp']
                }
            
            profile = user_profiles[user_id]
            item_id = row.get('item_id')
            category = row.get('category_id', 'unknown')
            interaction_type = row.get('interaction_type', 'view')
            
            # Add to recent items
            if item_id and len(profile['recent_items']) < self.max_interaction_history:
                profile['recent_items'].append(item_id)
            
            # Add to recent categories
            if len(profile['recent_categories']) < self.max_interaction_history:
                profile['recent_categories'].append(category)
            
            # Update interaction counts
            if item_id:
                profile['interaction_counts'][item_id] = \
                    profile['interaction_counts'].get(item_id, 0) + \
                    self.interaction_weights.get(interaction_type, 1.0)
        
        return user_profiles
    
    def _build_item_stats(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Build item-level statistics.
        
        Args:
            interactions: Historical interaction data
            
        Returns:
            DataFrame with item statistics
        """
        if 'item_id' not in interactions.columns:
            return pd.DataFrame()
        
        stats = interactions.groupby('item_id').agg({
            'user_id': 'count',
            'interaction_type': lambda x: (x == 'purchase').sum()
        }).rename(columns={
            'user_id': 'total_interactions',
            'interaction_type': 'purchase_count'
        })
        
        return stats
    
    def _build_category_affinity(self, interactions: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Build user-category affinity scores.
        
        Args:
            interactions: Historical interaction data
            
        Returns:
            Dictionary of user_id -> {category: affinity_score}
        """
        affinity = {}
        
        if 'user_id' not in interactions.columns or 'category_id' not in interactions.columns:
            return affinity
        
        for user_id in interactions['user_id'].unique():
            user_data = interactions[interactions['user_id'] == user_id]
            category_counts = user_data['category_id'].value_counts()
            
            # Normalize to get affinity scores
            total = category_counts.sum()
            if total > 0:
                affinity[user_id] = (category_counts / total).to_dict()
        
        return affinity
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate relevance scores based on interaction patterns.
        
        Args:
            features: Feature dataframe with user/item information
            
        Returns:
            Array of relevance scores
        """
        scores = np.zeros(len(features))
        
        for idx, row in features.iterrows():
            user_id = row.get('user_id')
            item_id = row.get('item_id')
            category = row.get('category_id', 'unknown')
            
            score = 0.0
            
            # User-based scoring
            if user_id and user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                
                # Recency score for item
                if item_id and item_id in profile['interaction_counts']:
                    score += profile['interaction_counts'][item_id] * 0.5
                
                # Category affinity
                if category and category in profile.get('recent_categories', []):
                    recency_factor = self._calculate_recency_factor(profile['last_activity'])
                    score += recency_factor * 2.0
            
            # Item popularity component
            if item_id and self.item_stats is not None and not self.item_stats.empty:
                if item_id in self.item_stats.index:
                    stats = self.item_stats.loc[item_id]
                    score += np.log1p(stats['total_interactions']) * 0.3
            
            scores[idx] = score
        
        return scores
    
    def _calculate_recency_factor(self, last_activity: datetime) -> float:
        """
        Calculate recency decay factor.
        
        Args:
            last_activity: Timestamp of last activity
            
        Returns:
            Recency factor between 0 and 1
        """
        if isinstance(last_activity, str):
            last_activity = pd.to_datetime(last_activity)
        
        days_since = (datetime.now() - last_activity).days
        half_life = self.recency_half_life
        
        # Exponential decay
        recency_factor = 0.5 ** (days_since / half_life)
        return max(0.0, min(1.0, recency_factor))
    
    def save(self, path: str) -> None:
        """Save model state."""
        import joblib
        from pathlib import Path
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'user_profiles': self.user_profiles,
            'item_stats': self.item_stats,
            'category_affinity': self.category_affinity,
            'config': self.config
        }
        
        joblib.dump(state, save_path / 'interaction_ranker.pkl')
        logger.info(f"Interaction ranker saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model state."""
        import joblib
        from pathlib import Path
        
        load_path = Path(path)
        state = joblib.load(load_path / 'interaction_ranker.pkl')
        
        self.user_profiles = state['user_profiles']
        self.item_stats = state['item_stats']
        self.category_affinity = state['category_affinity']
        self.is_trained = True
        
        logger.info(f"Interaction ranker loaded from {path}")
