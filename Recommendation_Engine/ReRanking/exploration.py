"""
Exploration Strategy - Re-ranking for exploration vs exploitation

Implements exploration strategies:
- Epsilon-greedy exploration
- Thompson sampling
- Upper Confidence Bound (UCB)
- Diversity-based exploration
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class ExplorationStrategy(BaseReRanker):
    """
    Re-ranker that balances exploration and exploitation.
    
    Strategies:
    - Epsilon-greedy random exploration
    - UCB-based uncertainty exploration
    - Thompson sampling for bandit optimization
    - Category diversity exploration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exploration strategy.
        
        Args:
            config: Configuration with exploration parameters
        """
        super().__init__(config)
        
        self.strategy = config.get('strategy', 'epsilon_greedy')  # epsilon_greedy, ucb, thompson
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.category_column = config.get('category_column', 'category_id')
        self.ucb_constant = config.get('ucb_constant', 2.0)
        
        # Track item statistics for UCB/Thompson
        self.item_pulls: Dict[int, int] = {}  # item_id -> number of times shown
        self.item_rewards: Dict[int, float] = {}  # item_id -> cumulative reward
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply exploration-aware re-ranking.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            Re-ranked DataFrame with exploration adjustments
        """
        if len(recommendations) == 0:
            return recommendations
        
        recs = recommendations.copy()
        
        # Apply exploration strategy
        if self.strategy == 'epsilon_greedy':
            recs = self._apply_epsilon_greedy(recs, context)
        elif self.strategy == 'ucb':
            recs = self._apply_ucb(recs)
        elif self.strategy == 'thompson':
            recs = self._apply_thompson(recs)
        else:
            logger.warning(f"Unknown exploration strategy: {self.strategy}")
        
        logger.info(f"Exploration strategy applied: {self.strategy}")
        return recs
    
    def _apply_epsilon_greedy(
        self,
        recs: pd.DataFrame,
        context: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply epsilon-greedy exploration.
        
        With probability epsilon, inject random items for exploration.
        
        Args:
            recs: Recommendations DataFrame
            context: Optional context
            
        Returns:
            Recommendations with exploration
        """
        # Decide whether to explore
        should_explore = np.random.random() < self.epsilon
        
        if not should_explore:
            # Exploit: use existing rankings
            recs['exploration_flag'] = 'exploit'
            return recs
        
        # Explore: shuffle or inject diverse items
        n_explore = max(1, len(recs) // 4)
        
        if self.category_column in recs.columns:
            # Diverse exploration: pick from different categories
            unique_categories = recs[self.category_column].unique()
            explore_items = []
            
            for cat in np.random.choice(unique_categories, size=min(n_explore, len(unique_categories)), replace=False):
                cat_items = recs[recs[self.category_column] == cat]
                if len(cat_items) > 0:
                    # Pick a non-top item for exploration
                    lower_items = cat_items.iloc[len(cat_items)//2:]
                    if len(lower_items) > 0:
                        explore_items.append(lower_items.sample(1).index[0])
            
            if explore_items:
                # Boost exploration items
                for idx in explore_items:
                    if 'ranking_score' in recs.columns:
                        recs.loc[idx, 'ranking_score'] += 0.1
        else:
            # Random exploration
            explore_indices = recs.sample(n=n_explore).index
            for idx in explore_indices:
                if 'ranking_score' in recs.columns:
                    recs.loc[idx, 'ranking_score'] += 0.1
        
        recs['exploration_flag'] = 'explore'
        
        # Re-sort by score
        if 'ranking_score' in recs.columns:
            recs = recs.sort_values('ranking_score', ascending=False).reset_index(drop=True)
        
        return recs
    
    def _apply_ucb(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Upper Confidence Bound (UCB) exploration.
        
        UCB score = mean_reward + c * sqrt(ln(total_pulls) / item_pulls)
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Recommendations with UCB-adjusted scores
        """
        total_pulls = sum(self.item_pulls.values()) + 1  # Avoid log(0)
        
        ucb_scores = np.zeros(len(recs))
        
        for idx, row in recs.iterrows():
            item_id = row.get('item_id')
            
            if item_id is None:
                continue
            
            pulls = self.item_pulls.get(item_id, 0)
            
            if pulls == 0:
                # Never pulled: high uncertainty, high UCB
                ucb_scores[idx] = float('inf')
            else:
                mean_reward = self.item_rewards.get(item_id, 0) / pulls
                exploration_bonus = self.ucb_constant * np.sqrt(np.log(total_pulls) / pulls)
                ucb_scores[idx] = mean_reward + exploration_bonus
        
        # Add UCB score to recommendations
        recs['ucb_score'] = ucb_scores
        
        # Combine with ranking score
        if 'ranking_score' in recs.columns:
            # Normalize UCB scores
            finite_ucb = ucb_scores[np.isfinite(ucb_scores)]
            if len(finite_ucb) > 0 and finite_ucb.max() > finite_ucb.min():
                normalized_ucb = np.clip(ucb_scores, 0, finite_ucb.max())
                normalized_ucb = (normalized_ucb - normalized_ucb.min()) / (normalized_ucb.max() - normalized_ucb.min())
            else:
                normalized_ucb = np.ones(len(ucb_scores)) * 0.5
            
            # Replace inf with max normalized value
            normalized_ucb[np.isinf(ucb_scores)] = 1.0
            
            recs['rerank_score'] = 0.7 * recs['ranking_score'] + 0.3 * normalized_ucb
            recs = recs.sort_values('rerank_score', ascending=False).reset_index(drop=True)
        
        return recs
    
    def _apply_thompson(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Thompson Sampling exploration.
        
        Samples from posterior distribution of rewards.
        Uses Beta distribution for binary rewards.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Recommendations with Thompson-sampled scores
        """
        thompson_scores = np.zeros(len(recs))
        
        for idx, row in recs.iterrows():
            item_id = row.get('item_id')
            
            if item_id is None:
                thompson_scores[idx] = 0.5
                continue
            
            # Get prior parameters (Beta distribution)
            successes = self.item_rewards.get(item_id, 1)  # Alpha - 1
            failures = self.item_pulls.get(item_id, 1) - successes  # Beta - 1
            
            # Ensure valid parameters
            successes = max(1, successes)
            failures = max(1, failures)
            
            # Sample from Beta distribution
            sampled_value = np.random.beta(successes, failures)
            thompson_scores[idx] = sampled_value
        
        # Add Thompson score to recommendations
        recs['thompson_score'] = thompson_scores
        
        # Combine with ranking score
        if 'ranking_score' in recs.columns:
            recs['rerank_score'] = 0.6 * recs['ranking_score'] + 0.4 * thompson_scores
            recs = recs.sort_values('rerank_score', ascending=False).reset_index(drop=True)
        
        return recs
    
    def record_outcome(
        self,
        item_id: int,
        reward: float
    ) -> None:
        """
        Record the outcome of a recommendation for learning.
        
        Args:
            item_id: Item that was recommended
            reward: Observed reward (e.g., click=1, no_click=0)
        """
        if item_id not in self.item_pulls:
            self.item_pulls[item_id] = 0
            self.item_rewards[item_id] = 0
        
        self.item_pulls[item_id] += 1
        self.item_rewards[item_id] += reward
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about exploration behavior.
        
        Returns:
            Dictionary with exploration statistics
        """
        if not self.item_pulls:
            return {'total_items_explored': 0}
        
        pulls_array = list(self.item_pulls.values())
        
        return {
            'total_items_explored': len(self.item_pulls),
            'total_pulls': sum(pulls_array),
            'avg_pulls_per_item': np.mean(pulls_array),
            'max_pulls': max(pulls_array),
            'min_pulls': min(pulls_array),
            'strategy': self.strategy,
            'epsilon': self.epsilon
        }
    
    def reset_statistics(self) -> None:
        """Reset all exploration statistics."""
        self.item_pulls.clear()
        self.item_rewards.clear()
        logger.info("Exploration statistics reset")
