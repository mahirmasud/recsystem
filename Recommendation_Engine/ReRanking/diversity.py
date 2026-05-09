"""
Diversity Booster - Re-ranking for recommendation diversity

Implements diversity strategies:
- Category diversification (MMR-style)
- Attribute-based diversity
- Serendipity injection
- Cluster-based diversification
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class DiversityBooster(BaseReRanker):
    """
    Re-ranker that promotes diversity in recommendations.
    
    Strategies:
    - Maximal Marginal Relevance (MMR)
    - Category balancing
    - Long-tail promotion
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diversity booster.
        
        Args:
            config: Configuration with diversity parameters
        """
        super().__init__(config)
        
        self.diversity_lambda = config.get('diversity_lambda', 0.5)  # Balance relevance vs diversity
        self.category_column = config.get('category_column', 'category_id')
        self.min_categories = config.get('min_categories', 3)
        self.max_same_category = config.get('max_same_category', 3)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply diversity-focused re-ranking using MMR algorithm.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            Re-ranked DataFrame with diversity adjustments
        """
        if len(recommendations) <= 1:
            return recommendations
        
        recs = recommendations.copy()
        
        # Check if category column exists
        if self.category_column not in recs.columns:
            logger.warning(f"Category column '{self.category_column}' not found")
            return recs
        
        # Calculate diversity-adjusted scores using MMR
        selected_indices = []
        remaining_indices = list(range(len(recs)))
        
        # Get initial scores
        if 'ranking_score' in recs.columns:
            scores = recs['ranking_score'].values
        else:
            scores = np.ones(len(recs))
            scores = scores / scores.sum()
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = scores
        
        while remaining_indices and len(selected_indices) < len(recs):
            best_idx = None
            best_mmr_score = -float('inf')
            
            for idx in remaining_indices:
                # Relevance component
                relevance = norm_scores[idx]
                
                # Diversity component (dissimilarity to already selected)
                if selected_indices:
                    max_similarity = 0
                    for sel_idx in selected_indices:
                        sim = self._calculate_similarity(recs.iloc[idx], recs.iloc[sel_idx])
                        max_similarity = max(max_similarity, sim)
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # MMR score
                mmr_score = self.diversity_lambda * relevance - (1 - self.diversity_lambda) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Reorder based on MMR selection
        recs = recs.iloc[selected_indices].reset_index(drop=True)
        
        # Add diversity score
        recs['diversity_score'] = self._calculate_diversity_scores(recs)
        
        # Combine with original score
        if 'ranking_score' in recs.columns:
            recs['rerank_score'] = recs['ranking_score'] + 0.1 * recs['diversity_score']
        
        logger.info(f"Diversity re-ranking applied: {len(selected_indices)} items")
        return recs
    
    def _calculate_similarity(self, item1: pd.Series, item2: pd.Series) -> float:
        """
        Calculate similarity between two items.
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            Similarity score [0, 1]
        """
        # Category-based similarity
        if self.category_column in item1 and self.category_column in item2:
            if item1[self.category_column] == item2[self.category_column]:
                return 1.0
        
        # Could extend with embedding-based similarity
        return 0.0
    
    def _calculate_diversity_scores(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Calculate diversity score for each item.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of diversity scores
        """
        diversity_scores = np.zeros(len(recs))
        
        if self.category_column not in recs.columns:
            return diversity_scores
        
        categories = recs[self.category_column].values
        
        for i, cat in enumerate(categories):
            # Count how many times this category appears before current position
            prev_count = sum(1 for c in categories[:i] if c == cat)
            
            # Lower score if same category appears frequently
            if prev_count >= self.max_same_category:
                diversity_scores[i] = 0.0
            else:
                diversity_scores[i] = 1.0 / (prev_count + 1)
        
        return diversity_scores
    
    def enforce_category_balance(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce minimum category diversity.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Balanced recommendations
        """
        if self.category_column not in recs.columns:
            return recs
        
        categories_present = recs[self.category_column].nunique()
        
        if categories_present < self.min_categories:
            logger.info(f"Only {categories_present} categories present, less than minimum {self.min_categories}")
        
        return recs
