"""
Business Booster - General business rule re-ranking

Implements comprehensive business-aware re-ranking:
- Campaign-based boosting
- Seasonal adjustments
- Inventory-level considerations
- Strategic goal alignment
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .reranker import BaseReRanker

logger = logging.getLogger(__name__)


class BusinessBooster(BaseReRanker):
    """
    Re-ranker that applies general business rules and strategies.
    
    Strategies:
    - Campaign-based score adjustments
    - Inventory-aware filtering
    - Strategic category promotion
    - Goal-driven optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize business booster.
        
        Args:
            config: Configuration with business parameters
        """
        super().__init__(config)
        
        self.campaign_rules = config.get('campaign_rules', [])
        self.inventory_column = config.get('inventory_column', 'stock_quantity')
        self.min_inventory_threshold = config.get('min_inventory_threshold', 0)
        self.category_boosts = config.get('category_boosts', {})
        self.strategic_categories = config.get('strategic_categories', [])
        self.category_column = config.get('category_column', 'category_id')
        
    def rerank(
        self,
        recommendations: pd.DataFrame,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply business rule-based re-ranking.
        
        Args:
            recommendations: DataFrame with items and ranking scores
            user_id: Optional user ID
            context: Optional context (may include campaign info)
            
        Returns:
            Re-ranked DataFrame with business adjustments
        """
        if len(recommendations) == 0:
            return recommendations
        
        recs = recommendations.copy()
        
        # Apply inventory filtering
        recs = self._filter_out_of_stock(recs)
        
        if len(recs) == 0:
            logger.warning("All items filtered out due to inventory constraints")
            return recommendations  # Return original if all filtered
        
        # Calculate business adjustment scores
        adjustment_scores = np.zeros(len(recs))
        
        # Apply campaign rules
        campaign_adjustments = self._apply_campaign_rules(recs, context)
        adjustment_scores += campaign_adjustments
        
        # Apply category boosts
        category_adjustments = self._apply_category_boosts(recs)
        adjustment_scores += category_adjustments
        
        # Apply strategic category promotion
        strategic_adjustments = self._apply_strategic_promotion(recs)
        adjustment_scores += strategic_adjustments
        
        # Add adjustment score to recommendations
        recs['business_adjustment'] = adjustment_scores
        
        # Adjust final scores
        if 'ranking_score' in recs.columns:
            score_range = recs['ranking_score'].max() - recs['ranking_score'].min()
            if score_range > 0:
                normalized_adjustment = adjustment_scores * score_range * self.weight
            else:
                normalized_adjustment = adjustment_scores * self.weight
            
            recs['rerank_score'] = recs['ranking_score'] + normalized_adjustment
        
        # Sort by adjusted score
        score_col = 'rerank_score' if 'rerank_score' in recs.columns else 'ranking_score'
        if score_col in recs.columns:
            recs = recs.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        logger.info(f"Business re-ranking applied: {len(recs)} items")
        return recs
    
    def _filter_out_of_stock(self, recs: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out items that are out of stock.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Filtered DataFrame
        """
        if self.inventory_column not in recs.columns:
            return recs
        
        in_stock = recs[self.inventory_column] >= self.min_inventory_threshold
        filtered_count = (~in_stock).sum()
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} out-of-stock items")
        
        return recs[in_stock].copy()
    
    def _apply_campaign_rules(
        self,
        recs: pd.DataFrame,
        context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Apply active campaign rules.
        
        Args:
            recs: Recommendations DataFrame
            context: Context with campaign information
            
        Returns:
            Array of campaign adjustments
        """
        adjustments = np.zeros(len(recs))
        
        if not self.campaign_rules:
            return adjustments
        
        # Get active campaigns from context
        active_campaigns = []
        if context and 'active_campaigns' in context:
            active_campaigns = context['active_campaigns']
        elif self.campaign_rules:
            # Use all configured campaigns if no context
            active_campaigns = [r.get('name') for r in self.campaign_rules]
        
        for rule in self.campaign_rules:
            campaign_name = rule.get('name')
            if campaign_name not in active_campaigns:
                continue
            
            boost_items = rule.get('items', [])
            boost_categories = rule.get('categories', [])
            boost_value = rule.get('boost', 0.1)
            
            for idx, row in recs.iterrows():
                should_boost = False
                
                # Check item-based rules
                if 'item_id' in row and row['item_id'] in boost_items:
                    should_boost = True
                
                # Check category-based rules
                if self.category_column in row and row[self.category_column] in boost_categories:
                    should_boost = True
                
                if should_boost:
                    adjustments[idx] += boost_value
        
        return adjustments
    
    def _apply_category_boosts(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Apply category-specific boosts.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of category adjustments
        """
        adjustments = np.zeros(len(recs))
        
        if not self.category_boosts or self.category_column not in recs.columns:
            return adjustments
        
        for idx, row in recs.iterrows():
            category = row[self.category_column]
            if category in self.category_boosts:
                adjustments[idx] += self.category_boosts[category]
        
        return adjustments
    
    def _apply_strategic_promotion(self, recs: pd.DataFrame) -> np.ndarray:
        """
        Apply strategic category promotion.
        
        Args:
            recs: Recommendations DataFrame
            
        Returns:
            Array of strategic adjustments
        """
        adjustments = np.zeros(len(recs))
        
        if not self.strategic_categories or self.category_column not in recs.columns:
            return adjustments
        
        strategic_set = set(self.strategic_categories)
        
        for idx, row in recs.iterrows():
            if row[self.category_column] in strategic_set:
                adjustments[idx] += 0.2  # Strategic boost value
        
        return adjustments
    
    def add_campaign_rule(
        self,
        name: str,
        items: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        boost: float = 0.1
    ) -> None:
        """
        Add a new campaign rule dynamically.
        
        Args:
            name: Campaign name
            items: List of item IDs to boost
            categories: List of categories to boost
            boost: Boost value to apply
        """
        rule = {
            'name': name,
            'items': items or [],
            'categories': categories or [],
            'boost': boost
        }
        self.campaign_rules.append(rule)
        logger.info(f"Added campaign rule: {name}")
    
    def set_strategic_categories(self, categories: List[str]) -> None:
        """
        Set strategic categories for promotion.
        
        Args:
            categories: List of strategic category IDs
        """
        self.strategic_categories = categories
        logger.info(f"Set strategic categories: {categories}")
