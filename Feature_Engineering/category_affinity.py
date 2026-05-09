"""
Category Affinity - Builds category preference features for users and items.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CategoryAffinityBuilder:
    """Builds category affinity features for recommendation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("CategoryAffinityBuilder initialized")
    
    def build_user_category_affinity(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build user category affinity features."""
        logger.info("Building user category affinity features")
        
        # Get top categories per user
        user_cat_counts = transactions_df.groupby(['user_id', 'category_id']).size().reset_index(name='count')
        
        # Calculate total purchases per user
        user_totals = user_cat_counts.groupby('user_id')['count'].sum().reset_index()
        user_totals.columns = ['user_id', 'total_purchases']
        
        # Merge to get ratios
        user_cat_ratios = user_cat_counts.merge(user_totals, on='user_id')
        user_cat_ratios['category_ratio'] = user_cat_ratios['count'] / user_cat_ratios['total_purchases']
        
        # Pivot to get category features per user
        category_features = user_cat_ratios.pivot_table(
            index='user_id',
            columns='category_id',
            values='category_ratio',
            fill_value=0
        ).reset_index()
        
        # Rename columns
        category_cols = [col for col in category_features.columns if col != 'user_id']
        category_features.columns = ['user_id'] + [f'cat_affinity_{col}' for col in category_cols]
        
        # Get top category per user
        top_cat = user_cat_counts.loc[user_cat_counts.groupby('user_id')['count'].idxmax()]
        top_cat = top_cat[['user_id', 'category_id']].copy()
        top_cat.columns = ['user_id', 'top_category_id']
        
        category_features = category_features.merge(top_cat, on='user_id', how='left')
        
        # Category diversity
        cat_diversity = user_cat_counts.groupby('user_id').agg(
            unique_categories=('category_id', 'nunique'),
        ).reset_index()
        cat_diversity['category_diversity_score'] = np.log1p(cat_diversity['unique_categories'])
        
        category_features = category_features.merge(cat_diversity, on='user_id', how='left')
        
        # Fill NaN
        category_features = category_features.fillna(0)
        
        return category_features
    
    def build_item_category_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build item category-based features."""
        logger.info("Building item category features")
        
        # Category-level aggregations for items
        item_cat_stats = transactions_df.groupby(['item_id', 'category_id']).agg(
            total_sold=('quantity', 'sum'),
            total_revenue=('net_sales', 'sum'),
            unique_buyers=('user_id', 'nunique'),
        ).reset_index()
        
        # Since each item typically belongs to one category, take first
        item_features = item_cat_stats.drop_duplicates(subset=['item_id'])
        
        # Fill NaN
        item_features = item_features.fillna(0)
        
        return item_features
