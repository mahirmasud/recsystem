"""
Aggregation Features - Builds aggregate features from transaction and interaction data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AggregationFeatureBuilder:
    """Builds aggregation-based features for users and items."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("AggregationFeatureBuilder initialized")
    
    def build_user_aggregations(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build user-level aggregation features."""
        logger.info("Building user aggregation features")
        
        agg_features = transactions_df.groupby('user_id').agg(
            # Count features
            total_transactions=('transaction_id', 'count'),
            total_items_purchased=('quantity', 'sum'),
            
            # Revenue features
            total_revenue=('net_sales', 'sum'),
            avg_transaction_value=('net_sales', 'mean'),
            std_transaction_value=('net_sales', 'std'),
            median_transaction_value=('net_sales', 'median'),
            
            # Item diversity
            unique_items=('item_id', 'nunique'),
            unique_categories=('category_id', 'nunique'),
            unique_brands=('brand_id', 'nunique') if 'brand_id' in transactions_df.columns else ('item_id', 'nunique'),
            
            # Price features
            avg_unit_price=('unit_price', 'mean'),
            min_unit_price=('unit_price', 'min'),
            max_unit_price=('unit_price', 'max'),
            
            # Discount features
            total_discount=('discount_amount', 'sum'),
            avg_discount_per_transaction=('discount_amount', 'mean'),
            discount_rate=('discount_amount', 'sum'),  # Will be normalized later
        ).reset_index()
        
        # Calculate derived aggregations
        agg_features['items_per_transaction'] = (
            agg_features['total_items_purchased'] / agg_features['total_transactions'].replace(0, 1)
        )
        
        agg_features['category_diversity_score'] = (
            agg_features['unique_categories'] / agg_features['unique_items'].replace(0, 1)
        )
        
        # Fill NaN
        agg_features = agg_features.fillna(0)
        
        return agg_features
    
    def build_item_aggregations(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build item-level aggregation features."""
        logger.info("Building item aggregation features")
        
        agg_features = transactions_df.groupby('item_id').agg(
            # Sales features
            total_units_sold=('quantity', 'sum'),
            total_revenue=('net_sales', 'sum'),
            total_transactions=('transaction_id', 'nunique'),
            
            # Customer reach
            unique_customers=('user_id', 'nunique'),
            
            # Price features
            avg_selling_price=('unit_price', 'mean'),
            min_selling_price=('unit_price', 'min'),
            max_selling_price=('unit_price', 'max'),
            
            # Discount features
            avg_discount=('discount_amount', 'mean'),
            total_discount_given=('discount_amount', 'sum'),
            
            # Quantity stats
            avg_quantity_per_transaction=('quantity', 'mean'),
            std_quantity=('quantity', 'std'),
        ).reset_index()
        
        # Calculate derived features
        agg_features['revenue_per_customer'] = (
            agg_features['total_revenue'] / agg_features['unique_customers'].replace(0, 1)
        )
        
        agg_features['purchase_frequency'] = (
            agg_features['total_transactions'] / agg_features['unique_customers'].replace(0, 1)
        )
        
        # Fill NaN
        agg_features = agg_features.fillna(0)
        
        return agg_features
