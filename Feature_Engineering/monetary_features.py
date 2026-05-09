"""
Monetary Features - Builds monetary value features for RFM analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MonetaryFeatureBuilder:
    """Builds monetary features for user behavior analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("MonetaryFeatureBuilder initialized")
    
    def build_monetary_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build monetary features from transaction data."""
        logger.info("Building monetary features")
        
        # Calculate monetary metrics per user
        monetary = transactions_df.groupby('user_id').agg(
            total_revenue=('net_sales', 'sum'),
            avg_transaction_value=('net_sales', 'mean'),
            std_transaction_value=('net_sales', 'std'),
            min_transaction_value=('net_sales', 'min'),
            max_transaction_value=('net_sales', 'max'),
            median_transaction_value=('net_sales', 'median'),
            total_discount_received=('discount_amount', 'sum'),
            avg_discount_per_transaction=('discount_amount', 'mean'),
        ).reset_index()
        
        # Monetary score
        max_revenue = monetary['total_revenue'].max()
        if max_revenue > 0:
            monetary['monetary_score'] = monetary['total_revenue'] / max_revenue
        else:
            monetary['monetary_score'] = 0.0
        
        # Monetary bins
        monetary['monetary_bin'] = pd.cut(
            monetary['total_revenue'],
            bins=[-1, 0, 100, 500, 1000, 5000, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # Spending consistency (lower std = more consistent)
        monetary['spending_consistency'] = 1 / (1 + monetary['std_transaction_value'])
        
        # Discount sensitivity
        monetary['discount_sensitivity'] = (
            monetary['total_discount_received'] / 
            monetary['total_revenue'].replace(0, 1)
        )
        
        # Average items per transaction (monetary perspective)
        item_counts = transactions_df.groupby('user_id')['quantity'].sum().reset_index()
        item_counts.columns = ['user_id', 'total_items']
        trans_counts = transactions_df.groupby('user_id')['transaction_id'].count().reset_index()
        trans_counts.columns = ['user_id', 'total_transactions']
        
        items_per_trans = item_counts.merge(trans_counts, on='user_id')
        items_per_trans['avg_items_per_transaction'] = (
            items_per_trans['total_items'] / 
            items_per_trans['total_transactions'].replace(0, 1)
        )
        
        monetary = monetary.merge(
            items_per_trans[['user_id', 'avg_items_per_transaction']], 
            on='user_id', how='left'
        )
        
        # Fill NaN
        monetary = monetary.fillna(0)
        
        return monetary
