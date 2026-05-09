"""
Frequency Features - Builds frequency-based features for RFM analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FrequencyFeatureBuilder:
    """Builds frequency features for user behavior analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("FrequencyFeatureBuilder initialized")
    
    def build_frequency_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build frequency features from transaction data."""
        logger.info("Building frequency features")
        
        # Calculate frequency per user
        frequency = transactions_df.groupby('user_id').agg(
            total_transactions=('transaction_id', 'count'),
            unique_items_purchased=('item_id', 'nunique'),
            unique_categories_purchased=('category_id', 'nunique'),
        ).reset_index()
        
        # Transaction frequency score
        max_freq = frequency['total_transactions'].max()
        if max_freq > 0:
            frequency['frequency_score'] = frequency['total_transactions'] / max_freq
        else:
            frequency['frequency_score'] = 0.0
        
        # Frequency bins
        frequency['frequency_bin'] = pd.cut(
            frequency['total_transactions'],
            bins=[-1, 1, 3, 5, 10, 20, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # Purchase rate (transactions per unique item)
        frequency['purchase_rate'] = (
            frequency['total_transactions'] / 
            frequency['unique_items_purchased'].replace(0, 1)
        )
        
        # Category exploration score
        frequency['category_exploration_score'] = (
            frequency['unique_categories_purchased'] /
            frequency['unique_items_purchased'].replace(0, 1)
        )
        
        # Fill NaN
        frequency = frequency.fillna(0)
        
        return frequency
