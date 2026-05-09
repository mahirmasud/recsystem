"""
Recency Features - Builds recency-based features for RFM analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RecencyFeatureBuilder:
    """Builds recency features for user behavior analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("RecencyFeatureBuilder initialized")
    
    def build_recency_features(self, transactions_df: pd.DataFrame,
                                reference_date: datetime) -> pd.DataFrame:
        """Build recency features from transaction data."""
        logger.info("Building recency features")
        
        # Calculate recency per user
        recency = transactions_df.groupby('user_id').agg(
            last_transaction_date=('transaction_date', 'max'),
        ).reset_index()
        
        # Days since last purchase
        recency['recency_days'] = (
            reference_date - recency['last_transaction_date']
        ).dt.days
        
        # Recency score (lower is better)
        max_recency = recency['recency_days'].max()
        if max_recency > 0:
            recency['recency_score'] = 1 - (recency['recency_days'] / max_recency)
        else:
            recency['recency_score'] = 1.0
        
        # Recency bins
        recency['recency_bin'] = pd.cut(
            recency['recency_days'],
            bins=[-1, 7, 30, 90, 180, 365, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # Days since first purchase (for context)
        first_purchase = transactions_df.groupby('user_id').agg(
            first_transaction_date=('transaction_date', 'min'),
        ).reset_index()
        
        recency = recency.merge(first_purchase, on='user_id', how='left')
        recency['days_since_first_purchase'] = (
            reference_date - recency['first_transaction_date']
        ).dt.days
        
        # Fill NaN
        recency = recency.fillna(0)
        
        return recency
