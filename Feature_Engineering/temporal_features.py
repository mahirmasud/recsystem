"""
Temporal Features - Builds time-based features for recommendation models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureBuilder:
    """Builds temporal features for users, items, and interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("TemporalFeatureBuilder initialized")
    
    def build_user_temporal_features(self, transactions_df: pd.DataFrame, 
                                      reference_date: datetime) -> pd.DataFrame:
        """Build user-level temporal features."""
        logger.info("Building user temporal features")
        
        # Group by user
        user_temporal = transactions_df.groupby('user_id').agg(
            first_transaction_date=('transaction_date', 'min'),
            last_transaction_date=('transaction_date', 'max'),
        ).reset_index()
        
        # Calculate recency and tenure
        user_temporal['days_since_last_purchase'] = (
            reference_date - user_temporal['last_transaction_date']
        ).dt.days
        
        user_temporal['customer_tenure_days'] = (
            user_temporal['last_transaction_date'] - user_temporal['first_transaction_date']
        ).dt.days
        
        # Time-based activity patterns
        transactions_df['hour'] = transactions_df['transaction_date'].dt.hour
        transactions_df['day_of_week'] = transactions_df['transaction_date'].dt.dayofweek
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        transactions_df['is_weekend'] = transactions_df['day_of_week'].isin([5, 6]).astype(int)
        
        hour_agg = transactions_df.groupby('user_id')['hour'].agg(
            ['mean', 'std', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12]
        ).reset_index()
        hour_agg.columns = ['user_id', 'avg_transaction_hour', 'std_transaction_hour', 'peak_transaction_hour']
        
        dow_agg = transactions_df.groupby('user_id')['day_of_week'].agg(
            ['mean', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3]
        ).reset_index()
        dow_agg.columns = ['user_id', 'avg_transaction_dow', 'peak_transaction_dow']
        
        # Merge all temporal features
        user_temporal = user_temporal.merge(hour_agg, on='user_id', how='left')
        user_temporal = user_temporal.merge(dow_agg, on='user_id', how='left')
        
        # Weekend preference
        weekend_pref = transactions_df.groupby('user_id')['is_weekend'].mean().reset_index()
        weekend_pref.columns = ['user_id', 'weekend_purchase_ratio']
        user_temporal = user_temporal.merge(weekend_pref, on='user_id', how='left')
        
        # Seasonal features
        month_agg = transactions_df.groupby('user_id')['month'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 6
        ).reset_index()
        month_agg.columns = ['user_id', 'peak_purchase_month']
        user_temporal = user_temporal.merge(month_agg, on='user_id', how='left')
        
        # Fill NaN
        user_temporal = user_temporal.fillna(0)
        
        return user_temporal
    
    def build_item_temporal_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build item-level temporal features."""
        logger.info("Building item temporal features")
        
        item_temporal = transactions_df.groupby('item_id').agg(
            first_sale_date=('transaction_date', 'min'),
            last_sale_date=('transaction_date', 'max'),
        ).reset_index()
        
        # Calculate item age and freshness
        item_temporal['item_age_days'] = (
            item_temporal['last_sale_date'] - item_temporal['first_sale_date']
        ).dt.days
        
        item_temporal['days_since_last_sale'] = (
            datetime.now() - item_temporal['last_sale_date']
        ).dt.days
        
        # Sales velocity
        item_temporal['sales_velocity'] = (
            transactions_df.groupby('item_id')['transaction_id'].count() / 
            item_temporal['item_age_days'].replace(0, 1)
        ).values
        
        # Fill NaN
        item_temporal = item_temporal.fillna(0)
        
        return item_temporal
    
    def build_interaction_temporal_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction-level temporal features."""
        logger.info("Building interaction temporal features")
        
        features = interactions_df.copy()
        
        # Extract time components
        features['interaction_hour'] = features['interaction_timestamp'].dt.hour
        features['interaction_day_of_week'] = features['interaction_timestamp'].dt.dayofweek
        features['interaction_month'] = features['interaction_timestamp'].dt.month
        features['interaction_is_weekend'] = features['interaction_day_of_week'].isin([5, 6]).astype(int)
        
        # Session duration (if available)
        if 'duration_seconds' in features.columns:
            features['duration_minutes'] = features['duration_seconds'] / 60
            features['is_long_session'] = (features['duration_seconds'] > 300).astype(int)
        
        return features
