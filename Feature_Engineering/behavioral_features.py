"""
Behavioral Features - Builds user behavior patterns from interaction data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralFeatureBuilder:
    """Builds behavioral features from user interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("BehavioralFeatureBuilder initialized")
    
    def build_behavioral_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive behavioral features from interactions."""
        logger.info("Building behavioral features")
        
        # Aggregate by user
        behavioral = interactions_df.groupby('user_id').agg(
            total_interactions=('interaction_id', 'count'),
            unique_items_viewed=('item_id', 'nunique'),
            avg_interaction_duration=('duration_seconds', 'mean') if 'duration_seconds' in interactions_df.columns else ('interaction_id', 'count'),
            total_session_time=('duration_seconds', 'sum') if 'duration_seconds' in interactions_df.columns else ('interaction_id', 'count'),
        ).reset_index()
        
        # Interaction type breakdown
        if 'interaction_type' in interactions_df.columns:
            type_counts = interactions_df.pivot_table(
                index='user_id',
                columns='interaction_type',
                values='interaction_id',
                aggfunc='count',
                fill_value=0
            ).reset_index()
            
            # Rename columns
            type_cols = [col for col in type_counts.columns if col != 'user_id']
            type_counts.columns = ['user_id'] + [f'interactions_{col}' for col in type_cols]
            
            behavioral = behavioral.merge(type_counts, on='user_id', how='left')
            
            # Calculate interaction ratios
            total = behavioral['total_interactions'].replace(0, 1)
            for col in type_cols:
                behavioral[f'ratio_{col}'] = behavioral[f'interactions_{col}'] / total
        
        # Device/platform breakdown
        if 'device_type' in interactions_df.columns:
            device_counts = interactions_df.pivot_table(
                index='user_id',
                columns='device_type',
                values='interaction_id',
                aggfunc='count',
                fill_value=0
            ).reset_index()
            
            device_cols = [col for col in device_counts.columns if col != 'user_id']
            device_counts.columns = ['user_id'] + [f'device_{col}' for col in device_cols]
            
            behavioral = behavioral.merge(device_counts, on='user_id', how='left')
        
        # Engagement score
        behavioral['engagement_score'] = (
            behavioral['total_interactions'] * 0.3 +
            behavioral['unique_items_viewed'] * 0.4 +
            (behavioral.get('avg_interaction_duration', 0) / 60).fillna(0) * 0.3
        )
        
        # Fill NaN
        behavioral = behavioral.fillna(0)
        
        return behavioral
