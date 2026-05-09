"""
Customer Value - Calculates Customer Lifetime Value (CLV) and related metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CustomerValueBuilder:
    """Calculates customer lifetime value and related metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("CustomerValueBuilder initialized")
    
    def calculate_clv(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value using RFM-based approach.
        
        Args:
            user_features: Dataframe with user features including recency, frequency, monetary
            
        Returns:
            Dataframe with CLV scores
        """
        logger.info("Calculating Customer Lifetime Value")
        
        clv_features = user_features[['user_id']].copy()
        
        # Get RFM components if available
        recency_col = 'recency_days' if 'recency_days' in user_features.columns else None
        frequency_col = 'total_transactions' if 'total_transactions' in user_features.columns else None
        monetary_col = 'total_revenue' if 'total_revenue' in user_features.columns else None
        
        # Simple CLV calculation: Frequency * Monetary * Recency factor
        if all([recency_col, frequency_col, monetary_col]):
            # Normalize each component
            freq_norm = user_features[frequency_col] / (user_features[frequency_col].max() + 1e-6)
            monet_norm = user_features[monetary_col] / (user_features[monetary_col].max() + 1e-6)
            
            # Recency factor (more recent = higher score)
            max_recency = user_features[recency_col].max()
            recency_factor = 1 - (user_features[recency_col] / (max_recency + 1e-6))
            
            # CLV score (0-1 range)
            clv_features['clv_score'] = freq_norm * monet_norm * recency_factor
            
            # Predicted future value (simplified)
            avg_order_value = user_features[monetary_col] / (user_features[frequency_col] + 1e-6)
            purchase_frequency_per_year = 365 / (user_features.get('avg_days_between_transactions', 365) + 1e-6)
            
            # Assume 2-year horizon
            clv_features['predicted_clv_2yr'] = avg_order_value * purchase_frequency_per_year * 2
            
        else:
            # Fallback: use available monetary metrics
            if monetary_col:
                clv_features['clv_score'] = user_features[monetary_col] / (user_features[monetary_col].max() + 1e-6)
                clv_features['predicted_clv_2yr'] = user_features[monetary_col] * 2
            else:
                clv_features['clv_score'] = 0.0
                clv_features['predicted_clv_2yr'] = 0.0
        
        # CLV segments
        clv_features['clv_segment'] = pd.cut(
            clv_features['clv_score'],
            bins=[-float('inf'), 0.2, 0.4, 0.6, 0.8, float('inf')],
            labels=['low', 'medium_low', 'medium', 'medium_high', 'high']
        )
        
        return clv_features
