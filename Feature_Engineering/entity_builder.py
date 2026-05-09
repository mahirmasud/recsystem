"""
Entity Builder - Constructs entity-level feature tables from standardized data.

Creates base entity tables for users, items, and interactions with initial features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EntityBuilder:
    """Builds entity-level feature tables from standardized datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EntityBuilder.
        
        Args:
            config: Configuration dictionary with entity definitions
        """
        self.config = config
        self.entities = {}
        logger.info("EntityBuilder initialized")
    
    def build_user_entity(self, transactions_df: pd.DataFrame, 
                          interactions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build user entity table with base features.
        
        Args:
            transactions_df: Standardized transactions dataframe
            interactions_df: Optional interactions dataframe
            
        Returns:
            User entity dataframe with base features
        """
        logger.info("Building user entity table")
        
        # Aggregate by user
        user_agg = transactions_df.groupby('user_id').agg(
            first_transaction_date=('transaction_date', 'min'),
            last_transaction_date=('transaction_date', 'max'),
            total_transactions=('transaction_id', 'count'),
            total_revenue=('net_sales', 'sum'),
            avg_transaction_value=('net_sales', 'mean'),
            std_transaction_value=('net_sales', 'std'),
            min_transaction_value=('net_sales', 'min'),
            max_transaction_value=('net_sales', 'max'),
            unique_items_purchased=('item_id', 'nunique'),
            unique_categories=('category_id', 'nunique'),
        ).reset_index()
        
        # Add interaction features if available
        if interactions_df is not None:
            interaction_agg = interactions_df.groupby('user_id').agg(
                total_interactions=('interaction_id', 'count'),
                unique_items_interacted=('item_id', 'nunique'),
                avg_interaction_duration=('duration_seconds', 'mean'),
            ).reset_index()
            
            user_agg = user_agg.merge(interaction_agg, on='user_id', how='left')
        
        # Fill NaN values
        user_agg = user_agg.fillna(0)
        
        # Calculate derived features
        user_agg['customer_tenure_days'] = (
            user_agg['last_transaction_date'] - user_agg['first_transaction_date']
        ).dt.days
        
        user_agg['avg_days_between_transactions'] = np.where(
            user_agg['total_transactions'] > 1,
            user_agg['customer_tenure_days'] / (user_agg['total_transactions'] - 1),
            0
        )
        
        self.entities['users'] = user_agg
        logger.info(f"User entity built with {len(user_agg)} records")
        
        return user_agg
    
    def build_item_entity(self, transactions_df: pd.DataFrame,
                          categories_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build item entity table with base features.
        
        Args:
            transactions_df: Standardized transactions dataframe
            categories_df: Optional categories dataframe
            
        Returns:
            Item entity dataframe with base features
        """
        logger.info("Building item entity table")
        
        # Aggregate by item
        item_agg = transactions_df.groupby('item_id').agg(
            first_sale_date=('transaction_date', 'min'),
            last_sale_date=('transaction_date', 'max'),
            total_sales=('quantity', 'sum'),
            total_revenue=('net_sales', 'sum'),
            total_transactions=('transaction_id', 'nunique'),
            unique_customers=('user_id', 'nunique'),
            avg_price=('unit_price', 'mean'),
            avg_discount=('discount_amount', 'mean'),
        ).reset_index()
        
        # Add category info if available
        if categories_df is not None:
            item_agg = item_agg.merge(
                categories_df[['item_id', 'category_name', 'category_level_1', 'category_level_2']],
                on='item_id',
                how='left'
            )
        
        # Fill NaN values
        item_agg = item_agg.fillna(0)
        
        # Calculate derived features
        item_agg['popularity_score'] = (
            item_agg['total_sales'] * item_agg['unique_customers']
        ) / item_agg['total_sales'].max()
        
        self.entities['items'] = item_agg
        logger.info(f"Item entity built with {len(item_agg)} records")
        
        return item_agg
    
    def build_interaction_entity(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build interaction entity table.
        
        Args:
            interactions_df: Standardized interactions dataframe
            
        Returns:
            Interaction entity dataframe
        """
        logger.info("Building interaction entity table")
        
        # Keep interactions as-is with basic transformations
        interaction_df = interactions_df.copy()
        
        # Create interaction type encoding
        if 'interaction_type' in interaction_df.columns:
            interaction_df['interaction_type_encoded'] = interaction_df['interaction_type'].astype('category').cat.codes
        
        self.entities['interactions'] = interaction_df
        logger.info(f"Interaction entity built with {len(interaction_df)} records")
        
        return interaction_df
    
    def get_entity(self, entity_name: str) -> Optional[pd.DataFrame]:
        """Get a specific entity dataframe."""
        return self.entities.get(entity_name)
    
    def get_all_entities(self) -> Dict[str, pd.DataFrame]:
        """Get all built entities."""
        return self.entities
    
    def save_entities(self, output_dir: str) -> None:
        """
        Save all entities to parquet files.
        
        Args:
            output_dir: Directory to save parquet files
        """
        for entity_name, df in self.entities.items():
            filepath = f"{output_dir}/{entity_name}_entity.parquet"
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {entity_name} entity to {filepath}")
