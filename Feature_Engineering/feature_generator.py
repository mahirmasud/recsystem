"""
Feature Generator - Main orchestrator for feature generation pipeline.

Coordinates all feature builders and manages the complete feature engineering workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import os

from .entity_builder import EntityBuilder
from .aggregation_features import AggregationFeatureBuilder
from .temporal_features import TemporalFeatureBuilder
from .behavioral_features import BehavioralFeatureBuilder
from .recency_features import RecencyFeatureBuilder
from .frequency_features import FrequencyFeatureBuilder
from .monetary_features import MonetaryFeatureBuilder
from .category_affinity import CategoryAffinityBuilder
from .customer_value import CustomerValueBuilder
from .feature_store import FeatureStore
from .feature_versioning import FeatureVersioning

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Main orchestrator for generating ML-ready features."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "output/features"):
        """
        Initialize FeatureGenerator.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save generated features
        """
        self.config = config
        self.output_dir = output_dir
        self.feature_store = FeatureStore(config)
        self.feature_versioning = FeatureVersioning(output_dir)
        
        # Initialize feature builders
        self.entity_builder = EntityBuilder(config)
        self.agg_builder = AggregationFeatureBuilder(config)
        self.temporal_builder = TemporalFeatureBuilder(config)
        self.behavioral_builder = BehavioralFeatureBuilder(config)
        self.recency_builder = RecencyFeatureBuilder(config)
        self.frequency_builder = FrequencyFeatureBuilder(config)
        self.monetary_builder = MonetaryFeatureBuilder(config)
        self.category_builder = CategoryAffinityBuilder(config)
        self.clv_builder = CustomerValueBuilder(config)
        
        logger.info("FeatureGenerator initialized")
    
    def generate_all_features(self, 
                              transactions_df: pd.DataFrame,
                              interactions_df: Optional[pd.DataFrame] = None,
                              items_df: Optional[pd.DataFrame] = None,
                              users_df: Optional[pd.DataFrame] = None,
                              reference_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate all feature sets for users, items, and interactions.
        
        Args:
            transactions_df: Standardized transactions dataframe
            interactions_df: Optional standardized interactions dataframe
            items_df: Optional items dataframe
            users_df: Optional users dataframe
            reference_date: Reference date for recency calculations
            
        Returns:
            Dictionary of feature dataframes by entity type
        """
        logger.info("Starting full feature generation pipeline")
        
        if reference_date is None:
            reference_date = datetime.now()
        
        # Build base entities
        logger.info("Building base entities...")
        user_entity = self.entity_builder.build_user_entity(
            transactions_df, interactions_df
        )
        item_entity = self.entity_builder.build_item_entity(
            transactions_df, items_df
        )
        
        if interactions_df is not None:
            interaction_entity = self.entity_builder.build_interaction_entity(interactions_df)
        
        # Generate user features
        logger.info("Generating user features...")
        user_features = self._generate_user_features(
            transactions_df, interactions_df, user_entity, reference_date
        )
        
        # Generate item features
        logger.info("Generating item features...")
        item_features = self._generate_item_features(
            transactions_df, item_entity
        )
        
        # Generate interaction features
        interaction_features = None
        if interactions_df is not None:
            logger.info("Generating interaction features...")
            interaction_features = self._generate_interaction_features(
                interactions_df, interaction_entity
            )
        
        # Store features
        self.feature_store.store_features('users', user_features)
        self.feature_store.store_features('items', item_features)
        if interaction_features is not None:
            self.feature_store.store_features('interactions', interaction_features)
        
        # Save to parquet
        self._save_features(user_features, item_features, interaction_features)
        
        # Create version
        version_id = self.feature_versioning.create_version({
            'users': user_features,
            'items': item_features,
            'interactions': interaction_features
        })
        
        logger.info(f"Feature generation complete. Version: {version_id}")
        
        return {
            'users': user_features,
            'items': item_features,
            'interactions': interaction_features
        }
    
    def _generate_user_features(self, 
                                 transactions_df: pd.DataFrame,
                                 interactions_df: Optional[pd.DataFrame],
                                 user_entity: pd.DataFrame,
                                 reference_date: datetime) -> pd.DataFrame:
        """Generate comprehensive user features."""
        
        features = user_entity.copy()
        
        # Aggregation features
        agg_features = self.agg_builder.build_user_aggregations(transactions_df)
        features = features.merge(agg_features, on='user_id', how='left')
        
        # Temporal features
        temporal_features = self.temporal_builder.build_user_temporal_features(
            transactions_df, reference_date
        )
        features = features.merge(temporal_features, on='user_id', how='left')
        
        # Behavioral features
        if interactions_df is not None:
            behavioral_features = self.behavioral_builder.build_behavioral_features(
                interactions_df
            )
            features = features.merge(behavioral_features, on='user_id', how='left')
        
        # RFM features
        recency_features = self.recency_builder.build_recency_features(
            transactions_df, reference_date
        )
        features = features.merge(recency_features, on='user_id', how='left')
        
        frequency_features = self.frequency_builder.build_frequency_features(
            transactions_df
        )
        features = features.merge(frequency_features, on='user_id', how='left')
        
        monetary_features = self.monetary_builder.build_monetary_features(
            transactions_df
        )
        features = features.merge(monetary_features, on='user_id', how='left')
        
        # Category affinity
        category_affinity = self.category_builder.build_user_category_affinity(
            transactions_df
        )
        features = features.merge(category_affinity, on='user_id', how='left')
        
        # CLV
        clv_features = self.clv_builder.calculate_clv(features)
        features = features.merge(clv_features, on='user_id', how='left')
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _generate_item_features(self,
                                 transactions_df: pd.DataFrame,
                                 item_entity: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive item features."""
        
        features = item_entity.copy()
        
        # Aggregation features
        agg_features = self.agg_builder.build_item_aggregations(transactions_df)
        features = features.merge(agg_features, on='item_id', how='left')
        
        # Temporal features
        temporal_features = self.temporal_builder.build_item_temporal_features(
            transactions_df
        )
        features = features.merge(temporal_features, on='item_id', how='left')
        
        # Category affinity (item perspective)
        category_features = self.category_builder.build_item_category_features(
            transactions_df
        )
        features = features.merge(category_features, on='item_id', how='left')
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _generate_interaction_features(self,
                                        interactions_df: pd.DataFrame,
                                        interaction_entity: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction-level features."""
        
        features = interaction_entity.copy()
        
        # Add temporal features
        temporal_features = self.temporal_builder.build_interaction_temporal_features(
            interactions_df
        )
        features = features.merge(temporal_features, on='interaction_id', how='left')
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _save_features(self, 
                       user_features: pd.DataFrame,
                       item_features: pd.DataFrame,
                       interaction_features: Optional[pd.DataFrame]) -> None:
        """Save feature tables to parquet files."""
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save user features
        user_path = os.path.join(self.output_dir, 'user_features.parquet')
        user_features.to_parquet(user_path, index=False)
        logger.info(f"Saved user features to {user_path}")
        
        # Save item features
        item_path = os.path.join(self.output_dir, 'item_features.parquet')
        item_features.to_parquet(item_path, index=False)
        logger.info(f"Saved item features to {item_path}")
        
        # Save interaction features
        if interaction_features is not None:
            interaction_path = os.path.join(self.output_dir, 'interaction_features.parquet')
            interaction_features.to_parquet(interaction_path, index=False)
            logger.info(f"Saved interaction features to {interaction_path}")
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about generated features."""
        return self.feature_store.get_feature_metadata()
