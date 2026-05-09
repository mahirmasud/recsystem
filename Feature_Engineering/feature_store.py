"""
Feature Store - Centralized storage and retrieval of features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureStore:
    """Centralized feature storage with metadata tracking."""
    
    def __init__(self, config: Dict[str, Any], store_path: str = "output/features"):
        self.config = config
        self.store_path = store_path
        self.feature_metadata = {}
        self.stored_features = {}
        
        # Initialize metadata file
        self.metadata_file = os.path.join(store_path, 'feature_metadata.json')
        self._load_metadata()
        
        logger.info("FeatureStore initialized")
    
    def _load_metadata(self) -> None:
        """Load existing feature metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.feature_metadata = json.load(f)
            logger.info(f"Loaded feature metadata from {self.metadata_file}")
    
    def _save_metadata(self) -> None:
        """Save feature metadata to file."""
        os.makedirs(self.store_path, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2, default=str)
        logger.info(f"Saved feature metadata to {self.metadata_file}")
    
    def store_features(self, entity_type: str, features_df: pd.DataFrame,
                       version: Optional[str] = None) -> None:
        """
        Store features for an entity type.
        
        Args:
            entity_type: Type of entity (users, items, interactions)
            features_df: DataFrame containing features
            version: Optional version identifier
        """
        logger.info(f"Storing {entity_type} features")
        
        # Store in memory
        self.stored_features[entity_type] = features_df
        
        # Update metadata
        timestamp = datetime.now().isoformat()
        feature_cols = [col for col in features_df.columns if col != entity_type.rstrip('s') + '_id']
        
        self.feature_metadata[entity_type] = {
            'feature_count': len(feature_cols),
            'record_count': len(features_df),
            'features': feature_cols,
            'last_updated': timestamp,
            'version': version or 'latest',
            'dtypes': {col: str(dtype) for col, dtype in features_df.dtypes.items()}
        }
        
        self._save_metadata()
    
    def get_features(self, entity_type: str) -> Optional[pd.DataFrame]:
        """Get stored features for an entity type."""
        return self.stored_features.get(entity_type)
    
    def get_feature_columns(self, entity_type: str) -> List[str]:
        """Get list of feature columns for an entity type."""
        metadata = self.feature_metadata.get(entity_type, {})
        return metadata.get('features', [])
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get all feature metadata."""
        return self.feature_metadata
    
    def get_feature_stats(self, entity_type: str, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific feature."""
        features_df = self.stored_features.get(entity_type)
        if features_df is None or feature_name not in features_df.columns:
            return None
        
        col = features_df[feature_name]
        stats = {
            'mean': float(col.mean()) if pd.api.types.is_numeric_dtype(col) else None,
            'std': float(col.std()) if pd.api.types.is_numeric_dtype(col) else None,
            'min': float(col.min()) if pd.api.types.is_numeric_dtype(col) else None,
            'max': float(col.max()) if pd.api.types.is_numeric_dtype(col) else None,
            'null_count': int(col.isnull().sum()),
            'unique_count': int(col.nunique())
        }
        
        return stats
