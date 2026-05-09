"""
Feature Versioning - Manages feature versions for reproducibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
import os
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureVersioning:
    """Manages feature versions for ML reproducibility."""
    
    def __init__(self, output_dir: str = "output/features"):
        self.output_dir = output_dir
        self.version_dir = os.path.join(output_dir, 'versions')
        self.version_history = []
        self.version_file = os.path.join(self.version_dir, 'version_history.json')
        
        # Create version directory
        os.makedirs(self.version_dir, exist_ok=True)
        
        # Load existing history
        self._load_version_history()
        
        logger.info("FeatureVersioning initialized")
    
    def _load_version_history(self) -> None:
        """Load version history from file."""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                self.version_history = json.load(f)
            logger.info(f"Loaded version history with {len(self.version_history)} versions")
    
    def _save_version_history(self) -> None:
        """Save version history to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.version_history, f, indent=2, default=str)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for change detection."""
        # Use sorted column names and shape for quick comparison
        data_str = f"{df.shape}{sorted(df.columns)}{df.dtypes.to_dict()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def create_version(self, features_dict: Dict[str, pd.DataFrame],
                       description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        Create a new feature version.
        
        Args:
            features_dict: Dictionary of entity_type -> DataFrame
            description: Optional description of this version
            tags: Optional list of tags
            
        Returns:
            Version ID string
        """
        timestamp = datetime.now()
        version_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        version_info = {
            'version_id': version_id,
            'timestamp': timestamp.isoformat(),
            'description': description,
            'tags': tags or [],
            'entities': {}
        }
        
        # Save each entity's features and compute metadata
        for entity_type, df in features_dict.items():
            if df is None:
                continue
                
            filepath = os.path.join(self.version_dir, f"{entity_type}_{version_id}.parquet")
            df.to_parquet(filepath, index=False)
            
            version_info['entities'][entity_type] = {
                'filepath': filepath,
                'record_count': len(df),
                'feature_count': len(df.columns) - 1,  # Exclude ID column
                'hash': self._compute_hash(df),
                'columns': list(df.columns)
            }
        
        self.version_history.append(version_info)
        self._save_version_history()
        
        logger.info(f"Created feature version {version_id}")
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load features from a specific version."""
        version_info = next(
            (v for v in self.version_history if v['version_id'] == version_id),
            None
        )
        
        if not version_info:
            logger.warning(f"Version {version_id} not found")
            return None
        
        features = {}
        for entity_type, info in version_info['entities'].items():
            filepath = info.get('filepath')
            if filepath and os.path.exists(filepath):
                features[entity_type] = pd.read_parquet(filepath)
        
        return features
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest version ID."""
        if not self.version_history:
            return None
        return self.version_history[-1]['version_id']
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available versions."""
        return self.version_history
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """Compare two feature versions."""
        v1 = next((v for v in self.version_history if v['version_id'] == version_id_1), None)
        v2 = next((v for v in self.version_history if v['version_id'] == version_id_2), None)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version_1': version_id_1,
            'version_2': version_id_2,
            'timestamp_diff': (
                datetime.fromisoformat(v2['timestamp']) - 
                datetime.fromisoformat(v1['timestamp'])
            ).total_seconds() / 3600,  # Hours
            'entity_changes': {}
        }
        
        all_entities = set(v1['entities'].keys()) | set(v2['entities'].keys())
        
        for entity in all_entities:
            e1 = v1['entities'].get(entity, {})
            e2 = v2['entities'].get(entity, {})
            
            comparison['entity_changes'][entity] = {
                'record_count_change': e2.get('record_count', 0) - e1.get('record_count', 0),
                'feature_count_change': e2.get('feature_count', 0) - e1.get('feature_count', 0),
                'hash_changed': e1.get('hash') != e2.get('hash')
            }
        
        return comparison
