"""
Feature Exporter - Exports features in multiple formats for downstream consumption.

Supports:
- Parquet export (primary format)
- CSV export
- JSON metadata export
- Feature registry export
- Lineage tracking export
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureExporter:
    """Exports features to various formats with metadata."""
    
    def __init__(self, output_dir: str = "output/features"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"FeatureExporter initialized with output_dir={output_dir}")
    
    def export_features(self,
                       features_dict: Dict[str, pd.DataFrame],
                       format: str = 'parquet',
                       prefix: str = '') -> Dict[str, str]:
        """
        Export features to files.
        
        Args:
            features_dict: Dictionary of entity_type -> DataFrame
            format: Export format ('parquet', 'csv')
            prefix: Optional filename prefix
            
        Returns:
            Dictionary of entity_type -> filepath
        """
        logger.info(f"Exporting {len(features_dict)} feature tables in {format} format")
        
        exported_files = {}
        
        for entity_type, df in features_dict.items():
            if df is None:
                continue
            
            filename = f"{prefix}{entity_type}_features"
            filepath = self._save_dataframe(df, filename, format)
            
            if filepath:
                exported_files[entity_type] = filepath
                logger.info(f"Exported {entity_type} features to {filepath}")
        
        return exported_files
    
    def _save_dataframe(self, 
                        df: pd.DataFrame, 
                        filename: str, 
                        format: str) -> Optional[str]:
        """Save dataframe to file."""
        filepath = os.path.join(self.output_dir, f"{filename}.{format}")
        
        try:
            if format == 'parquet':
                df.to_parquet(filepath, index=False)
            elif format == 'csv':
                df.to_csv(filepath, index=False)
            else:
                logger.warning(f"Unknown format: {format}")
                return None
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            return None
    
    def export_metadata(self, 
                       metadata: Dict[str, Any],
                       filename: str = 'feature_metadata.json') -> str:
        """Export feature metadata to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert any non-serializable objects
        serializable_metadata = self._make_serializable(metadata)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, default=str)
        
        logger.info(f"Exported metadata to {filepath}")
        return filepath
    
    def export_feature_registry(self,
                                feature_definitions: Dict[str, List],
                                filename: str = 'feature_registry.json') -> str:
        """
        Export feature registry with definitions.
        
        Args:
            feature_definitions: Dictionary of entity_type -> list of feature defs
            filename: Output filename
            
        Returns:
            Filepath to exported registry
        """
        registry = {
            'export_timestamp': datetime.now().isoformat(),
            'entities': {}
        }
        
        for entity_type, defs in feature_definitions.items():
            entity_registry = {
                'feature_count': len(defs),
                'features': []
            }
            
            for feat_def in defs:
                try:
                    feat_info = {
                        'name': feat_def.get_name() if hasattr(feat_def, 'get_name') else str(feat_def),
                        'description': feat_def.get_description() if hasattr(feat_def, 'get_description') else '',
                        'formula': str(feat_def)
                    }
                    entity_registry['features'].append(feat_info)
                except Exception as e:
                    logger.warning(f"Could not serialize feature definition: {e}")
            
            registry['entities'][entity_type] = entity_registry
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
        
        logger.info(f"Exported feature registry to {filepath}")
        return filepath
    
    def export_lineage(self,
                      lineage_records: List[Dict[str, Any]],
                      filename: str = 'feature_lineage.json') -> str:
        """
        Export feature lineage information.
        
        Args:
            lineage_records: List of lineage records
            filename: Output filename
            
        Returns:
            Filepath to exported lineage
        """
        filepath = os.path.join(self.output_dir, filename)
        
        serializable_records = [self._make_serializable(r) for r in lineage_records]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_records, f, indent=2, default=str)
        
        logger.info(f"Exported feature lineage to {filepath}")
        return filepath
    
    def export_all(self,
                  features_dict: Dict[str, pd.DataFrame],
                  metadata: Dict[str, Any],
                  feature_definitions: Optional[Dict[str, List]] = None,
                  lineage_records: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Export all feature artifacts.
        
        Args:
            features_dict: Features by entity type
            metadata: Feature metadata
            feature_definitions: Optional feature definitions
            lineage_records: Optional lineage records
            
        Returns:
            Dictionary of exported file paths
        """
        exports = {}
        
        # Export features (both parquet and csv)
        exports['parquet'] = self.export_features(features_dict, format='parquet')
        exports['csv'] = self.export_features(features_dict, format='csv')
        
        # Export metadata
        exports['metadata'] = self.export_metadata(metadata)
        
        # Export registry if definitions provided
        if feature_definitions:
            exports['registry'] = self.export_feature_registry(feature_definitions)
        
        # Export lineage if provided
        if lineage_records:
            exports['lineage'] = self.export_lineage(lineage_records)
        
        logger.info(f"Completed full export: {len(exports)} artifact types")
        return exports
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.api.types.is_datetime64_any_dtype(obj) if isinstance(obj, pd.Series) else False:
            return obj.tolist()
        else:
            return obj
    
    def load_features(self, 
                     entity_type: str, 
                     format: str = 'parquet') -> Optional[pd.DataFrame]:
        """Load previously exported features."""
        filename = f"{entity_type}_features.{format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Features file not found: {filepath}")
            return None
        
        try:
            if format == 'parquet':
                df = pd.read_parquet(filepath)
            elif format == 'csv':
                df = pd.read_csv(filepath)
            else:
                return None
            
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None
