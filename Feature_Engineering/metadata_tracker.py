"""
Metadata Tracker - Tracks feature metadata, lineage, and versioning information.

Provides:
- Feature lineage tracking
- Schema evolution tracking
- Generation metadata
- Reproducibility support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
import os
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataTracker:
    """Tracks feature metadata for reproducibility and lineage."""
    
    def __init__(self, output_dir: str = "output/features"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(output_dir, 'metadata_tracker.json')
        self.lineage_records = []
        self.schema_versions = {}
        
        self._load_existing_metadata()
        logger.info("MetadataTracker initialized")
    
    def _load_existing_metadata(self):
        """Load existing metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.lineage_records = data.get('lineage', [])
                self.schema_versions = data.get('schema_versions', {})
            logger.info(f"Loaded {len(self.lineage_records)} lineage records")
    
    def _save_metadata(self):
        """Save metadata to file."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'lineage': self.lineage_records,
            'schema_versions': self.schema_versions,
            'total_features_tracked': sum(
                len(r.get('features', [])) for r in self.lineage_records
            )
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def track_feature_generation(self,
                                 entity_type: str,
                                 features_df: pd.DataFrame,
                                 generation_method: str,
                                 source_tables: List[str],
                                 parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a feature generation event.
        
        Args:
            entity_type: Type of entity (users, items, etc.)
            features_df: Generated features DataFrame
            generation_method: Method used (e.g., 'dfs', 'custom', 'aggregation')
            source_tables: List of source table names
            parameters: Optional generation parameters
            
        Returns:
            Lineage record ID
        """
        timestamp = datetime.now()
        record_id = f"{entity_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Compute feature hash for change detection
        feature_hash = self._compute_dataframe_hash(features_df)
        
        # Build feature list with metadata
        features_info = []
        for col in features_df.columns:
            feat_info = {
                'name': col,
                'dtype': str(features_df[col].dtype),
                'null_count': int(features_df[col].isnull().sum()),
                'unique_count': int(features_df[col].nunique()),
            }
            
            if pd.api.types.is_numeric_dtype(features_df[col]):
                feat_info['stats'] = {
                    'mean': float(features_df[col].mean()) if not features_df[col].empty else None,
                    'std': float(features_df[col].std()) if not features_df[col].empty else None,
                    'min': float(features_df[col].min()) if not features_df[col].empty else None,
                    'max': float(features_df[col].max()) if not features_df[col].empty else None,
                }
            
            features_info.append(feat_info)
        
        record = {
            'record_id': record_id,
            'timestamp': timestamp.isoformat(),
            'entity_type': entity_type,
            'generation_method': generation_method,
            'source_tables': source_tables,
            'parameters': parameters or {},
            'feature_count': len(features_df.columns),
            'record_count': len(features_df),
            'feature_hash': feature_hash,
            'features': features_info,
        }
        
        self.lineage_records.append(record)
        self._save_metadata()
        
        logger.info(f"Tracked feature generation: {record_id}")
        return record_id
    
    def track_schema_version(self,
                            table_name: str,
                            schema_info: Dict[str, Any],
                            version: Optional[str] = None) -> str:
        """
        Track a schema version.
        
        Args:
            table_name: Name of the table
            schema_info: Schema information dictionary
            version: Optional explicit version string
            
        Returns:
            Version identifier
        """
        if version is None:
            # Auto-generate version based on content hash
            schema_hash = hashlib.md5(
                json.dumps(schema_info, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            version = f"v{schema_hash}"
        
        version_record = {
            'table_name': table_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'schema_info': schema_info,
            'column_count': len(schema_info.get('columns', {})),
            'row_count': schema_info.get('row_count', 0),
        }
        
        if table_name not in self.schema_versions:
            self.schema_versions[table_name] = []
        
        self.schema_versions[table_name].append(version_record)
        self._save_metadata()
        
        logger.info(f"Tracked schema version {version} for {table_name}")
        return version
    
    def get_lineage_for_entity(self, entity_type: str) -> List[Dict]:
        """Get all lineage records for an entity type."""
        return [r for r in self.lineage_records if r['entity_type'] == entity_type]
    
    def get_latest_lineage(self, entity_type: str) -> Optional[Dict]:
        """Get most recent lineage record for an entity type."""
        records = self.get_lineage_for_entity(entity_type)
        if not records:
            return None
        return max(records, key=lambda r: r['timestamp'])
    
    def compare_feature_versions(self, 
                                record_id_1: str,
                                record_id_2: str) -> Dict[str, Any]:
        """
        Compare two feature generation versions.
        
        Args:
            record_id_1: First record ID
            record_id_2: Second record ID
            
        Returns:
            Comparison report
        """
        rec1 = next((r for r in self.lineage_records if r['record_id'] == record_id_1), None)
        rec2 = next((r for r in self.lineage_records if r['record_id'] == record_id_2), None)
        
        if not rec1 or not rec2:
            return {'error': 'One or both records not found'}
        
        comparison = {
            'record_1': record_id_1,
            'record_2': record_id_2,
            'time_difference_hours': (
                datetime.fromisoformat(rec2['timestamp']) - 
                datetime.fromisoformat(rec1['timestamp'])
            ).total_seconds() / 3600,
            'feature_count_change': rec2['feature_count'] - rec1['feature_count'],
            'record_count_change': rec2['record_count'] - rec1['record_count'],
            'hash_changed': rec1['feature_hash'] != rec2['feature_hash'],
            'method_changed': rec1['generation_method'] != rec2['generation_method'],
        }
        
        return comparison
    
    def get_feature_history(self, feature_name: str, 
                           entity_type: Optional[str] = None) -> List[Dict]:
        """
        Get history of a specific feature across generations.
        
        Args:
            feature_name: Name of the feature
            entity_type: Optional entity type filter
            
        Returns:
            List of records where this feature appeared
        """
        history = []
        
        for record in self.lineage_records:
            if entity_type and record['entity_type'] != entity_type:
                continue
            
            # Check if feature exists in this record
            for feat in record.get('features', []):
                if feat['name'] == feature_name:
                    history.append({
                        'record_id': record['record_id'],
                        'timestamp': record['timestamp'],
                        'entity_type': record['entity_type'],
                        'generation_method': record['generation_method'],
                        'feature_stats': feat.get('stats'),
                    })
                    break
        
        return history
    
    def export_lineage_report(self, output_path: Optional[str] = None) -> str:
        """Export complete lineage report."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'lineage_report.json')
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_records': len(self.lineage_records),
                'entities_tracked': list(set(r['entity_type'] for r in self.lineage_records)),
                'tables_versioned': list(self.schema_versions.keys()),
            },
            'lineage_records': self.lineage_records,
            'schema_versions': self.schema_versions,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported lineage report to {output_path}")
        return output_path
    
    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe structure for change detection."""
        # Use shape, column names, and dtypes for quick comparison
        data_str = f"{df.shape}{sorted(df.columns)}{df.dtypes.to_dict()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall tracking statistics."""
        if not self.lineage_records:
            return {'status': 'no_records'}
        
        entity_counts = {}
        method_counts = {}
        
        for record in self.lineage_records:
            entity = record['entity_type']
            method = record['generation_method']
            
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_records': len(self.lineage_records),
            'entities': entity_counts,
            'methods': method_counts,
            'tables_versioned': len(self.schema_versions),
            'total_features_tracked': sum(
                r['feature_count'] for r in self.lineage_records
            ),
        }
