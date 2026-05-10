"""
Dynamic Feature Mapper - Maps standardized schemas to feature engineering logic.

Dynamically adapts feature generation to:
- Varying schemas
- Different entity structures
- Multiple business domains
- Unseen data patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from datetime import datetime

from .schema_analyzer import SchemaAnalyzer
from .primitive_registry import PrimitiveRegistry

logger = logging.getLogger(__name__)


class DynamicFeatureMapper:
    """Maps schemas to appropriate feature engineering strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schema_analyzer = SchemaAnalyzer()
        self.primitive_registry = PrimitiveRegistry()
        self.mappings_cache = {}
        
        # Default column mappings for common patterns
        self.column_patterns = {
            'id_columns': ['id', '_id', 'ID', 'Id'],
            'timestamp_columns': ['date', 'time', 'timestamp', 'created', 'updated', '_at', '_on'],
            'value_columns': ['amount', 'value', 'price', 'cost', 'revenue', 'sales', 'quantity'],
            'category_columns': ['category', 'type', 'status', 'group', 'segment', 'class'],
            'user_columns': ['user', 'customer', 'client', 'member', 'account'],
            'item_columns': ['item', 'product', 'service', 'sku', 'article'],
            'transaction_columns': ['transaction', 'order', 'purchase', 'payment', 'event'],
        }
        
        logger.info("DynamicFeatureMapper initialized")
    
    def analyze_and_map(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Analyze a dataframe and create feature mapping.
        
        Args:
            df: Input DataFrame
            table_name: Name of the table
            
        Returns:
            Mapping dictionary with detected columns and recommendations
        """
        logger.info(f"Analyzing and mapping schema for '{table_name}'")
        
        # Analyze schema
        schema_info = self.schema_analyzer.analyze_dataframe(df, table_name)
        
        # Detect domain
        domain = self.schema_analyzer.detect_domain(schema_info)
        
        # Create column mapping
        column_mapping = self._map_columns(schema_info)
        
        # Generate feature recommendations
        feature_recommendations = self._recommend_features(schema_info, domain)
        
        mapping = {
            'table_name': table_name,
            'domain': domain,
            'schema_info': schema_info,
            'column_mapping': column_mapping,
            'feature_recommendations': feature_recommendations,
            'recommended_primitives': self.primitive_registry.get_primitives_for_domain(domain),
        }
        
        self.mappings_cache[table_name] = mapping
        return mapping
    
    def _map_columns(self, schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Map columns to their semantic roles."""
        columns = list(schema_info.get('columns', {}).keys())
        column_lower = {c: c.lower() for c in columns}
        
        mapping = {
            'primary_key': schema_info.get('primary_keys', []),
            'foreign_keys': schema_info.get('foreign_keys', []),
            'timestamps': schema_info.get('timestamp_columns', []),
            'numeric': schema_info.get('numeric_columns', []),
            'categorical': schema_info.get('categorical_columns', []),
            'boolean': schema_info.get('boolean_columns', []),
            'text': schema_info.get('text_columns', []),
            'value_features': [],
            'entity_ids': [],
            'item_ids': [],
            'user_ids': [],
        }
        
        # Identify value features (numeric columns that should be aggregated)
        for col in columns:
            col_l = column_lower[col]
            if any(kw in col_l for kw in self.column_patterns['value_columns']):
                mapping['value_features'].append(col)
        
        # Identify entity IDs from foreign keys
        for fk in schema_info.get('foreign_keys', []):
            fk_l = fk.lower()
            if any(kw in fk_l for kw in self.column_patterns['user_columns']):
                mapping['user_ids'].append(fk)
            elif any(kw in fk_l for kw in self.column_patterns['item_columns']):
                mapping['item_ids'].append(fk)
            else:
                mapping['entity_ids'].append(fk)
        
        return mapping
    
    def _recommend_features(self, schema_info: Dict[str, Any], 
                           domain: str) -> List[Dict[str, Any]]:
        """Generate feature recommendations based on schema and domain."""
        recommendations = []
        
        numeric_cols = schema_info.get('numeric_columns', [])
        categorical_cols = schema_info.get('categorical_columns', [])
        timestamps = schema_info.get('timestamp_columns', [])
        
        # Aggregation recommendations
        if numeric_cols:
            recommendations.append({
                'feature_type': 'aggregation',
                'applicable_columns': numeric_cols,
                'suggested_operations': ['sum', 'mean', 'std', 'min', 'max', 'count'],
                'priority': 'high'
            })
        
        # Temporal recommendations
        if timestamps:
            recommendations.append({
                'feature_type': 'temporal',
                'applicable_columns': timestamps,
                'suggested_operations': ['time_since', 'day_of_week', 'hour', 'month', 'is_weekend'],
                'priority': 'high'
            })
        
        # Categorical recommendations
        if categorical_cols:
            recommendations.append({
                'feature_type': 'categorical_encoding',
                'applicable_columns': categorical_cols[:5],  # Top 5
                'suggested_operations': ['one_hot', 'frequency_encoding', 'target_encoding'],
                'priority': 'medium'
            })
            
            recommendations.append({
                'feature_type': 'category_affinity',
                'applicable_columns': categorical_cols,
                'suggested_operations': ['entropy', 'mode', 'n_unique'],
                'priority': 'medium'
            })
        
        # Domain-specific recommendations
        domain_features = self._get_domain_specific_features(domain, schema_info)
        recommendations.extend(domain_features)
        
        return recommendations
    
    def _get_domain_specific_features(self, domain: str, 
                                      schema_info: Dict[str, Any]) -> List[Dict]:
        """Get domain-specific feature recommendations."""
        recommendations = []
        
        if domain == 'ecommerce':
            recommendations.append({
                'feature_type': 'rfm_analysis',
                'description': 'Recency, Frequency, Monetary features',
                'required_columns': ['timestamp', 'value', 'user_id'],
                'priority': 'high'
            })
            recommendations.append({
                'feature_type': 'purchase_patterns',
                'description': 'Purchase frequency and basket analysis',
                'priority': 'medium'
            })
        
        elif domain == 'fintech':
            recommendations.append({
                'feature_type': 'transaction_velocity',
                'description': 'Transaction speed and frequency metrics',
                'priority': 'high'
            })
            recommendations.append({
                'feature_type': 'risk_indicators',
                'description': 'Unusual pattern detection features',
                'priority': 'high'
            })
        
        elif domain == 'healthcare':
            recommendations.append({
                'feature_type': 'patient_history',
                'description': 'Historical treatment and outcome features',
                'priority': 'high'
            })
        
        elif domain == 'edtech':
            recommendations.append({
                'feature_type': 'learning_progress',
                'description': 'Student progress and engagement metrics',
                'priority': 'high'
            })
        
        elif domain == 'saas':
            recommendations.append({
                'feature_type': 'usage_metrics',
                'description': 'Feature usage and engagement tracking',
                'priority': 'high'
            })
            recommendations.append({
                'feature_type': 'churn_indicators',
                'description': 'Usage decline and risk signals',
                'priority': 'high'
            })
        
        elif domain == 'analytics':
            recommendations.append({
                'feature_type': 'funnel_analysis',
                'description': 'Conversion funnel features',
                'priority': 'medium'
            })
            recommendations.append({
                'feature_type': 'session_metrics',
                'description': 'Session-based behavioral features',
                'priority': 'high'
            })
        
        return recommendations
    
    def get_feature_config_for_table(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get cached feature configuration for a table."""
        return self.mappings_cache.get(table_name)
    
    def adapt_to_new_schema(self, 
                           df: pd.DataFrame, 
                           table_name: str,
                           existing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt feature configuration to a new/changed schema.
        
        Args:
            df: New DataFrame with potentially different schema
            table_name: Table name
            existing_config: Previous configuration if available
            
        Returns:
            Updated configuration
        """
        logger.info(f"Adapting to schema changes for '{table_name}'")
        
        new_mapping = self.analyze_and_map(df, table_name)
        
        if existing_config:
            # Compare and identify changes
            old_cols = set(existing_config.get('column_mapping', {}).get('numeric', []))
            new_cols = set(new_mapping['column_mapping']['numeric'])
            
            added_cols = new_cols - old_cols
            removed_cols = old_cols - new_cols
            
            if added_cols:
                logger.info(f"Added numeric columns: {added_cols}")
            if removed_cols:
                logger.warning(f"Removed numeric columns: {removed_cols}")
            
            new_mapping['schema_changes'] = {
                'added_columns': list(added_cols),
                'removed_columns': list(removed_cols),
                'adaptation_timestamp': datetime.now().isoformat()
            }
        
        return new_mapping
    
    def generate_feature_pipeline(self, 
                                  mappings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a complete feature engineering pipeline from mappings.
        
        Args:
            mappings: Dictionary of table_name -> mapping
            
        Returns:
            Pipeline configuration
        """
        pipeline = {
            'entities': {},
            'relationships': [],
            'feature_groups': [],
            'primitives': {}
        }
        
        for table_name, mapping in mappings.items():
            # Entity configuration
            col_map = mapping.get('column_mapping', {})
            pipeline['entities'][table_name] = {
                'index': col_map.get('primary_key', [None])[0],
                'time_index': col_map.get('timestamps', [None])[0],
                'feature_columns': {
                    'numeric': col_map.get('numeric', []),
                    'categorical': col_map.get('categorical', []),
                    'value': col_map.get('value_features', []),
                }
            }
            
            # Primitives for this entity
            domain = mapping.get('domain', 'generic')
            pipeline['primitives'][table_name] = self.primitive_registry.get_primitives_for_domain(domain)
        
        return pipeline
