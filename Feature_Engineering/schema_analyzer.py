"""
Schema Analyzer - Analyzes dataset schemas to detect entities, relationships, and column types.

Automatically detects:
- Primary keys
- Foreign keys  
- Timestamp columns
- Numeric vs categorical columns
- Entity boundaries
- Relationship patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """Analyzes dataset schemas for automated feature engineering."""
    
    def __init__(self):
        self.schema_cache = {}
        self.detected_entities = {}
        self.detected_relationships = []
        
        # Common naming patterns for detection
        self.pk_patterns = ['id', '_id', 'ID', 'Id']
        self.fk_patterns = ['_id', 'Id', 'ID', '_fk', 'ref_']
        self.timestamp_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 
                                   '_at', '_on', 'datetime']
        self.numeric_keywords = ['amount', 'value', 'count', 'quantity', 'price', 
                                'cost', 'revenue', 'score', 'rate', 'total']
        self.categorical_keywords = ['type', 'category', 'status', 'name', 'code', 
                                    'class', 'group', 'segment']
        
        logger.info("SchemaAnalyzer initialized")
    
    def analyze_dataframe(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Analyze a dataframe to detect schema characteristics.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the table/dataframe
            
        Returns:
            Dictionary with schema analysis results
        """
        logger.info(f"Analyzing schema for table: {table_name}")
        
        schema_info = {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'timestamp_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'boolean_columns': [],
            'text_columns': [],
            'potential_entities': [],
            'uniqueness_stats': {},
            'null_stats': {},
        }
        
        for col in df.columns:
            col_info = self._analyze_column(df, col)
            schema_info['columns'][col] = col_info
            
            # Detect primary key
            if self._is_primary_key_candidate(df, col):
                schema_info['primary_keys'].append(col)
            
            # Detect foreign key
            if self._is_foreign_key_candidate(col):
                schema_info['foreign_keys'].append(col)
            
            # Detect timestamp
            if self._is_timestamp_column(df, col):
                schema_info['timestamp_columns'].append(col)
            
            # Categorize column type
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                schema_info['numeric_columns'].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                schema_info['boolean_columns'].append(col)
            elif pd.api.types.is_categorical_dtype(dtype) or df[col].dtype == 'object':
                if df[col].nunique() < len(df) * 0.5:  # Low cardinality
                    schema_info['categorical_columns'].append(col)
                else:
                    schema_info['text_columns'].append(col)
            
            # Calculate uniqueness
            schema_info['uniqueness_stats'][col] = df[col].nunique() / len(df) if len(df) > 0 else 0
            schema_info['null_stats'][col] = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        
        # Detect potential entities (tables with unique IDs)
        for pk in schema_info['primary_keys']:
            if schema_info['uniqueness_stats'].get(pk, 0) >= 0.9:
                schema_info['potential_entities'].append({
                    'entity_id_column': pk,
                    'confidence': schema_info['uniqueness_stats'][pk]
                })
        
        self.schema_cache[table_name] = schema_info
        return schema_info
    
    def _analyze_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze individual column properties."""
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': list(df[col].dropna().head(3).values),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['min'] = float(df[col].min()) if not df[col].empty else None
            col_info['max'] = float(df[col].max()) if not df[col].empty else None
            col_info['mean'] = float(df[col].mean()) if not df[col].empty else None
            col_info['std'] = float(df[col].std()) if not df[col].empty else None
        
        return col_info
    
    def _is_primary_key_candidate(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column could be a primary key."""
        col_lower = col.lower()
        
        # Check naming pattern
        is_pk_named = any(pattern in col_lower for pattern in self.pk_patterns)
        
        # Check uniqueness
        is_unique = df[col].nunique() == len(df)
        
        # Check null-free
        no_nulls = df[col].isnull().sum() == 0
        
        return (is_pk_named or is_unique) and no_nulls
    
    def _is_foreign_key_candidate(self, col: str) -> bool:
        """Check if column could be a foreign key based on naming."""
        col_lower = col.lower()
        
        # Ends with _id but isn't the primary key of this table
        if col_lower.endswith('_id') and col_lower not in ['id']:
            return True
        
        # Has ref_ prefix
        if col_lower.startswith('ref_'):
            return True
        
        return False
    
    def _is_timestamp_column(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column contains timestamp data."""
        col_lower = col.lower()
        
        # Check naming pattern
        if any(pattern in col_lower for pattern in self.timestamp_patterns):
            return True
        
        # Check actual dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        
        return False
    
    def detect_domain(self, schema_info: Dict[str, Any]) -> str:
        """
        Detect business domain from schema information.
        
        Returns:
            Domain string (ecommerce, fintech, healthcare, etc.)
        """
        all_columns = set(schema_info.get('columns', {}).keys())
        column_names_lower = {c.lower() for c in all_columns}
        
        domain_indicators = {
            'ecommerce': {'transaction', 'product', 'cart', 'checkout', 'revenue', 
                         'sales', 'order', 'item', 'customer', 'purchase'},
            'fintech': {'payment', 'transfer', 'account', 'balance', 'interest', 
                       'loan', 'credit', 'debit', 'bank', 'financial'},
            'healthcare': {'patient', 'diagnosis', 'treatment', 'medication', 
                          'symptom', 'doctor', 'hospital', 'clinical'},
            'edtech': {'student', 'course', 'lesson', 'quiz', 'grade', 
                      'enrollment', 'teacher', 'school', 'learning'},
            'saas': {'subscription', 'user_session', 'feature_usage', 'plan',
                    'tenant', 'license', 'api_call'},
            'crm': {'lead', 'opportunity', 'contact', 'deal', 'pipeline',
                   'sales_rep', 'account_manager'},
            'erp': {'inventory', 'shipment', 'warehouse', 'procurement',
                   'supplier', 'manufacturing', 'bom'},
            'analytics': {'event', 'pageview', 'click', 'session', 'conversion',
                         'funnel', 'attribution'},
            'behavioral': {'action', 'behavior', 'activity', 'engagement',
                          'interaction', 'response'},
        }
        
        scores = {}
        for domain, keywords in domain_indicators.items():
            score = len(keywords & column_names_lower)
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return 'generic'
        
        detected_domain = max(scores, key=scores.get)
        logger.info(f"Detected domain: {detected_domain} (score: {scores[detected_domain]})")
        return detected_domain
    
    def find_related_tables(self, table1_schema: Dict, table2_schema: Dict) -> Optional[Dict]:
        """
        Find potential relationships between two tables.
        
        Returns:
            Relationship info dict or None
        """
        fk_candidates = table1_schema.get('foreign_keys', [])
        pk_candidates = table2_schema.get('primary_keys', [])
        
        for fk in fk_candidates:
            # Extract referenced table name from FK column name
            fk_base = fk.replace('_id', '').replace('Id', '').replace('ID', '')
            
            for pk in pk_candidates:
                pk_base = pk.replace('_id', '').replace('Id', '').replace('ID', '')
                
                if fk_base.lower() == pk_base.lower() or fk_base.lower() == table2_schema.get('table_name', '').lower():
                    return {
                        'parent_table': table2_schema['table_name'],
                        'child_table': table1_schema['table_name'],
                        'parent_key': pk,
                        'child_key': fk,
                        'relationship_type': 'one-to-many',
                        'confidence': 'high'
                    }
        
        return None
    
    def get_entity_recommendations(self, schema_info: Dict[str, Any]) -> List[Dict]:
        """Get recommendations for entity configuration."""
        recommendations = []
        
        # Recommend primary entity
        pks = schema_info.get('primary_keys', [])
        if pks:
            recommendations.append({
                'entity_type': 'primary',
                'recommended_id': pks[0],
                'reason': 'Primary key detected'
            })
        
        # Recommend time index
        timestamps = schema_info.get('timestamp_columns', [])
        if timestamps:
            recommendations.append({
                'entity_type': 'time_index',
                'recommended_column': timestamps[0],
                'reason': 'Timestamp column detected'
            })
        
        # Recommend categorical columns for encoding
        categoricals = schema_info.get('categorical_columns', [])
        if categoricals:
            recommendations.append({
                'entity_type': 'categorical_features',
                'recommended_columns': categoricals[:5],  # Top 5
                'reason': 'Low-cardinality categorical columns'
            })
        
        return recommendations


class RelationshipDetector:
    """Detects relationships between tables for EntitySet construction."""
    
    def __init__(self):
        self.relationships = []
        logger.info("RelationshipDetector initialized")
    
    def detect_relationships(self, schemas: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        Detect relationships across multiple table schemas.
        
        Args:
            schemas: Dictionary of table_name -> schema_info
            
        Returns:
            List of relationship dictionaries
        """
        logger.info(f"Detecting relationships across {len(schemas)} tables")
        
        self.relationships = []
        table_names = list(schemas.keys())
        
        schema_analyzer = SchemaAnalyzer()
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                # Check table1 -> table2
                rel = schema_analyzer.find_related_tables(schemas[table1], schemas[table2])
                if rel:
                    self.relationships.append(rel)
                
                # Check table2 -> table1
                rel_reverse = schema_analyzer.find_related_tables(schemas[table2], schemas[table1])
                if rel_reverse:
                    self.relationships.append(rel_reverse)
        
        logger.info(f"Detected {len(self.relationships)} relationships")
        return self.relationships
    
    def validate_relationship(self, parent_df: pd.DataFrame, child_df: pd.DataFrame,
                             parent_key: str, child_key: str) -> Dict[str, Any]:
        """Validate a detected relationship."""
        validation = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check for orphan records
        parent_ids = set(parent_df[parent_key].unique())
        child_ids = set(child_df[child_key].dropna().unique())
        
        orphans = child_ids - parent_ids
        if orphans:
            validation['valid'] = False
            validation['issues'].append(f'{len(orphans)} orphan records in child table')
        
        # Check referential integrity percentage
        if len(child_ids) > 0:
            integrity_pct = (len(child_ids - orphans) / len(child_ids)) * 100
            validation['stats']['referential_integrity_pct'] = integrity_pct
        
        # Check for duplicate parent keys
        dup_count = parent_df[parent_key].duplicated().sum()
        if dup_count > 0:
            validation['issues'].append(f'{dup_count} duplicate keys in parent table')
        
        return validation
