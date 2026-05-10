"""
Entity Builder - Constructs Featuretools EntitySets from standardized data.

Dynamically builds EntitySets with:
- Automatic entity detection
- Relationship inference
- Time index assignment
- Multi-domain support
"""

import featuretools as ft
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from .schema_analyzer import SchemaAnalyzer, RelationshipDetector

logger = logging.getLogger(__name__)


class EntityBuilder:
    """Builds Featuretools EntitySets from standardized datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schema_analyzer = SchemaAnalyzer()
        self.relationship_detector = RelationshipDetector()
        self.entitysets = {}
        self.schemas = {}
        
        logger.info("EntityBuilder initialized with Featuretools support")
    
    def build_entityset(self, 
                       dataframes: Dict[str, pd.DataFrame],
                       name: str = "entityset",
                       entity_config: Optional[Dict[str, Any]] = None) -> ft.EntitySet:
        """
        Build a Featuretools EntitySet from multiple dataframes.
        
        Args:
            dataframes: Dictionary of table_name -> DataFrame
            name: Name for the EntitySet
            entity_config: Optional configuration for entities
            
        Returns:
            Featuretools EntitySet
        """
        logger.info(f"Building EntitySet '{name}' from {len(dataframes)} dataframes")
        
        # Analyze schemas first
        for table_name, df in dataframes.items():
            self.schemas[table_name] = self.schema_analyzer.analyze_dataframe(df, table_name)
        
        # Detect relationships
        relationships = self.relationship_detector.detect_relationships(self.schemas)
        
        # Create EntitySet
        es = ft.EntitySet(id=name)
        
        # Add dataframes as entities
        for table_name, df in dataframes.items():
            schema_info = self.schemas[table_name]
            
            # Determine entity configuration
            if entity_config and table_name in entity_config:
                cfg = entity_config[table_name]
                index = cfg.get('index')
                time_index = cfg.get('time_index')
            else:
                # Auto-detect
                index = self._auto_detect_index(schema_info)
                time_index = self._auto_detect_time_index(schema_info)
            
            # Make index column string type for Featuretools compatibility
            df_copy = df.copy()
            if index and index in df_copy.columns:
                if not pd.api.types.is_string_dtype(df_copy[index]):
                    df_copy[index] = df_copy[index].astype(str)
            
            # Add dataframe to EntitySet
            es.add_dataframe(
                dataframe_name=table_name,
                dataframe=df_copy,
                index=index,
                time_index=time_index,
                make_index=(index is None)
            )
            
            logger.info(f"Added entity '{table_name}' with index={index}, time_index={time_index}")
        
        # Add relationships
        for rel in relationships:
            try:
                relationship = ft.Relationship(
                    parent_es=es,
                    parent_dataframe_name=rel['parent_table'],
                    parent_column_name=rel['parent_key'],
                    child_es=es,
                    child_dataframe_name=rel['child_table'],
                    child_column_name=rel['child_key']
                )
                es.add_relationship(relationship)
                logger.info(f"Added relationship: {rel['parent_table']}.{rel['parent_key']} -> {rel['child_table']}.{rel['child_key']}")
            except Exception as e:
                logger.warning(f"Could not add relationship {rel}: {e}")
        
        self.entitysets[name] = es
        return es
    
    def _auto_detect_index(self, schema_info: Dict[str, Any]) -> Optional[str]:
        """Auto-detect primary key index column."""
        pks = schema_info.get('primary_keys', [])
        if pks:
            return pks[0]
        
        # Fallback: find column ending with '_id' that matches table name
        table_name = schema_info.get('table_name', '')
        for col in schema_info.get('columns', {}).keys():
            col_lower = col.lower()
            if col_lower == f"{table_name.lower()}_id" or col_lower == 'id':
                return col
        
        return None
    
    def _auto_detect_time_index(self, schema_info: Dict[str, Any]) -> Optional[str]:
        """Auto-detect timestamp column for time index."""
        timestamps = schema_info.get('timestamp_columns', [])
        if timestamps:
            return timestamps[0]
        return None
    
    def build_from_single_table(self, 
                                df: pd.DataFrame,
                                entity_name: str = "data",
                                index_col: Optional[str] = None,
                                time_col: Optional[str] = None) -> ft.EntitySet:
        """
        Build EntitySet from a single table (no relationships).
        
        Args:
            df: Input DataFrame
            entity_name: Name for the entity
            index_col: Optional index column name
            time_col: Optional time column name
            
        Returns:
            Featuretools EntitySet
        """
        logger.info(f"Building single-table EntitySet '{entity_name}'")
        
        # Analyze schema if index/time not provided
        if not index_col or not time_col:
            schema_info = self.schema_analyzer.analyze_dataframe(df, entity_name)
            if not index_col:
                index_col = self._auto_detect_index(schema_info)
            if not time_col:
                time_col = self._auto_detect_time_index(schema_info)
        
        es = ft.EntitySet(id=entity_name)
        
        df_copy = df.copy()
        if index_col and index_col in df_copy.columns:
            if not pd.api.types.is_string_dtype(df_copy[index_col]):
                df_copy[index_col] = df_copy[index_col].astype(str)
        
        es.add_dataframe(
            dataframe_name=entity_name,
            dataframe=df_copy,
            index=index_col,
            time_index=time_col,
            make_index=(index_col is None)
        )
        
        self.entitysets[entity_name] = es
        return es
    
    def get_entityset(self, name: str) -> Optional[ft.EntitySet]:
        """Get an EntitySet by name."""
        return self.entitysets.get(name)
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all analyzed schemas."""
        return self.schemas
    
    def get_detected_domain(self) -> str:
        """Get detected business domain from schemas."""
        if not self.schemas:
            return 'generic'
        
        # Combine all columns from all schemas
        all_columns = set()
        for schema in self.schemas.values():
            all_columns.update(schema.get('columns', {}).keys())
        
        combined_schema = {'columns': {col: {} for col in all_columns}}
        return self.schema_analyzer.detect_domain(combined_schema)
    
    def export_entityset_info(self, es: ft.EntitySet) -> Dict[str, Any]:
        """Export EntitySet metadata for documentation."""
        info = {
            'id': es.id,
            'entities': [],
            'relationships': [],
            'entity_count': len(es.dataframes),
            'relationship_count': len(es.relationships)
        }
        
        for df in es.dataframes:
            entity_info = {
                'name': df.ww.name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'index': df.ww.index,
                'time_index': df.ww.time_index
            }
            info['entities'].append(entity_info)
        
        for rel in es.relationships:
            rel_info = {
                'parent': rel.parent_dataframe.ww.name,
                'parent_column': rel.parent_column.name,
                'child': rel.child_dataframe.ww.name,
                'child_column': rel.child_column.name
            }
            info['relationships'].append(rel_info)
        
        return info
