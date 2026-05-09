"""
Mapping applier for standardization.
Applies trusted column mappings to transform source data to canonical format.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


class MappingApplier:
    """
    Applies column mappings to transform source DataFrames to canonical format.
    
    Features:
    - Column renaming based on trusted mappings
    - Type conversion
    - Handling of missing optional columns
    - Source column tracking for lineage
    """
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize mapping applier.
        
        Args:
            config_reader: ConfigReader instance (created if not provided)
        """
        self.config_reader = config_reader or ConfigReader()
    
    def apply_mapping(
        self,
        df: pd.DataFrame,
        entity_type: str,
        source_column_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Apply column mapping to transform DataFrame to canonical format.
        
        Args:
            df: Source DataFrame
            entity_type: Entity type name (users, items, etc.)
            source_column_map: Optional override mapping
        
        Returns:
            DataFrame with canonical column names
        """
        logger.info(f"Applying mapping for entity type: {entity_type}")
        
        # Get mapping from config
        if source_column_map is None:
            source_column_map = self.config_reader.get_column_mapping(entity_type)
        
        # Build rename dictionary (only for columns that exist in source)
        available_columns = set(df.columns)
        rename_dict = {
            src: canonical 
            for src, canonical in source_column_map.items()
            if src in available_columns
        }
        
        # Track which columns were mapped
        mapped_columns = list(rename_dict.values())
        unmapped_canonical = set(self.config_reader.get_entity_mapping(entity_type).keys()) - set(mapped_columns)
        
        if unmapped_canonical:
            logger.debug(f"Optional columns not found in source: {unmapped_canonical}")
        
        # Rename columns
        df_result = df.rename(columns=rename_dict)
        
        logger.info(f"Mapped {len(rename_dict)} columns for {entity_type}")
        return df_result
    
    def select_and_map(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> pd.DataFrame:
        """
        Select only mapped columns and apply mapping.
        
        Args:
            df: Source DataFrame
            entity_type: Entity type name
        
        Returns:
            DataFrame with only canonical columns
        """
        logger.info(f"Selecting and mapping columns for {entity_type}")
        
        # Get source-to-canonical mapping
        source_column_map = self.config_reader.get_column_mapping(entity_type)
        
        # Find which source columns exist
        available_source = [col for col in source_column_map.keys() if col in df.columns]
        
        if not available_source:
            raise ValueError(f"No matching source columns found for entity {entity_type}")
        
        # Select and map
        df_selected = df[available_source].copy()
        df_result = self.apply_mapping(df_selected, entity_type, source_column_map)
        
        logger.info(f"Selected {len(available_source)} columns for {entity_type}")
        return df_result
    
    def add_missing_optional_columns(
        self,
        df: pd.DataFrame,
        entity_type: str,
        fill_values: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Add missing optional columns with default values.
        
        Args:
            df: DataFrame with some canonical columns
            entity_type: Entity type name
            fill_values: Custom fill values for missing columns
        
        Returns:
            DataFrame with all canonical columns present
        """
        logger.info(f"Adding missing optional columns for {entity_type}")
        
        # Get all expected columns
        entity_mapping = self.config_reader.get_entity_mapping(entity_type)
        all_canonical = list(entity_mapping.keys())
        
        # Default fill values
        default_fills = {
            'int64': 0,
            'float64': 0.0,
            'string': '',
            'datetime64[ns]': pd.NaT,
            'object': None
        }
        
        fill_values = fill_values or {}
        
        # Add missing columns
        for col in all_canonical:
            if col not in df.columns:
                dtype = entity_mapping[col].get('dtype', 'object')
                fill_value = fill_values.get(col, default_fills.get(dtype, None))
                df[col] = fill_value
                logger.debug(f"Added missing column {col} with default value")
        
        return df
    
    def get_mapped_dtypes(self, entity_type: str) -> Dict[str, str]:
        """
        Get expected dtypes for canonical columns.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Dictionary of column to dtype
        """
        return self.config_reader.get_dtype_mapping(entity_type)
    
    def convert_dtypes(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> pd.DataFrame:
        """
        Convert columns to expected canonical dtypes.
        
        Args:
            df: DataFrame with canonical column names
            entity_type: Entity type name
        
        Returns:
            DataFrame with correct dtypes
        """
        logger.info(f"Converting dtypes for {entity_type}")
        
        dtype_map = self.get_mapped_dtypes(entity_type)
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype == 'string':
                        df[col] = df[col].astype('string')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")
        
        return df
    
    def get_lineage_info(
        self,
        entity_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Get lineage information for column mappings.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Dictionary with source and transformation info
        """
        mappings = self.config_reader.get_entity_mapping(entity_type)
        lineage = {}
        
        for canonical, details in mappings.items():
            lineage[canonical] = {
                'source_column': details['source_column'],
                'source_table': details.get('source_table', ''),
                'source_dtype': details.get('dtype', 'unknown'),
                'transformation': 'direct_mapping'
            }
        
        return lineage
