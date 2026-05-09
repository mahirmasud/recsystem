"""
DataFrame standardizer for the recommendation system.
Orchestrates the full standardization pipeline for each entity type.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from shared.constants import Constants
from Standardized_Data_Layer.canonical_schema import CanonicalSchema
from Standardized_Data_Layer.config_reader import ConfigReader
from Standardized_Data_Layer.mapping_applier import MappingApplier
from Standardized_Data_Layer.null_handler import NullHandler


logger = get_logger(__name__)


class DataFrameStandardizer:
    """
    Main orchestrator for DataFrame standardization.
    
    Coordinates:
    - Column mapping application
    - Type conversion
    - Null handling
    - Schema validation
    - Output formatting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize standardizer.
        
        Args:
            config_path: Path to rec_config.json
        """
        self.config_reader = ConfigReader(config_path)
        self.mapping_applier = MappingApplier(self.config_reader)
        self.null_handler = NullHandler()
    
    def standardize(
        self,
        df: pd.DataFrame,
        entity_type: str,
        add_missing_columns: bool = True,
        convert_types: bool = True,
        handle_nulls: bool = True
    ) -> pd.DataFrame:
        """
        Full standardization pipeline for a DataFrame.
        
        Args:
            df: Source DataFrame
            entity_type: Entity type name
            add_missing_columns: Whether to add missing optional columns
            convert_types: Whether to convert to canonical dtypes
            handle_nulls: Whether to handle null values
        
        Returns:
            Standardized DataFrame
        """
        logger.info(f"Starting standardization for {entity_type} ({len(df)} rows)")
        
        # Step 1: Apply column mappings
        df_result = self.mapping_applier.select_and_map(df, entity_type)
        
        # Step 2: Add missing optional columns
        if add_missing_columns:
            df_result = self.mapping_applier.add_missing_optional_columns(
                df_result, entity_type
            )
        
        # Step 3: Convert data types
        if convert_types:
            df_result = self.mapping_applier.convert_dtypes(df_result, entity_type)
        
        # Step 4: Handle null values
        if handle_nulls:
            df_result = self.null_handler.handle_all(df_result, entity_type)
        
        # Step 5: Ensure canonical column order
        expected_columns = CanonicalSchema.get_columns(entity_type)
        available_columns = [col for col in expected_columns if col in df_result.columns]
        df_result = df_result[available_columns]
        
        logger.info(f"Completed standardization for {entity_type}")
        return df_result
    
    def standardize_multiple(
        self,
        dataframes: Dict[str, pd.DataFrame],
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Standardize multiple DataFrames for different entity types.
        
        Args:
            dataframes: Dictionary mapping entity type to DataFrame
            entity_types: List of entity types to process (default: all keys)
        
        Returns:
            Dictionary of standardized DataFrames
        """
        entity_types = entity_types or list(dataframes.keys())
        results = {}
        
        for entity_type in entity_types:
            if entity_type not in dataframes:
                logger.warning(f"No data provided for {entity_type}, skipping")
                continue
            
            try:
                results[entity_type] = self.standardize(
                    dataframes[entity_type],
                    entity_type
                )
            except Exception as e:
                logger.error(f"Failed to standardize {entity_type}: {e}")
                raise
        
        return results
    
    def get_standardization_report(
        self,
        original_df: pd.DataFrame,
        standardized_df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Generate report on standardization transformations.
        
        Args:
            original_df: Original source DataFrame
            standardized_df: Result of standardization
            entity_type: Entity type name
        
        Returns:
            Report dictionary
        """
        lineage = self.mapping_applier.get_lineage_info(entity_type)
        
        return {
            'entity_type': entity_type,
            'original_rows': len(original_df),
            'standardized_rows': len(standardized_df),
            'original_columns': list(original_df.columns),
            'standardized_columns': list(standardized_df.columns),
            'column_mappings': {
                col: info['source_column'] 
                for col, info in lineage.items()
            },
            'null_counts_before': original_df.isnull().sum().to_dict(),
            'null_counts_after': standardized_df.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in standardized_df.dtypes.items()}
        }
