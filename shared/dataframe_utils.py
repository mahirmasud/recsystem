"""
DataFrame utility functions for the recommendation system.
Common operations for data manipulation and transformation.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from shared.logger import get_logger


logger = get_logger(__name__)


class DataFrameUtils:
    """
    Utility class for common DataFrame operations.
    
    Provides reusable functions for:
    - Type conversion
    - Null handling
    - Column operations
    - Memory optimization
    """
    
    @staticmethod
    def convert_dtypes(
        df: pd.DataFrame,
        dtype_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert columns to specified data types.
        
        Args:
            df: Input DataFrame
            dtype_map: Dictionary mapping column names to dtypes
        
        Returns:
            DataFrame with converted types
        """
        df = df.copy()
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype == 'string':
                        df[col] = df[col].astype('string')
                    else:
                        df[col] = df[col].astype(dtype)
                    logger.debug(f"Converted {col} to {dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")
        return df
    
    @staticmethod
    def rename_columns(
        df: pd.DataFrame,
        mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Rename columns based on mapping dictionary.
        
        Args:
            df: Input DataFrame
            mapping: Dictionary mapping old names to new names
        
        Returns:
            DataFrame with renamed columns
        """
        df = df.copy()
        df.rename(columns=mapping, inplace=True)
        logger.debug(f"Renamed columns: {mapping}")
        return df
    
    @staticmethod
    def select_columns(
        df: pd.DataFrame,
        columns: List[str],
        strict: bool = False
    ) -> pd.DataFrame:
        """
        Select specific columns from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of column names to select
            strict: If True, raise error for missing columns
        
        Returns:
            DataFrame with selected columns
        """
        available_cols = [col for col in columns if col in df.columns]
        missing_cols = set(columns) - set(available_cols)
        
        if missing_cols and strict:
            raise KeyError(f"Missing columns: {missing_cols}")
        
        if missing_cols:
            logger.warning(f"Columns not found and excluded: {missing_cols}")
        
        return df[available_cols].copy()
    
    @staticmethod
    def handle_nulls(
        df: pd.DataFrame,
        numeric_fill: float = 0,
        categorical_fill: str = 'UNKNOWN',
        datetime_fill: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Handle null values based on column type.
        
        Args:
            df: Input DataFrame
            numeric_fill: Fill value for numeric columns
            categorical_fill: Fill value for string/object columns
            datetime_fill: Fill value for datetime columns
        
        Returns:
            DataFrame with nulls handled
        """
        df = df.copy()
        
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(numeric_fill)
                elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                    df[col] = df[col].fillna(categorical_fill)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    if datetime_fill is not None:
                        df[col] = df[col].fillna(datetime_fill)
                
                logger.debug(f"Handled nulls in {col}")
        
        return df
    
    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with optimized memory usage
        """
        df = df.copy()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize integer types
            if pd.api.types.is_integer_dtype(col_type):
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val >= 0 and max_val <= 255:
                    df[col] = df[col].astype('uint8')
                elif min_val >= -128 and max_val <= 127:
                    df[col] = df[col].astype('int8')
                elif min_val >= 0 and max_val <= 65535:
                    df[col] = df[col].astype('uint16')
                elif min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype('int16')
            
            # Optimize float types
            elif pd.api.types.is_float_dtype(col_type):
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val >= -3.4e38 and max_val <= 3.4e38:
                    df[col] = df[col].astype('float32')
        
        logger.info(f"Optimized memory usage for DataFrame")
        return df
    
    @staticmethod
    def ensure_index(
        df: pd.DataFrame,
        index_col: str,
        unique: bool = True
    ) -> pd.DataFrame:
        """
        Ensure DataFrame has proper index.
        
        Args:
            df: Input DataFrame
            index_col: Column to use as index
            unique: Whether to check for uniqueness
        
        Returns:
            DataFrame with proper index
        """
        df = df.copy()
        
        if index_col not in df.columns:
            raise KeyError(f"Index column {index_col} not found")
        
        if unique and df[index_col].duplicated().any():
            logger.warning(f"Index column {index_col} has duplicates")
        
        df.set_index(index_col, inplace=True)
        return df
    
    @staticmethod
    def add_computed_column(
        df: pd.DataFrame,
        new_col: str,
        formula: str,
        fill_na: bool = True
    ) -> pd.DataFrame:
        """
        Add computed column using eval formula.
        
        Args:
            df: Input DataFrame
            new_col: Name for new column
            formula: Formula string for computation
            fill_na: Whether to fill NaN results with 0
        
        Returns:
            DataFrame with new column
        """
        df = df.copy()
        try:
            df[new_col] = df.eval(formula)
            if fill_na:
                df[new_col] = df[new_col].fillna(0)
            logger.debug(f"Added computed column {new_col}")
        except Exception as e:
            logger.error(f"Failed to compute column {new_col}: {e}")
            df[new_col] = np.nan
        
        return df
    
    @staticmethod
    def drop_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', False)
        
        Returns:
            DataFrame without duplicates
        """
        original_len = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed = original_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    @staticmethod
    def filter_by_condition(
        df: pd.DataFrame,
        condition: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame using query condition.
        
        Args:
            df: Input DataFrame
            condition: Query condition string
        
        Returns:
            Filtered DataFrame
        """
        try:
            return df.query(condition).copy()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return df.copy()
    
    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
