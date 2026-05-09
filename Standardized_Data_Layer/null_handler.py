"""
Null value handler for the standardization layer.
Provides strategies for handling missing values based on column type and business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from shared.logger import get_logger
from Standardized_Data_Layer.canonical_schema import CanonicalSchema


logger = get_logger(__name__)


class NullHandler:
    """
    Handles null values in standardized DataFrames.
    
    Strategies:
    - Numeric columns: fill with 0 or median
    - String columns: fill with empty string or 'UNKNOWN'
    - Datetime columns: fill with NaT or specific date
    - Required columns: raise error if nulls present
    """
    
    # Default fill values by dtype
    FILL_VALUES = {
        'int64': 0,
        'float64': 0.0,
        'string': '',
        'object': None,
        'datetime64[ns]': pd.NaT
    }
    
    # Alternative fill values for specific columns
    COLUMN_SPECIFIC_FILLS = {
        'discount': 0.0,
        'quantity': 1,
        'unit_price': 0.0,
        'total_amount': 0.0,
        'price': 0.0,
        'cost': 0.0,
        'country': 'UNKNOWN',
        'gender': 'UNKNOWN',
        'brand': 'UNKNOWN',
        'item_name': 'Unknown Item',
        'category_name': 'Uncategorized',
        'interaction_type': 'unknown',
        'device_type': 'unknown',
    }
    
    def __init__(self, strategy: str = 'default'):
        """
        Initialize null handler.
        
        Args:
            strategy: Fill strategy ('default', 'median', 'mode', 'drop')
        """
        self.strategy = strategy
    
    def handle_all(
        self,
        df: pd.DataFrame,
        entity_type: str,
        check_required: bool = True
    ) -> pd.DataFrame:
        """
        Handle all null values in DataFrame.
        
        Args:
            df: Input DataFrame
            entity_type: Entity type name
            check_required: Whether to validate required columns
        
        Returns:
            DataFrame with nulls handled
        """
        logger.info(f"Handling nulls for {entity_type} using strategy: {self.strategy}")
        
        df_result = df.copy()
        
        # Check required columns first
        if check_required:
            self._check_required_columns(df_result, entity_type)
        
        # Apply fill strategy
        if self.strategy == 'drop':
            df_result = df_result.dropna()
        elif self.strategy == 'median':
            df_result = self._fill_with_median(df_result)
        elif self.strategy == 'mode':
            df_result = self._fill_with_mode(df_result)
        else:  # default
            df_result = self._fill_with_defaults(df_result, entity_type)
        
        null_count = df_result.isnull().sum().sum()
        logger.info(f"Remaining nulls after handling: {null_count}")
        
        return df_result
    
    def _fill_with_defaults(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> pd.DataFrame:
        """Fill nulls with default values based on column type."""
        for col in df.columns:
            if df[col].isnull().any():
                fill_value = self._get_fill_value(col, df[col].dtype)
                df[col] = df[col].fillna(fill_value)
                logger.debug(f"Filled {col} with {fill_value}")
        return df
    
    def _fill_with_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric nulls with median, others with defaults."""
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                else:
                    fill_value = self.FILL_VALUES.get(str(df[col].dtype), None)
                df[col] = df[col].fillna(fill_value)
        return df
    
    def _fill_with_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill categorical nulls with mode, others with defaults."""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    fill_value = df[col].mode()
                    fill_value = fill_value.iloc[0] if len(fill_value) > 0 else 'UNKNOWN'
                elif pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                else:
                    fill_value = self.FILL_VALUES.get(str(df[col].dtype), None)
                df[col] = df[col].fillna(fill_value)
        return df
    
    def _get_fill_value(self, column: str, dtype: Any) -> Any:
        """Get appropriate fill value for a column."""
        # Check for column-specific fill
        if column in self.COLUMN_SPECIFIC_FILLS:
            return self.COLUMN_SPECIFIC_FILLS[column]
        
        # Fall back to dtype-based fill
        dtype_str = str(dtype)
        for key, value in self.FILL_VALUES.items():
            if key in dtype_str:
                return value
        
        return None
    
    def _check_required_columns(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> None:
        """Check that required columns don't have nulls."""
        try:
            required = CanonicalSchema.get_required_columns(entity_type)
        except KeyError:
            return  # Entity type not found, skip check
        
        for col in required:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(
                    f"Required column {col} has {null_count} null values in {entity_type}"
                )
    
    def handle_specific_column(
        self,
        df: pd.DataFrame,
        column: str,
        fill_value: Any = None
    ) -> pd.DataFrame:
        """
        Handle nulls in a specific column.
        
        Args:
            df: Input DataFrame
            column: Column to handle
            fill_value: Value to use for filling (auto-detected if None)
        
        Returns:
            DataFrame with column nulls handled
        """
        df_result = df.copy()
        
        if fill_value is None:
            fill_value = self._get_fill_value(column, df_result[column].dtype)
        
        df_result[column] = df_result[column].fillna(fill_value)
        logger.debug(f"Handled nulls in column {column}")
        
        return df_result
    
    def get_null_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate report on null values in DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with null statistics
        """
        null_counts = df.isnull().sum()
        total_rows = len(df)
        
        report = {
            'total_rows': total_rows,
            'columns_with_nulls': [],
            'null_details': {}
        }
        
        for col in df.columns:
            null_count = null_counts[col]
            if null_count > 0:
                report['columns_with_nulls'].append(col)
                report['null_details'][col] = {
                    'null_count': int(null_count),
                    'null_percentage': round(null_count / total_rows * 100, 2),
                    'non_null_count': int(total_rows - null_count)
                }
        
        report['total_null_cells'] = int(null_counts.sum())
        report['columns_without_nulls'] = len(df.columns) - len(report['columns_with_nulls'])
        
        return report
