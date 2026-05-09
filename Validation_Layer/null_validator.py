"""Null validator for detecting missing values."""

import pandas as pd
from typing import Dict, Any, List, Optional, Set
from shared.logger import get_logger
from Standardized_Data_Layer.canonical_schema import CanonicalSchema


logger = get_logger(__name__)


class NullValidator:
    """Validates presence of required fields and detects null values."""
    
    # Columns that must never be null
    CRITICAL_COLUMNS = {
        'users': ['user_id', 'email'],
        'items': ['item_id', 'name'],
        'transactions': ['transaction_id', 'user_id', 'item_id', 'amount'],
        'interactions': ['interaction_id', 'user_id', 'item_id', 'timestamp'],
        'categories': ['category_id', 'name']
    }
    
    # Columns where nulls are warnings
    OPTIONAL_COLUMNS = {
        'users': ['age', 'gender', 'location'],
        'items': ['description', 'brand', 'image_url'],
        'transactions': ['discount', 'coupon_code'],
        'interactions': ['session_id', 'device_type'],
        'categories': ['parent_category_id']
    }
    
    def __init__(self, critical_columns: Optional[Dict[str, List[str]]] = None):
        """Initialize null validator.
        
        Args:
            critical_columns: Custom critical columns per entity type
        """
        self.critical_columns = critical_columns or self.CRITICAL_COLUMNS
        self.optional_columns = self.OPTIONAL_COLUMNS
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """Validate null values in DataFrame.
        
        Args:
            df: DataFrame to validate
            entity_type: Type of entity (users, items, transactions, etc.)
        
        Returns:
            Dictionary with issues list and passed flag
        """
        logger.info(f"Validating nulls for {entity_type}")
        issues = []
        
        # Get critical columns for this entity type
        critical_cols = self.critical_columns.get(entity_type, [])
        optional_cols = self.optional_columns.get(entity_type, [])
        
        # Check critical columns
        for col in critical_cols:
            if col not in df.columns:
                issues.append({
                    'validation_type': 'null',
                    'column': col,
                    'issue': f"Critical column '{col}' is missing from {entity_type}",
                    'severity': 'critical',
                    'row_count': len(df)
                })
                continue
            
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                issues.append({
                    'validation_type': 'null',
                    'column': col,
                    'issue': f"Critical column '{col}' has {null_count} null values ({null_percentage:.2f}%)",
                    'severity': 'critical',
                    'null_count': int(null_count),
                    'null_percentage': round(null_percentage, 2),
                    'row_indices': df[df[col].isna()].index.tolist()[:100]  # First 100
                })
        
        # Check optional columns (warnings only)
        for col in optional_cols:
            if col not in df.columns:
                continue
            
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                # Only warn if more than 50% are null
                if null_percentage > 50:
                    issues.append({
                        'validation_type': 'null',
                        'column': col,
                        'issue': f"Optional column '{col}' has {null_count} null values ({null_percentage:.2f}%)",
                        'severity': 'warning',
                        'null_count': int(null_count),
                        'null_percentage': round(null_percentage, 2)
                    })
        
        # Check all other columns for extreme null percentages
        checked_cols = set(critical_cols + optional_cols)
        for col in df.columns:
            if col in checked_cols:
                continue
            
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                if null_percentage > 90:
                    issues.append({
                        'validation_type': 'null',
                        'column': col,
                        'issue': f"Column '{col}' is {null_percentage:.2f}% null - consider removing",
                        'severity': 'warning',
                        'null_count': int(null_count),
                        'null_percentage': round(null_percentage, 2)
                    })
        
        passed = len([i for i in issues if i['severity'] == 'critical']) == 0
        
        return {
            'issues': issues,
            'passed': passed,
            'total_nulls': sum(i.get('null_count', 0) for i in issues),
            'critical_columns_checked': critical_cols,
            'optional_columns_checked': optional_cols
        }
    
    def get_null_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of null values across all columns.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Summary statistics about null values
        """
        null_counts = df.isna().sum()
        null_percentages = (df.isna().sum() / len(df)) * 100
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_with_nulls': int((null_counts > 0).sum()),
            'null_details': {}
        }
        
        for col in df.columns:
            if null_counts[col] > 0:
                summary['null_details'][col] = {
                    'null_count': int(null_counts[col]),
                    'null_percentage': round(null_percentages[col], 2),
                    'non_null_count': int(len(df) - null_counts[col])
                }
        
        return summary