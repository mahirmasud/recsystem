"""Uniqueness validator for detecting duplicate records."""

import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
from shared.logger import get_logger
from Standardized_Data_Layer.canonical_schema import CanonicalSchema


logger = get_logger(__name__)


class UniquenessValidator:
    """Validates uniqueness constraints on primary keys and unique columns."""
    
    # Primary keys that must be unique per entity type
    PRIMARY_KEYS = {
        'users': ['user_id'],
        'items': ['item_id'],
        'transactions': ['transaction_id'],
        'interactions': ['interaction_id'],
        'categories': ['category_id']
    }
    
    # Additional unique constraints per entity type
    UNIQUE_CONSTRAINTS = {
        'users': [['email']],
        'items': [['sku']],  # Optional SKU uniqueness
        'transactions': [],
        'interactions': [],
        'categories': [['name']]
    }
    
    def __init__(self, primary_keys: Optional[Dict[str, List[str]]] = None):
        """Initialize uniqueness validator.
        
        Args:
            primary_keys: Custom primary keys per entity type
        """
        self.primary_keys = primary_keys or self.PRIMARY_KEYS
        self.unique_constraints = self.UNIQUE_CONSTRAINTS
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """Validate uniqueness constraints.
        
        Args:
            df: DataFrame to validate
            entity_type: Type of entity
        
        Returns:
            Dictionary with issues list and passed flag
        """
        logger.info(f"Validating uniqueness for {entity_type}")
        issues = []
        
        # Check primary key uniqueness
        pk_cols = self.primary_keys.get(entity_type, [])
        if pk_cols:
            pk_issues = self._check_uniqueness(df, pk_cols, entity_type, 'primary_key')
            issues.extend(pk_issues)
        
        # Check additional unique constraints
        unique_cols_list = self.unique_constraints.get(entity_type, [])
        for unique_cols in unique_cols_list:
            # Only check if all columns exist
            if all(col in df.columns for col in unique_cols):
                constraint_issues = self._check_uniqueness(
                    df, unique_cols, entity_type, 'unique_constraint'
                )
                issues.extend(constraint_issues)
        
        # Check for completely duplicate rows
        dup_issues = self._check_duplicate_rows(df, entity_type)
        issues.extend(dup_issues)
        
        passed = len([i for i in issues if i['severity'] == 'critical']) == 0
        
        return {
            'issues': issues,
            'passed': passed,
            'duplicate_count': sum(i.get('duplicate_count', 0) for i in issues),
            'primary_key_checked': pk_cols
        }
    
    def _check_uniqueness(
        self,
        df: pd.DataFrame,
        columns: List[str],
        entity_type: str,
        constraint_type: str
    ) -> List[Dict[str, Any]]:
        """Check uniqueness for specified columns.
        
        Args:
            df: DataFrame to check
            columns: Columns to check for uniqueness
            entity_type: Entity type
            constraint_type: Type of constraint (primary_key or unique_constraint)
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Remove nulls before checking duplicates
        df_clean = df.dropna(subset=columns)
        
        if df_clean.empty:
            return issues
        
        duplicates = df_clean[df_clean.duplicated(subset=columns, keep=False)]
        duplicate_count = len(duplicates)
        
        if duplicate_count > 0:
            # Get unique duplicate groups
            duplicate_groups = duplicates.groupby(columns).size().reset_index(name='count')
            num_duplicate_groups = len(duplicate_groups)
            
            severity = 'critical' if constraint_type == 'primary_key' else 'warning'
            
            issues.append({
                'validation_type': 'uniqueness',
                'constraint_type': constraint_type,
                'columns': columns,
                'issue': f"Found {duplicate_count} duplicate records across {num_duplicate_groups} groups for {columns} in {entity_type}",
                'severity': severity,
                'duplicate_count': int(duplicate_count),
                'duplicate_groups': int(num_duplicate_groups),
                'sample_duplicates': duplicate_groups.head(10).to_dict('records')
            })
        
        return issues
    
    def _check_duplicate_rows(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """Check for completely duplicate rows.
        
        Args:
            df: DataFrame to check
            entity_type: Entity type
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Check for exact duplicate rows
        duplicates = df[df.duplicated(keep=False)]
        duplicate_count = len(duplicates)
        
        if duplicate_count > 0:
            # Get count of duplicate groups
            duplicate_groups = duplicates.groupby(list(df.columns)).size().reset_index(name='count')
            num_duplicate_groups = len(duplicate_groups)
            
            issues.append({
                'validation_type': 'uniqueness',
                'constraint_type': 'full_row_duplicate',
                'columns': list(df.columns),
                'issue': f"Found {duplicate_count} completely duplicate rows across {num_duplicate_groups} groups in {entity_type}",
                'severity': 'warning',
                'duplicate_count': int(duplicate_count),
                'duplicate_groups': int(num_duplicate_groups)
            })
        
        return issues
    
    def get_uniqueness_summary(self, df: pd.DataFrame, entity_type: str) -> Dict[str, Any]:
        """Get summary of uniqueness metrics.
        
        Args:
            df: DataFrame to analyze
            entity_type: Entity type
        
        Returns:
            Summary statistics about uniqueness
        """
        pk_cols = self.primary_keys.get(entity_type, [])
        
        summary = {
            'total_rows': len(df),
            'primary_key_columns': pk_cols,
            'uniqueness_metrics': {}
        }
        
        # Check uniqueness for each column
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df) - df[col].isna().sum()
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            summary['uniqueness_metrics'][col] = {
                'unique_values': int(unique_count),
                'non_null_count': int(total_count),
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'is_unique_key': unique_count == total_count and total_count == len(df)
            }
        
        # Check primary key specifically
        if pk_cols and all(col in df.columns for col in pk_cols):
            pk_unique = df[pk_cols].drop_duplicates()
            summary['primary_key_status'] = {
                'is_unique': len(pk_unique) == len(df),
                'unique_count': len(pk_unique),
                'total_count': len(df),
                'duplicate_count': len(df) - len(pk_unique)
            }
        
        return summary
