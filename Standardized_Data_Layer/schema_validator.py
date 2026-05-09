"""
Schema validator for standardized data.
Validates DataFrames against canonical schema definitions.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from shared.logger import get_logger
from Standardized_Data_Layer.canonical_schema import CanonicalSchema


logger = get_logger(__name__)


class SchemaValidator:
    """
    Validates DataFrames against canonical schema definitions.
    
    Checks:
    - Required columns present
    - Correct data types
    - Primary key uniqueness
    - No duplicate rows
    """
    
    def __init__(self):
        """Initialize schema validator."""
        self.validation_errors: List[Dict[str, Any]] = []
        self.validation_warnings: List[Dict[str, Any]] = []
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str,
        strict: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate DataFrame against canonical schema.
        
        Args:
            df: DataFrame to validate
            entity_type: Entity type name
            strict: If True, fail on warnings too
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info(f"Validating schema for {entity_type}")
        
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check entity type exists
        if not CanonicalSchema.validate_entity_type(entity_type):
            self.validation_errors.append({
                'type': 'unknown_entity',
                'message': f"Unknown entity type: {entity_type}"
            })
            return False, self._build_report(entity_type, df)
        
        # Run validations
        self._check_required_columns(df, entity_type)
        self._check_data_types(df, entity_type)
        self._check_primary_key_uniqueness(df, entity_type)
        self._check_duplicates(df, entity_type)
        
        is_valid = len(self.validation_errors) == 0
        if strict:
            is_valid = is_valid and len(self.validation_warnings) == 0
        
        return is_valid, self._build_report(entity_type, df)
    
    def _check_required_columns(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> None:
        """Check that all required columns are present."""
        try:
            required = CanonicalSchema.get_required_columns(entity_type)
        except KeyError:
            return
        
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            self.validation_errors.append({
                'type': 'missing_required_columns',
                'entity_type': entity_type,
                'columns': missing,
                'message': f"Missing required columns: {missing}"
            })
        else:
            logger.debug(f"All required columns present for {entity_type}")
    
    def _check_data_types(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> None:
        """Check column data types match schema."""
        try:
            expected_dtypes = CanonicalSchema.get_all_dtypes(entity_type)
        except KeyError:
            return
        
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                continue
            
            actual_dtype = str(df[col].dtype)
            
            # Normalize dtype comparison
            if not self._dtypes_match(actual_dtype, expected_dtype):
                self.validation_warnings.append({
                    'type': 'dtype_mismatch',
                    'entity_type': entity_type,
                    'column': col,
                    'expected': expected_dtype,
                    'actual': actual_dtype,
                    'message': f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}"
                })
    
    def _dtypes_match(self, actual: str, expected: str) -> bool:
        """Check if dtypes match (with some flexibility)."""
        # Exact match
        if actual == expected:
            return True
        
        # Numeric flexibility
        numeric_dtypes = ['int64', 'int32', 'float64', 'float32']
        if expected in numeric_dtypes and actual in numeric_dtypes:
            return True
        
        # String flexibility
        string_dtypes = ['string', 'object', 'str']
        if expected in string_dtypes and actual in string_dtypes:
            return True
        
        return False
    
    def _check_primary_key_uniqueness(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> None:
        """Check primary key column has unique values."""
        try:
            pk = CanonicalSchema.get_primary_key(entity_type)
        except KeyError:
            return
        
        if pk not in df.columns:
            return
        
        duplicates = df[pk].duplicated().sum()
        
        if duplicates > 0:
            self.validation_errors.append({
                'type': 'primary_key_not_unique',
                'entity_type': entity_type,
                'column': pk,
                'duplicate_count': int(duplicates),
                'message': f"Primary key {pk} has {duplicates} duplicate values"
            })
        else:
            logger.debug(f"Primary key {pk} is unique for {entity_type}")
    
    def _check_duplicates(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> None:
        """Check for duplicate rows."""
        total_duplicates = df.duplicated().sum()
        
        if total_duplicates > 0:
            self.validation_warnings.append({
                'type': 'duplicate_rows',
                'entity_type': entity_type,
                'count': int(total_duplicates),
                'message': f"Found {total_duplicates} duplicate rows"
            })
    
    def _build_report(
        self,
        entity_type: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build validation report."""
        return {
            'entity_type': entity_type,
            'row_count': len(df),
            'column_count': len(df.columns),
            'is_valid': len(self.validation_errors) == 0,
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings),
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    def validate_multiple(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """
        Validate multiple DataFrames.
        
        Args:
            dataframes: Dictionary mapping entity type to DataFrame
        
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for entity_type, df in dataframes.items():
            is_valid, report = self.validate(df, entity_type)
            results[entity_type] = (is_valid, report)
        
        return results
    
    def get_validation_summary(
        self,
        results: Dict[str, Tuple[bool, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Args:
            results: Validation results from validate_multiple
        
        Returns:
            Summary dictionary
        """
        total_valid = sum(1 for is_valid, _ in results.values() if is_valid)
        total_invalid = len(results) - total_valid
        
        return {
            'total_entities': len(results),
            'valid_entities': total_valid,
            'invalid_entities': total_invalid,
            'all_valid': total_invalid == 0,
            'entity_results': {
                entity: {'valid': is_valid, 'errors': report['error_count']}
                for entity, (is_valid, report) in results.items()
            }
        }
