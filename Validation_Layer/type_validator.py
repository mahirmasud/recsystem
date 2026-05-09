"""Data type validator."""

import pandas as pd
from typing import Dict, Any, List
from shared.logger import get_logger
from Standardized_Data_Layer.canonical_schema import CanonicalSchema


logger = get_logger(__name__)


class TypeValidator:
    """Validates data types match schema."""
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """Validate data types."""
        issues = []
        
        try:
            expected_dtypes = CanonicalSchema.get_all_dtypes(entity_type)
        except KeyError:
            return {'issues': [], 'passed': True}
        
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                continue
            
            actual_dtype = str(df[col].dtype)
            
            if not self._dtypes_compatible(actual_dtype, expected_dtype):
                issues.append({
                    'validation_type': 'type',
                    'column': col,
                    'issue': f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}",
                    'expected': expected_dtype,
                    'actual': actual_dtype,
                    'severity': 'warning'
                })
        
        return {'issues': issues, 'passed': len(issues) == 0}
    
    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if dtypes are compatible."""
        if actual == expected:
            return True
        
        numeric = ['int64', 'int32', 'float64', 'float32']
        string = ['string', 'object', 'str']
        
        if expected in numeric and actual in numeric:
            return True
        if expected in string and actual in string:
            return True
        
        return False
