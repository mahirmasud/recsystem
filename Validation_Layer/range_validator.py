"""Range validator for numeric values."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from shared.logger import get_logger
from shared.config import Config


logger = get_logger(__name__)


class RangeValidator:
    """Validates numeric values are within expected ranges."""
    
    DEFAULT_RANGES = {
        'quantity': (0, 10000),
        'price': (0, float('inf')),
        'unit_price': (0, float('inf')),
        'total_amount': (0, float('inf')),
        'discount': (0, float('inf')),
        'age': (0, 150),
        'stock_quantity': (0, float('inf'))
    }
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """Validate value ranges."""
        issues = []
        
        try:
            config = Config()
            custom_ranges = config.get_valid_ranges()
        except Exception:
            custom_ranges = {}
        
        ranges = {**self.DEFAULT_RANGES, **custom_ranges}
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min < min_val:
                issues.append({
                    'validation_type': 'range',
                    'column': col,
                    'issue': f"Column {col} has values below minimum {min_val} (found {col_min})",
                    'severity': 'warning'
                })
            
            if col_max > max_val:
                issues.append({
                    'validation_type': 'range',
                    'column': col,
                    'issue': f"Column {col} has values above maximum {max_val} (found {col_max})",
                    'severity': 'warning'
                })
        
        positive_cols = ['quantity', 'price', 'unit_price', 'total_amount', 'amount']
        for col in positive_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append({
                        'validation_type': 'range',
                        'column': col,
                        'issue': f"Column {col} has {negative_count} negative values",
                        'severity': 'critical'
                    })
        
        return {'issues': issues, 'passed': len(issues) == 0}
