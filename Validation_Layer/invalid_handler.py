"""Invalid record handler."""

import pandas as pd
from typing import Dict, Any, List, Optional
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


class InvalidRecordHandler:
    """Handles invalid records found during validation."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize handler."""
        self.output_dir = output_dir or Constants.REPORTS_DIR
    
    def extract_invalid_records(
        self,
        df: pd.DataFrame,
        issues: List[Dict[str, Any]],
        entity_type: str
    ) -> pd.DataFrame:
        """Extract records that have validation issues."""
        invalid_indices = set()
        
        for issue in issues:
            if issue.get('row_indices'):
                invalid_indices.update(issue['row_indices'])
        
        if not invalid_indices:
            return pd.DataFrame()
        
        return df.iloc[list(invalid_indices)].copy()
    
    def save_invalid_records(
        self,
        invalid_df: pd.DataFrame,
        entity_type: str,
        filename: Optional[str] = None
    ) -> str:
        """Save invalid records to file."""
        if invalid_df.empty:
            logger.info(f"No invalid records to save for {entity_type}")
            return None
        
        if filename is None:
            filename = f"{entity_type}_invalid_records.parquet"
        
        filepath = self.output_dir / filename
        invalid_df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(invalid_df)} invalid records to {filepath}")
        return str(filepath)
    
    def create_correction_report(
        self,
        issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create report with suggested corrections."""
        corrections = []
        
        for issue in issues:
            correction = {
                'issue': issue.get('issue'),
                'column': issue.get('column'),
                'severity': issue.get('severity'),
                'suggested_action': self._get_suggested_action(issue)
            }
            corrections.append(correction)
        
        return {'corrections': corrections, 'total_issues': len(corrections)}
    
    def _get_suggested_action(self, issue: Dict[str, Any]) -> str:
        """Get suggested action for an issue."""
        issue_type = issue.get('validation_type', '')
        
        if issue_type == 'null':
            return "Fill null values with appropriate defaults or remove records"
        elif issue_type == 'type':
            return "Convert column to expected data type"
        elif issue_type == 'range':
            return "Review and correct out-of-range values"
        elif issue_type == 'uniqueness':
            return "Remove duplicate records or investigate data source"
        elif issue_type == 'relationship':
            return "Fix referential integrity or add missing referenced records"
        else:
            return "Manual review required"
