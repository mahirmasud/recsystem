"""Severity classifier for validation issues."""

import pandas as pd
from typing import Dict, Any, List, Optional
from shared.logger import get_logger


logger = get_logger(__name__)


class SeverityClassifier:
    """Classifies validation issues by severity level."""
    
    # Severity weights for scoring
    SEVERITY_WEIGHTS = {
        'critical': 1.0,
        'warning': 0.5,
        'info': 0.1
    }
    
    # Default severity mappings by validation type
    DEFAULT_SEVERITY_MAP = {
        'null': {
            'primary_key': 'critical',
            'required_field': 'critical',
            'optional_field': 'warning'
        },
        'type': {
            'mismatch': 'warning',
            'coercion_failed': 'critical'
        },
        'range': {
            'negative_value': 'critical',
            'out_of_bounds': 'warning',
            'extreme_outlier': 'warning'
        },
        'uniqueness': {
            'primary_key_duplicate': 'critical',
            'unique_constraint_violation': 'warning',
            'row_duplicate': 'warning'
        },
        'relationship': {
            'orphaned_record': 'critical',
            'referential_integrity': 'critical',
            'missing_reference': 'warning'
        },
        'business': {
            'rule_violation': 'warning',
            'critical_rule_violation': 'critical'
        }
    }
    
    def __init__(self, severity_map: Optional[Dict[str, Dict]] = None):
        """Initialize severity classifier.
        
        Args:
            severity_map: Custom severity mappings
        """
        self.severity_map = {**self.DEFAULT_SEVERITY_MAP}
        if severity_map:
            for validation_type, mappings in severity_map.items():
                if validation_type not in self.severity_map:
                    self.severity_map[validation_type] = {}
                self.severity_map[validation_type].update(mappings)
    
    def classify(self, issue: Dict[str, Any]) -> str:
        """Classify the severity of a validation issue.
        
        Args:
            issue: Validation issue dictionary
        
        Returns:
            Severity level (critical, warning, info)
        """
        # If severity already specified, use it
        if 'severity' in issue and issue['severity'] in self.SEVERITY_WEIGHTS:
            return issue['severity']
        
        validation_type = issue.get('validation_type', '')
        subtype = self._get_subtype(issue)
        
        # Look up severity in map
        if validation_type in self.severity_map:
            type_map = self.severity_map[validation_type]
            if subtype in type_map:
                return type_map[subtype]
        
        # Default severity based on characteristics
        return self._infer_severity(issue)
    
    def _get_subtype(self, issue: Dict[str, Any]) -> str:
        """Extract subtype from issue.
        
        Args:
            issue: Validation issue dictionary
        
        Returns:
            Subtype string
        """
        constraint_type = issue.get('constraint_type', '')
        column = issue.get('column', '')
        issue_text = issue.get('issue', '').lower()
        
        # Check for primary key references
        if 'primary' in constraint_type or 'primary' in issue_text:
            return 'primary_key'
        
        # Check for required/mandatory fields
        if 'required' in issue_text or 'critical' in issue_text:
            return 'required_field'
        
        # Check for duplicates
        if 'duplicate' in issue_text:
            if 'primary' in issue_text:
                return 'primary_key_duplicate'
            return 'unique_constraint_violation'
        
        # Check for orphaned records
        if 'orphan' in issue_text:
            return 'orphaned_record'
        
        # Check for negative values
        if 'negative' in issue_text:
            return 'negative_value'
        
        # Check for range issues
        if 'range' in issue_text or 'bounds' in issue_text:
            return 'out_of_bounds'
        
        # Check for type mismatches
        if 'type' in issue.get('validation_type', ''):
            if 'coerce' in issue_text or 'convert' in issue_text:
                return 'coercion_failed'
            return 'mismatch'
        
        # Check for business rule violations
        if issue.get('severity') == 'critical':
            return 'critical_rule_violation'
        
        return 'rule_violation'
    
    def _infer_severity(self, issue: Dict[str, Any]) -> str:
        """Infer severity from issue characteristics.
        
        Args:
            issue: Validation issue dictionary
        
        Returns:
            Inferred severity level
        """
        # Critical if affects data integrity
        if issue.get('validation_type') in ['relationship', 'uniqueness']:
            if 'primary' in str(issue.get('constraint_type', '')):
                return 'critical'
        
        # Critical if large percentage affected
        violation_pct = issue.get('violation_percentage', 0)
        null_pct = issue.get('null_percentage', 0)
        
        if violation_pct > 20 or null_pct > 20:
            return 'critical'
        elif violation_pct > 5 or null_pct > 5:
            return 'warning'
        
        # Default to warning for most issues
        return 'warning'
    
    def batch_classify(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify severity for multiple issues.
        
        Args:
            issues: List of validation issues
        
        Returns:
            Issues with classified severity
        """
        classified = []
        for issue in issues:
            classified_issue = issue.copy()
            classified_issue['severity'] = self.classify(issue)
            classified.append(classified_issue)
        
        return classified
    
    def get_severity_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of issues by severity.
        
        Args:
            issues: List of validation issues
        
        Returns:
            Count of issues per severity level
        """
        distribution = {'critical': 0, 'warning': 0, 'info': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'warning')
            if severity in distribution:
                distribution[severity] += 1
        
        return distribution
    
    def filter_by_severity(
        self,
        issues: List[Dict[str, Any]],
        min_severity: str = 'warning'
    ) -> List[Dict[str, Any]]:
        """Filter issues by minimum severity.
        
        Args:
            issues: List of validation issues
            min_severity: Minimum severity to include
        
        Returns:
            Filtered list of issues
        """
        severity_order = {'critical': 2, 'warning': 1, 'info': 0}
        min_level = severity_order.get(min_severity, 0)
        
        return [
            issue for issue in issues
            if severity_order.get(issue.get('severity', 'info'), 0) >= min_level
        ]
    
    def calculate_weighted_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate weighted score based on severity.
        
        Higher score = more severe issues
        
        Args:
            issues: List of validation issues
        
        Returns:
            Weighted severity score
        """
        total_weight = 0.0
        
        for issue in issues:
            severity = issue.get('severity', 'warning')
            weight = self.SEVERITY_WEIGHTS.get(severity, 0.5)
            
            # Adjust weight by violation count if available
            count = issue.get('violation_count', issue.get('null_count', 1))
            total_weight += weight * min(count / 100, 10)  # Cap at 10x
        
        return round(total_weight, 2)
