"""
Main data validator orchestrating all validation checks.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from shared.logger import get_logger
from shared.constants import Constants
from Validation_Layer.null_validator import NullValidator
from Validation_Layer.type_validator import TypeValidator
from Validation_Layer.range_validator import RangeValidator
from Validation_Layer.uniqueness_validator import UniquenessValidator
from Validation_Layer.relationship_validator import RelationshipValidator
from Validation_Layer.business_validator import BusinessValidator
from Validation_Layer.severity_classifier import SeverityClassifier
from Validation_Layer.score_calculator import QualityScoreCalculator


logger = get_logger(__name__)


class DataValidator:
    """
    Main validator orchestrating all validation checks.
    
    Coordinates:
    - Null validation
    - Type validation
    - Range validation
    - Uniqueness validation
    - Relationship validation
    - Business rule validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize all validators."""
        self.null_validator = NullValidator()
        self.type_validator = TypeValidator()
        self.range_validator = RangeValidator()
        self.uniqueness_validator = UniquenessValidator()
        self.relationship_validator = RelationshipValidator()
        self.business_validator = BusinessValidator()
        self.severity_classifier = SeverityClassifier()
        self.score_calculator = QualityScoreCalculator()
        
        self.all_issues: List[Dict[str, Any]] = []
    
    def validate(
        self,
        dataframes: Dict[str, pd.DataFrame],
        run_all: bool = True
    ) -> Dict[str, Any]:
        """
        Run all validations on provided DataFrames.
        
        Args:
            dataframes: Dictionary mapping entity type to DataFrame
            run_all: Whether to run all validators
        
        Returns:
            Complete validation report
        """
        logger.info("Starting comprehensive validation")
        self.all_issues = []
        
        results = {
            'entity_results': {},
            'overall_score': 0.0,
            'total_issues': 0,
            'critical_issues': 0,
            'warning_issues': 0
        }
        
        for entity_type, df in dataframes.items():
            entity_result = self._validate_entity(df, entity_type, run_all)
            results['entity_results'][entity_type] = entity_result
            
            # Aggregate issues
            all_entity_issues = (
                entity_result.get('null_issues', []) +
                entity_result.get('type_issues', []) +
                entity_result.get('range_issues', []) +
                entity_result.get('uniqueness_issues', []) +
                entity_result.get('relationship_issues', []) +
                entity_result.get('business_issues', [])
            )
            
            self.all_issues.extend(all_entity_issues)
            results['total_issues'] += len(all_entity_issues)
            results['critical_issues'] += sum(1 for i in all_entity_issues if i.get('severity') == 'critical')
            results['warning_issues'] += sum(1 for i in all_entity_issues if i.get('severity') == 'warning')
        
        # Calculate overall score
        results['overall_score'] = self.score_calculator.calculate_overall_score(results)
        
        logger.info(f"Validation complete. Score: {results['overall_score']:.2f}")
        return results
    
    def _validate_entity(
        self,
        df: pd.DataFrame,
        entity_type: str,
        run_all: bool
    ) -> Dict[str, Any]:
        """Validate a single entity."""
        result = {
            'entity_type': entity_type,
            'row_count': len(df),
            'null_issues': [],
            'type_issues': [],
            'range_issues': [],
            'uniqueness_issues': [],
            'relationship_issues': [],
            'business_issues': [],
            'quality_score': 0.0
        }
        
        # Null validation
        null_result = self.null_validator.validate(df, entity_type)
        result['null_issues'] = null_result.get('issues', [])
        
        # Type validation
        type_result = self.type_validator.validate(df, entity_type)
        result['type_issues'] = type_result.get('issues', [])
        
        if run_all:
            # Range validation
            range_result = self.range_validator.validate(df, entity_type)
            result['range_issues'] = range_result.get('issues', [])
            
            # Uniqueness validation
            uniqueness_result = self.uniqueness_validator.validate(df, entity_type)
            result['uniqueness_issues'] = uniqueness_result.get('issues', [])
        
        # Calculate quality score
        result['quality_score'] = self.score_calculator.calculate_entity_score(result)
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'total_issues': len(self.all_issues),
            'by_severity': {
                'critical': sum(1 for i in self.all_issues if i.get('severity') == 'critical'),
                'warning': sum(1 for i in self.all_issues if i.get('severity') == 'warning'),
                'info': sum(1 for i in self.all_issues if i.get('severity') == 'info')
            },
            'by_type': {
                'null': sum(1 for i in self.all_issues if i.get('validation_type') == 'null'),
                'type': sum(1 for i in self.all_issues if i.get('validation_type') == 'type'),
                'range': sum(1 for i in self.all_issues if i.get('validation_type') == 'range'),
                'uniqueness': sum(1 for i in self.all_issues if i.get('validation_type') == 'uniqueness')
            }
        }
