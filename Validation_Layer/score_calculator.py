"""Quality score calculator for validation results."""

import pandas as pd
from typing import Dict, Any, List, Optional
from shared.logger import get_logger


logger = get_logger(__name__)


class QualityScoreCalculator:
    """Calculates quality scores based on validation results."""
    
    # Score weights by validation type
    VALIDATION_WEIGHTS = {
        'null': 0.25,
        'type': 0.15,
        'range': 0.15,
        'uniqueness': 0.20,
        'relationship': 0.15,
        'business': 0.10
    }
    
    # Severity penalties
    SEVERITY_PENALTIES = {
        'critical': 10.0,
        'warning': 3.0,
        'info': 0.5
    }
    
    # Base score (starts at 100, penalties reduce it)
    BASE_SCORE = 100.0
    
    # Minimum possible score
    MIN_SCORE = 0.0
    
    def __init__(
        self,
        validation_weights: Optional[Dict[str, float]] = None,
        severity_penalties: Optional[Dict[str, float]] = None
    ):
        """Initialize score calculator.
        
        Args:
            validation_weights: Custom weights per validation type
            severity_penalties: Custom penalties per severity level
        """
        self.weights = {**self.VALIDATION_WEIGHTS}
        if validation_weights:
            self.weights.update(validation_weights)
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
        
        self.penalties = {**self.SEVERITY_PENALTIES}
        if severity_penalties:
            self.penalties.update(severity_penalties)
    
    def calculate_entity_score(
        self,
        entity_result: Dict[str, Any]
    ) -> float:
        """Calculate quality score for a single entity.
        
        Args:
            entity_result: Validation result for an entity
        
        Returns:
            Quality score (0-100)
        """
        score = self.BASE_SCORE
        
        # Collect all issues by type
        issue_types = {
            'null': entity_result.get('null_issues', []),
            'type': entity_result.get('type_issues', []),
            'range': entity_result.get('range_issues', []),
            'uniqueness': entity_result.get('uniqueness_issues', []),
            'relationship': entity_result.get('relationship_issues', []),
            'business': entity_result.get('business_issues', [])
        }
        
        # Calculate penalty for each issue type
        for issue_type, issues in issue_types.items():
            weight = self.weights.get(issue_type, 0.1)
            type_penalty = self._calculate_type_penalty(issues)
            score -= type_penalty * weight
        
        # Ensure score is within bounds
        score = max(self.MIN_SCORE, min(self.BASE_SCORE, score))
        
        return round(score, 2)
    
    def _calculate_type_penalty(
        self,
        issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate penalty for a specific issue type.
        
        Args:
            issues: List of issues for a type
        
        Returns:
            Penalty value
        """
        if not issues:
            return 0.0
        
        total_penalty = 0.0
        
        for issue in issues:
            severity = issue.get('severity', 'warning')
            base_penalty = self.penalties.get(severity, 1.0)
            
            # Scale by violation count if available
            count = issue.get('violation_count', issue.get('null_count', 1))
            count_factor = min(count / 100, 5)  # Cap at 5x
            
            total_penalty += base_penalty * count_factor
        
        return total_penalty
    
    def calculate_overall_score(
        self,
        validation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score across all entities.
        
        Args:
            validation_results: Complete validation results
        
        Returns:
            Overall quality score (0-100)
        """
        entity_results = validation_results.get('entity_results', {})
        
        if not entity_results:
            return self.BASE_SCORE
        
        # Weight by entity row count
        total_rows = sum(
            r.get('row_count', 0) for r in entity_results.values()
        )
        
        if total_rows == 0:
            return self.BASE_SCORE
        
        weighted_scores = []
        for entity_type, result in entity_results.items():
            entity_score = self.calculate_entity_score(result)
            row_count = result.get('row_count', 0)
            weight = row_count / total_rows
            weighted_scores.append(entity_score * weight)
        
        overall_score = sum(weighted_scores)
        
        # Apply additional penalties for critical issues
        critical_count = validation_results.get('critical_issues', 0)
        if critical_count > 0:
            critical_penalty = min(critical_count * 2, 20)  # Max 20 point penalty
            overall_score -= critical_penalty
        
        overall_score = max(self.MIN_SCORE, min(self.BASE_SCORE, overall_score))
        
        return round(overall_score, 2)
    
    def calculate_dimension_scores(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate scores for each quality dimension.
        
        Args:
            validation_results: Complete validation results
        
        Returns:
            Scores per dimension
        """
        entity_results = validation_results.get('entity_results', {})
        
        dimensions = {
            'completeness': [],  # Null issues
            'accuracy': [],  # Type and range issues
            'consistency': [],  # Uniqueness and relationship issues
            'validity': []  # Business rule issues
        }
        
        for entity_type, result in entity_results.items():
            # Completeness (inverse of null issues)
            null_issues = len(result.get('null_issues', []))
            completeness = max(0, 100 - (null_issues * 5))
            dimensions['completeness'].append(completeness)
            
            # Accuracy (type and range)
            type_issues = len(result.get('type_issues', []))
            range_issues = len(result.get('range_issues', []))
            accuracy = max(0, 100 - ((type_issues + range_issues) * 5))
            dimensions['accuracy'].append(accuracy)
            
            # Consistency (uniqueness and relationships)
            unique_issues = len(result.get('uniqueness_issues', []))
            rel_issues = len(result.get('relationship_issues', []))
            consistency = max(0, 100 - ((unique_issues + rel_issues) * 5))
            dimensions['consistency'].append(consistency)
            
            # Validity (business rules)
            business_issues = len(result.get('business_issues', []))
            validity = max(0, 100 - (business_issues * 5))
            dimensions['validity'].append(validity)
        
        # Average across entities
        dimension_scores = {}
        for dim, scores in dimensions.items():
            if scores:
                dimension_scores[dim] = round(sum(scores) / len(scores), 2)
            else:
                dimension_scores[dim] = self.BASE_SCORE
        
        return dimension_scores
    
    def get_quality_grade(self, score: float) -> Dict[str, Any]:
        """Get letter grade and description for a score.
        
        Args:
            score: Quality score (0-100)
        
        Returns:
            Grade information
        """
        if score >= 95:
            return {
                'grade': 'A',
                'description': 'Excellent',
                'status': 'Production Ready'
            }
        elif score >= 85:
            return {
                'grade': 'B',
                'description': 'Good',
                'status': 'Minor Issues'
            }
        elif score >= 75:
            return {
                'grade': 'C',
                'description': 'Acceptable',
                'status': 'Review Recommended'
            }
        elif score >= 65:
            return {
                'grade': 'D',
                'description': 'Needs Improvement',
                'status': 'Action Required'
            }
        else:
            return {
                'grade': 'F',
                'description': 'Critical Issues',
                'status': 'Do Not Use'
            }
    
    def generate_score_report(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive score report.
        
        Args:
            validation_results: Complete validation results
        
        Returns:
            Detailed score report
        """
        overall_score = self.calculate_overall_score(validation_results)
        dimension_scores = self.calculate_dimension_scores(validation_results)
        grade_info = self.get_quality_grade(overall_score)
        
        entity_scores = {}
        for entity_type, result in validation_results.get('entity_results', {}).items():
            entity_scores[entity_type] = {
                'score': self.calculate_entity_score(result),
                'row_count': result.get('row_count', 0),
                'issue_count': sum([
                    len(result.get(f'{t}_issues', []))
                    for t in ['null', 'type', 'range', 'uniqueness', 'relationship', 'business']
                ])
            }
        
        return {
            'overall_score': overall_score,
            'grade': grade_info['grade'],
            'description': grade_info['description'],
            'status': grade_info['status'],
            'dimension_scores': dimension_scores,
            'entity_scores': entity_scores,
            'total_issues': validation_results.get('total_issues', 0),
            'critical_issues': validation_results.get('critical_issues', 0)
        }
