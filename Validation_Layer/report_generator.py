"""Validation report generator."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


class ValidationReportGenerator:
    """Generates validation reports."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize report generator."""
        self.output_dir = Path(output_dir) if output_dir else Constants.REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        validation_results: Dict[str, Any],
        format: str = 'json'
    ) -> str:
        """Generate validation report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = {
            'report_type': 'data_validation',
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'overall_score': validation_results.get('overall_score', 0),
                'total_issues': validation_results.get('total_issues', 0),
                'critical_issues': validation_results.get('critical_issues', 0),
                'warning_issues': validation_results.get('warning_issues', 0)
            },
            'entity_details': validation_results.get('entity_results', {}),
            'grade': self._calculate_grade(validation_results.get('overall_score', 0))
        }
        
        if format == 'json':
            filepath = self.output_dir / f"validation_report_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            filepath = self.output_dir / f"validation_report_{timestamp}.txt"
            with open(filepath, 'w') as f:
                f.write(self._format_text_report(report))
        
        logger.info(f"Generated validation report: {filepath}")
        return str(filepath)
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 95:
            return 'A - Excellent'
        elif score >= 85:
            return 'B - Good'
        elif score >= 75:
            return 'C - Acceptable'
        elif score >= 65:
            return 'D - Needs Improvement'
        else:
            return 'F - Critical Issues'
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as text."""
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Generated: {report['generated_at']}",
            f"Grade: {report['grade']}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Overall Score: {report['summary']['overall_score']}/100",
            f"Total Issues: {report['summary']['total_issues']}",
            f"Critical: {report['summary']['critical_issues']}",
            f"Warnings: {report['summary']['warning_issues']}",
            "",
            "ENTITY DETAILS",
            "-" * 40
        ]
        
        for entity, details in report['entity_details'].items():
            lines.append(f"\n{entity.upper()}")
            lines.append(f"  Quality Score: {details.get('quality_score', 'N/A')}")
            lines.append(f"  Row Count: {details.get('row_count', 'N/A')}")
            
            for issue_type in ['null_issues', 'type_issues', 'range_issues']:
                count = len(details.get(issue_type, []))
                if count > 0:
                    lines.append(f"  {issue_type.replace('_issues', '')}: {count} issues")
        
        lines.extend(["", "=" * 60])
        return "\n".join(lines)
    
    def generate_summary_email(
        self,
        validation_results: Dict[str, Any]
    ) -> str:
        """Generate email-ready summary."""
        score = validation_results.get('overall_score', 0)
        total = validation_results.get('total_issues', 0)
        critical = validation_results.get('critical_issues', 0)
        
        subject = f"Data Validation Report - Score: {score}/100"
        
        body = f"""
Data Validation Summary

Overall Quality Score: {score}/100
Total Issues Found: {total}
Critical Issues: {critical}

{'ACTION REQUIRED: Critical issues found!' if critical > 0 else 'All validations passed successfully.'}

Please review the detailed report for more information.
"""
        return f"Subject: {subject}\n\n{body}"
