"""
Explanation Logger - Logs detailed explanations for rule applications.

Responsible for:
- Recording why each rule was applied
- Tracking score changes per item
- Generating human-readable explanations
- Supporting audit and compliance requirements
- Providing transparency for recommendation decisions
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from shared.logger import get_logger
from shared.exceptions import RuleEngineError

logger = get_logger(__name__)


class ExplanationEntry:
    """Represents a single explanation entry."""
    
    def __init__(self,
                 item_id: str,
                 rule_id: str,
                 rule_name: str,
                 rule_type: str,
                 action: str,
                 reason: str,
                 score_before: float,
                 score_after: float,
                 metadata: Optional[Dict[str, Any]] = None):
        self.item_id = item_id
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.rule_type = rule_type
        self.action = action
        self.reason = reason
        self.score_before = score_before
        self.score_after = score_after
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'item_id': self.item_id,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'rule_type': self.rule_type,
            'action': self.action,
            'reason': self.reason,
            'score_before': self.score_before,
            'score_after': self.score_after,
            'score_change': self.score_after - self.score_before,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def to_human_readable(self) -> str:
        """Generate human-readable explanation."""
        change = self.score_after - self.score_before
        change_str = f"+{change:.4f}" if change > 0 else f"{change:.4f}"
        
        if self.action == 'filter':
            return f"Item {self.item_id} was filtered because: {self.reason}"
        elif self.action == 'boost':
            return (f"Item {self.item_id} score boosted from {self.score_before:.4f} "
                   f"to {self.score_after:.4f} ({change_str}) because: {self.reason}")
        elif self.action == 'reduce':
            return (f"Item {self.item_id} score reduced from {self.score_before:.4f} "
                   f"to {self.score_after:.4f} ({change_str}) because: {self.reason}")
        else:
            return f"Item {self.item_id} affected by rule '{self.rule_name}': {self.reason}"


class ExplanationLogger:
    """Logs and manages explanations for rule applications."""
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize ExplanationLogger.
        
        Args:
            log_path: Optional path to save explanation logs
        """
        self.log_path = Path(log_path) if log_path else None
        self.explanations: List[ExplanationEntry] = []
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log(self,
            item_id: str,
            rule_id: str,
            rule_name: str,
            rule_type: str,
            action: str,
            reason: str,
            score_before: float,
            score_after: float,
            metadata: Optional[Dict[str, Any]] = None):
        """
        Log an explanation entry.
        
        Args:
            item_id: Item identifier
            rule_id: Rule identifier
            rule_name: Human-readable rule name
            rule_type: Type of rule
            action: Action taken (filter, boost, reduce, etc.)
            reason: Reason for the action
            score_before: Score before rule application
            score_after: Score after rule application
            metadata: Additional metadata
        """
        entry = ExplanationEntry(
            item_id=item_id,
            rule_id=rule_id,
            rule_name=rule_name,
            rule_type=rule_type,
            action=action,
            reason=reason,
            score_before=score_before,
            score_after=score_after,
            metadata=metadata
        )
        
        self.explanations.append(entry)
        
        logger.debug(
            f"Logged explanation: {entry.to_human_readable()}"
        )
    
    def log_filter(self,
                   item_id: str,
                   rule: Dict[str, Any],
                   reason: str,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Log a filter action.
        
        Args:
            item_id: Filtered item ID
            rule: Rule information
            reason: Filter reason
            metadata: Additional metadata
        """
        self.log(
            item_id=item_id,
            rule_id=rule.get('id', 'unknown'),
            rule_name=rule.get('name', 'Unknown Rule'),
            rule_type='filter',
            action='filter',
            reason=reason,
            score_before=0,
            score_after=0,
            metadata=metadata
        )
    
    def log_boost(self,
                  item_id: str,
                  rule: Dict[str, Any],
                  reason: str,
                  score_before: float,
                  score_after: float,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log a boost action.
        
        Args:
            item_id: Boosted item ID
            rule: Rule information
            reason: Boost reason
            score_before: Score before boost
            score_after: Score after boost
            metadata: Additional metadata
        """
        self.log(
            item_id=item_id,
            rule_id=rule.get('id', 'unknown'),
            rule_name=rule.get('name', 'Unknown Rule'),
            rule_type='boost',
            action='boost',
            reason=reason,
            score_before=score_before,
            score_after=score_after,
            metadata=metadata
        )
    
    def get_item_explanations(self, item_id: str) -> List[ExplanationEntry]:
        """
        Get all explanations for a specific item.
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of explanation entries for the item
        """
        return [e for e in self.explanations if e.item_id == item_id]
    
    def get_rule_explanations(self, rule_id: str) -> List[ExplanationEntry]:
        """
        Get all explanations for a specific rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            List of explanation entries for the rule
        """
        return [e for e in self.explanations if e.rule_id == rule_id]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all explanations.
        
        Returns:
            Dictionary with explanation summary
        """
        if not self.explanations:
            return {'total_explanations': 0}
        
        # Count by rule type
        by_type = {}
        for e in self.explanations:
            by_type[e.rule_type] = by_type.get(e.rule_type, 0) + 1
        
        # Count by action
        by_action = {}
        for e in self.explanations:
            by_action[e.action] = by_action.get(e.action, 0) + 1
        
        # Calculate average score changes
        boosts = [e for e in self.explanations if e.action == 'boost']
        avg_boost = sum(e.score_after - e.score_before for e in boosts) / len(boosts) if boosts else 0
        
        return {
            'session_id': self.session_id,
            'total_explanations': len(self.explanations),
            'by_rule_type': by_type,
            'by_action': by_action,
            'items_affected': len(set(e.item_id for e in self.explanations)),
            'rules_applied': len(set(e.rule_id for e in self.explanations)),
            'average_boost_change': avg_boost,
            'timestamp_range': {
                'start': self.explanations[0].timestamp.isoformat(),
                'end': self.explanations[-1].timestamp.isoformat()
            }
        }
    
    def generate_report(self, 
                       output_path: Optional[str] = None,
                       format: str = 'json') -> str:
        """
        Generate a detailed explanation report.
        
        Args:
            output_path: Path to save report (optional)
            format: Output format ('json' or 'text')
            
        Returns:
            Report content as string
        """
        if format == 'json':
            report = json.dumps({
                'session_id': self.session_id,
                'generated_at': datetime.now().isoformat(),
                'summary': self.get_summary(),
                'explanations': [e.to_dict() for e in self.explanations]
            }, indent=2)
        else:
            lines = [
                "=" * 80,
                "RULE EXPLANATION REPORT",
                f"Session: {self.session_id}",
                f"Generated: {datetime.now().isoformat()}",
                "=" * 80,
                "",
                "SUMMARY",
                "-" * 40,
            ]
            
            summary = self.get_summary()
            for key, value in summary.items():
                if key not in ['timestamp_range', 'session_id']:
                    lines.append(f"  {key}: {value}")
            
            lines.extend([
                "",
                "DETAILED EXPLANATIONS",
                "-" * 40,
            ])
            
            for i, e in enumerate(self.explanations[:100], 1):  # Limit to 100
                lines.append(f"\n{i}. {e.to_human_readable()}")
            
            lines.append("\n" + "=" * 80)
            report = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(report)
            logger.info(f"Saved explanation report to {path}")
        
        return report
    
    def export_for_audit(self, output_path: str) -> str:
        """
        Export explanations in audit-friendly format.
        
        Args:
            output_path: Path to save audit file
            
        Returns:
            Path to saved file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        audit_data = {
            'audit_timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'total_decisions': len(self.explanations),
            'decisions': [e.to_dict() for e in self.explanations]
        }
        
        with open(path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Exported audit data to {path}")
        return str(path)
    
    def clear(self):
        """Clear all logged explanations."""
        self.explanations.clear()
        logger.info("Cleared explanation log")
    
    def __len__(self) -> int:
        """Return number of logged explanations."""
        return len(self.explanations)
