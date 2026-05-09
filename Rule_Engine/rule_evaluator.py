"""Rule Evaluator - Evaluates rule effectiveness."""
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class RuleEvaluator:
    """Evaluates the impact of business rules."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rule_stats = {}
        
    def evaluate_filter_impact(self, before: List[Dict], after: List[Dict], 
                                rule_name: str) -> Dict[str, Any]:
        """Evaluate impact of a filter rule."""
        filtered_count = len(before) - len(after)
        return {
            'rule': rule_name,
            'items_before': len(before),
            'items_after': len(after),
            'filtered_out': filtered_count,
            'filter_rate': filtered_count / len(before) if before else 0
        }
    
    def evaluate_boost_impact(self, candidates: List[Dict], 
                               rule_name: str) -> Dict[str, Any]:
        """Evaluate impact of a boost rule."""
        boosted = [c for c in candidates if c.get('score', 0) > c.get('original_score', c.get('score', 0))]
        return {
            'rule': rule_name,
            'total_candidates': len(candidates),
            'boosted_count': len(boosted),
            'avg_score_change': sum(c.get('score', 0) - c.get('original_score', 0) for c in boosted) / len(boosted) if boosted else 0
        }
    
    def track_rule_performance(self, rule_name: str, metrics: Dict[str, Any]):
        """Track rule performance over time."""
        if rule_name not in self.rule_stats:
            self.rule_stats[rule_name] = []
        self.rule_stats[rule_name].append(metrics)
        
    def get_rule_report(self) -> Dict[str, Any]:
        """Generate rule performance report."""
        report = {}
        for rule_name, stats in self.rule_stats.items():
            if stats:
                report[rule_name] = {
                    'evaluations': len(stats),
                    'avg_impact': sum(s.get('filter_rate', s.get('avg_score_change', 0)) for s in stats) / len(stats)
                }
        return report
