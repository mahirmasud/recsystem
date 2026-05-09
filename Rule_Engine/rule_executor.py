"""
Rule Executor - Main engine for executing business rules.

Responsible for:
- Orchestrating rule execution
- Managing rule application order
- Handling rule failures gracefully
- Collecting execution statistics
- Providing unified interface for all rule types
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from shared.logger import get_logger
from shared.exceptions import RuleEngineError
from .rule_parser import ParsedRule, RuleType, RuleAction
from .filter_rules import FilterRules
from .boost_rules import BoostRules
from .conditional_rules import ConditionalRules
from .context_rules import ContextRules
from .campaign_rules import CampaignRules

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Status of rule execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionResult:
    """Result of rule execution."""
    
    def __init__(self, 
                 status: ExecutionStatus,
                 recommendations: List[Dict[str, Any]],
                 rules_applied: List[str],
                 rules_skipped: List[str],
                 rules_failed: List[str],
                 explanations: List[Dict[str, Any]],
                 execution_time_ms: float):
        self.status = status
        self.recommendations = recommendations
        self.rules_applied = rules_applied
        self.rules_skipped = rules_skipped
        self.rules_failed = rules_failed
        self.explanations = explanations
        self.execution_time_ms = execution_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'recommendation_count': len(self.recommendations),
            'rules_applied': self.rules_applied,
            'rules_skipped': self.rules_skipped,
            'rules_failed': self.rules_failed,
            'explanations': self.explanations,
            'execution_time_ms': self.execution_time_ms
        }


class RuleExecutor:
    """Main engine for executing business rules on recommendations."""
    
    def __init__(self):
        """Initialize RuleExecutor with all rule handlers."""
        self.filter_rules = FilterRules()
        self.boost_rules = BoostRules()
        self.conditional_rules = ConditionalRules()
        self.context_rules = ContextRules()
        self.campaign_rules = CampaignRules()
        
        self.execution_history: List[Dict[str, Any]] = []
        self.explanations: List[Dict[str, Any]] = []
        
    def execute(self,
                recommendations: List[Dict[str, Any]],
                rules: List[ParsedRule],
                user_context: Dict[str, Any],
                item_catalog: Dict[str, Dict[str, Any]],
                stop_on_filter: bool = False) -> ExecutionResult:
        """
        Execute all rules on recommendations.
        
        Args:
            recommendations: List of recommendation items
            rules: List of parsed rules to apply
            user_context: User context dictionary
            item_catalog: Item metadata lookup
            stop_on_filter: If True, stop processing if all items filtered
            
        Returns:
            ExecutionResult with processed recommendations
        """
        start_time = datetime.now()
        
        if not recommendations:
            logger.warning("No recommendations to process")
            return ExecutionResult(
                status=ExecutionStatus.SKIPPED,
                recommendations=[],
                rules_applied=[],
                rules_skipped=[r.id for r in rules],
                rules_failed=[],
                explanations=[],
                execution_time_ms=0
            )
        
        rules_applied = []
        rules_skipped = []
        rules_failed = []
        self.explanations = []
        
        current_recommendations = recommendations.copy()
        
        # Process rules by type in order: filter -> conditional -> context -> campaign -> boost
        rule_order = [
            RuleType.FILTER,
            RuleType.CONDITIONAL,
            RuleType.CONTEXT,
            RuleType.CAMPAIGN,
            RuleType.BOOST,
        ]
        
        for rule_type in rule_order:
            type_rules = [r for r in rules if r.type == rule_type]
            
            for rule in type_rules:
                try:
                    result = self._execute_single_rule(
                        current_recommendations,
                        rule,
                        user_context,
                        item_catalog
                    )
                    
                    if result is not None:
                        current_recommendations = result
                        rules_applied.append(rule.id)
                        logger.debug(f"Rule '{rule.id}' applied successfully")
                    else:
                        rules_skipped.append(rule.id)
                        logger.debug(f"Rule '{rule.id}' skipped (conditions not met)")
                    
                    # Check if all items filtered
                    if stop_on_filter and not current_recommendations:
                        logger.warning("All items filtered, stopping execution")
                        break
                        
                except Exception as e:
                    logger.error(f"Rule '{rule.id}' failed: {e}")
                    rules_failed.append(rule.id)
                    self.explanations.append({
                        'rule_id': rule.id,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            if not current_recommendations:
                break
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Determine overall status
        if rules_failed:
            status = ExecutionStatus.PARTIAL if current_recommendations else ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.SUCCESS
        
        # Record execution
        self.execution_history.append({
            'timestamp': start_time.isoformat(),
            'status': status.value,
            'rules_applied': len(rules_applied),
            'rules_skipped': len(rules_skipped),
            'rules_failed': len(rules_failed),
            'input_count': len(recommendations),
            'output_count': len(current_recommendations),
            'execution_time_ms': execution_time_ms
        })
        
        logger.info(
            f"Rule execution completed: {len(rules_applied)} applied, "
            f"{len(rules_skipped)} skipped, {len(rules_failed)} failed. "
            f"Output: {len(current_recommendations)} items in {execution_time_ms:.2f}ms"
        )
        
        return ExecutionResult(
            status=status,
            recommendations=current_recommendations,
            rules_applied=rules_applied,
            rules_skipped=rules_skipped,
            rules_failed=rules_failed,
            explanations=self.explanations,
            execution_time_ms=execution_time_ms
        )
    
    def _execute_single_rule(self,
                            recommendations: List[Dict[str, Any]],
                            rule: ParsedRule,
                            user_context: Dict[str, Any],
                            item_catalog: Dict[str, Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a single rule.
        
        Args:
            recommendations: Current recommendations
            rule: Rule to execute
            user_context: User context
            item_catalog: Item catalog
            
        Returns:
            Modified recommendations or None if skipped
        """
        rule_type = rule.type
        
        if rule_type == RuleType.FILTER:
            result = self.filter_rules.apply(
                recommendations, rule, user_context, item_catalog
            )
            self._log_explanation(rule, 'filter', len(recommendations) - len(result))
            return result
        
        elif rule_type == RuleType.BOOST:
            result = self.boost_rules.apply(
                recommendations, rule, user_context, item_catalog
            )
            stats = self.boost_rules.get_boost_stats()
            self._log_explanation(rule, 'boost', stats.get('total_boosted', 0))
            return result
        
        elif rule_type == RuleType.CONDITIONAL:
            result = self.conditional_rules.apply(
                recommendations, rule, user_context, item_catalog
            )
            stats = self.conditional_rules.get_evaluation_stats()
            self._log_explanation(rule, 'conditional', stats.get('conditions_met', 0))
            return result
        
        elif rule_type == RuleType.CONTEXT:
            result = self.context_rules.apply(
                recommendations, rule, user_context, item_catalog
            )
            stats = self.context_rules.get_context_stats()
            self._log_explanation(rule, 'context', stats.get('contexts_matched', 0))
            return result
        
        elif rule_type == RuleType.CAMPAIGN:
            # Load campaign from rule if not already loaded
            if rule.id not in self.campaign_rules.campaigns:
                self.campaign_rules.load_from_rule(rule)
            
            result = self.campaign_rules.apply(
                recommendations, rule, user_context, item_catalog
            )
            stats = self.campaign_rules.get_campaign_stats()
            self._log_explanation(rule, 'campaign', stats.get('total_applications', 0))
            return result
        
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return recommendations
    
    def _log_explanation(self, 
                        rule: ParsedRule, 
                        rule_type: str, 
                        items_affected: int):
        """
        Log explanation for rule application.
        
        Args:
            rule: Applied rule
            rule_type: Type of rule
            items_affected: Number of items affected
        """
        self.explanations.append({
            'rule_id': rule.id,
            'rule_name': rule.name,
            'rule_type': rule_type,
            'items_affected': items_affected,
            'priority': rule.priority,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about rule executions.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_history:
            return {'total_executions': 0}
        
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['status'] == 'success')
        partial = sum(1 for e in self.execution_history if e['status'] == 'partial')
        failed = sum(1 for e in self.execution_history if e['status'] == 'failed')
        
        avg_time = sum(e['execution_time_ms'] for e in self.execution_history) / total
        avg_input = sum(e['input_count'] for e in self.execution_history) / total
        avg_output = sum(e['output_count'] for e in self.execution_history) / total
        
        return {
            'total_executions': total,
            'successful': successful,
            'partial': partial,
            'failed': failed,
            'success_rate': successful / total,
            'average_execution_time_ms': avg_time,
            'average_input_count': avg_input,
            'average_output_count': avg_output,
            'recent_executions': self.execution_history[-10:]
        }
    
    def reset(self):
        """Reset all rule state."""
        self.filter_rules.reset()
        self.boost_rules.reset()
        self.conditional_rules.reset()
        self.context_rules.reset()
        self.campaign_rules.reset()
        self.execution_history.clear()
        self.explanations.clear()
        logger.info("Rule executor state reset")
