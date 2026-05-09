"""
Chain Executor - Executes rules in chains with dependencies and conditions.

Responsible for:
- Managing rule execution chains
- Handling rule dependencies
- Conditional chain execution
- Short-circuit evaluation
- Chain-level statistics and logging
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from shared.logger import get_logger
from shared.exceptions import RuleEngineError
from .rule_parser import ParsedRule, RuleType
from .rule_executor import RuleExecutor, ExecutionResult, ExecutionStatus

logger = get_logger(__name__)


class ChainCondition(Enum):
    """Conditions for chain execution."""
    ALWAYS = "always"
    IF_PREVIOUS_SUCCESS = "if_previous_success"
    IF_PREVIOUS_FAILED = "if_previous_failed"
    IF_ITEMS_REMAIN = "if_items_remain"
    IF_NO_ITEMS = "if_no_items"
    CUSTOM = "custom"


@dataclass
class ChainStep:
    """Represents a single step in a rule chain."""
    step_id: str
    rule_ids: List[str]
    condition: ChainCondition
    custom_condition: Optional[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'rule_ids': self.rule_ids,
            'condition': self.condition.value,
            'custom_condition': self.custom_condition,
            'description': self.description
        }


@dataclass
class RuleChain:
    """Represents a complete rule execution chain."""
    chain_id: str
    name: str
    steps: List[ChainStep]
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain_id': self.chain_id,
            'name': self.name,
            'enabled': self.enabled,
            'priority': self.priority,
            'steps': [s.to_dict() for s in self.steps],
            'metadata': self.metadata
        }


class ChainExecutor:
    """Executes rules in configured chains with dependency handling."""
    
    def __init__(self):
        """Initialize ChainExecutor."""
        self.chains: Dict[str, RuleChain] = {}
        self.rules: Dict[str, ParsedRule] = {}
        self.executor = RuleExecutor()
        self.execution_history: List[Dict[str, Any]] = []
        
    def register_chain(self, chain: RuleChain):
        """
        Register a rule chain.
        
        Args:
            chain: RuleChain to register
        """
        self.chains[chain.chain_id] = chain
        logger.info(f"Registered rule chain: {chain.name} ({chain.chain_id})")
    
    def register_rules(self, rules: List[ParsedRule]):
        """
        Register rules for use in chains.
        
        Args:
            rules: List of parsed rules
        """
        for rule in rules:
            self.rules[rule.id] = rule
        logger.info(f"Registered {len(rules)} rules for chain execution")
    
    def execute_chain(self,
                     chain_id: str,
                     recommendations: List[Dict[str, Any]],
                     user_context: Dict[str, Any],
                     item_catalog: Dict[str, Dict[str, Any]]) -> ExecutionResult:
        """
        Execute a complete rule chain.
        
        Args:
            chain_id: ID of chain to execute
            recommendations: Input recommendations
            user_context: User context
            item_catalog: Item catalog
            
        Returns:
            ExecutionResult with final recommendations
        """
        if chain_id not in self.chains:
            raise RuleEngineError(f"Chain not found: {chain_id}")
        
        chain = self.chains[chain_id]
        
        if not chain.enabled:
            logger.warning(f"Chain '{chain.name}' is disabled, skipping")
            return ExecutionResult(
                status=ExecutionStatus.SKIPPED,
                recommendations=recommendations,
                rules_applied=[],
                rules_skipped=[r for step in chain.steps for r in step.rule_ids],
                rules_failed=[],
                explanations=[],
                execution_time_ms=0
            )
        
        start_time = datetime.now()
        current_recommendations = recommendations.copy()
        
        all_rules_applied = []
        all_rules_skipped = []
        all_rules_failed = []
        all_explanations = []
        
        previous_status = ExecutionStatus.SUCCESS
        
        # Execute each step in the chain
        for i, step in enumerate(chain.steps):
            logger.debug(f"Executing chain step {i+1}/{len(chain.steps)}: {step.step_id}")
            
            # Check if step should execute based on condition
            should_execute = self._check_step_condition(
                step,
                previous_status,
                current_recommendations
            )
            
            if not should_execute:
                logger.info(f"Skipping step '{step.step_id}' due to condition: {step.condition.value}")
                all_rules_skipped.extend(step.rule_ids)
                continue
            
            # Get rules for this step
            step_rules = [
                self.rules[rid] for rid in step.rule_ids
                if rid in self.rules
            ]
            
            if not step_rules:
                logger.warning(f"No rules found for step '{step.step_id}'")
                continue
            
            # Execute rules in this step
            result = self.executor.execute(
                current_recommendations,
                step_rules,
                user_context,
                item_catalog
            )
            
            current_recommendations = result.recommendations
            all_rules_applied.extend(result.rules_applied)
            all_rules_skipped.extend(result.rules_skipped)
            all_rules_failed.extend(result.rules_failed)
            all_explanations.extend(result.explanations)
            
            previous_status = result.status
            
            # Check for short-circuit conditions
            if not current_recommendations:
                logger.warning("No items remaining after step, stopping chain")
                break
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Determine overall status
        if all_rules_failed:
            status = ExecutionStatus.PARTIAL if current_recommendations else ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.SUCCESS
        
        # Record execution
        execution_record = {
            'chain_id': chain_id,
            'chain_name': chain.name,
            'timestamp': start_time.isoformat(),
            'status': status.value,
            'steps_executed': len(chain.steps),
            'rules_applied': len(all_rules_applied),
            'input_count': len(recommendations),
            'output_count': len(current_recommendations),
            'execution_time_ms': execution_time_ms
        }
        self.execution_history.append(execution_record)
        
        logger.info(
            f"Chain '{chain.name}' completed: {len(all_rules_applied)} rules applied, "
            f"{len(current_recommendations)} items output in {execution_time_ms:.2f}ms"
        )
        
        return ExecutionResult(
            status=status,
            recommendations=current_recommendations,
            rules_applied=all_rules_applied,
            rules_skipped=all_rules_skipped,
            rules_failed=all_rules_failed,
            explanations=all_explanations,
            execution_time_ms=execution_time_ms
        )
    
    def _check_step_condition(self,
                             step: ChainStep,
                             previous_status: ExecutionStatus,
                             current_recommendations: List[Dict[str, Any]]) -> bool:
        """
        Check if a step should execute based on its condition.
        
        Args:
            step: Chain step to check
            previous_status: Status of previous step
            current_recommendations: Current recommendation list
            
        Returns:
            True if step should execute
        """
        condition = step.condition
        
        if condition == ChainCondition.ALWAYS:
            return True
        
        elif condition == ChainCondition.IF_PREVIOUS_SUCCESS:
            return previous_status == ExecutionStatus.SUCCESS
        
        elif condition == ChainCondition.IF_PREVIOUS_FAILED:
            return previous_status in [ExecutionStatus.FAILED, ExecutionStatus.PARTIAL]
        
        elif condition == ChainCondition.IF_ITEMS_REMAIN:
            return len(current_recommendations) > 0
        
        elif condition == ChainCondition.IF_NO_ITEMS:
            return len(current_recommendations) == 0
        
        elif condition == ChainCondition.CUSTOM:
            # Evaluate custom condition expression
            if not step.custom_condition:
                return True
            
            try:
                eval_context = {
                    'items_remaining': len(current_recommendations),
                    'previous_status': previous_status.value,
                    'success': ExecutionStatus.SUCCESS.value,
                    'failed': ExecutionStatus.FAILED.value
                }
                result = eval(step.custom_condition, {"__builtins__": {}}, eval_context)
                return bool(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate custom condition: {e}")
                return True
        
        return True
    
    def execute_all_chains(self,
                          recommendations: List[Dict[str, Any]],
                          user_context: Dict[str, Any],
                          item_catalog: Dict[str, Dict[str, Any]],
                          order_by_priority: bool = True) -> Dict[str, ExecutionResult]:
        """
        Execute all registered chains.
        
        Args:
            recommendations: Input recommendations
            user_context: User context
            item_catalog: Item catalog
            order_by_priority: Whether to execute by priority order
            
        Returns:
            Dictionary mapping chain_id to ExecutionResult
        """
        chains = list(self.chains.values())
        
        if order_by_priority:
            chains.sort(key=lambda c: c.priority, reverse=True)
        
        results = {}
        current_recommendations = recommendations
        
        for chain in chains:
            result = self.execute_chain(
                chain.chain_id,
                current_recommendations,
                user_context,
                item_catalog
            )
            results[chain.chain_id] = result
            
            # Use output of one chain as input to next
            current_recommendations = result.recommendations
        
        return results
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about chain executions.
        
        Returns:
            Dictionary with chain statistics
        """
        if not self.execution_history:
            return {'total_executions': 0}
        
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['status'] == 'success')
        
        avg_time = sum(e['execution_time_ms'] for e in self.execution_history) / total
        
        # Group by chain
        by_chain = {}
        for e in self.execution_history:
            cid = e['chain_id']
            if cid not in by_chain:
                by_chain[cid] = {'count': 0, 'successful': 0}
            by_chain[cid]['count'] += 1
            if e['status'] == 'success':
                by_chain[cid]['successful'] += 1
        
        return {
            'total_executions': total,
            'successful': successful,
            'success_rate': successful / total,
            'average_execution_time_ms': avg_time,
            'chains_registered': len(self.chains),
            'by_chain': by_chain,
            'recent_executions': self.execution_history[-10:]
        }
    
    def reset(self):
        """Reset chain executor state."""
        self.executor.reset()
        self.execution_history.clear()
        logger.info("Chain executor state reset")
