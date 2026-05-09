"""
Conditional Rules - Applies rules based on complex logical conditions.

Responsible for:
- IF-THEN rule evaluation
- Multi-condition logic (AND, OR, NOT)
- Nested condition evaluation
- User segment-based conditions
- Dynamic rule activation
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from shared.logger import get_logger
from .rule_parser import ParsedRule, RuleAction

logger = get_logger(__name__)


class ConditionalRules:
    """Applies conditional rules with complex logic evaluation."""
    
    def __init__(self):
        """Initialize ConditionalRules."""
        self.evaluation_history: List[Dict[str, Any]] = []
        
    def apply(self,
              recommendations: List[Dict[str, Any]],
              rule: ParsedRule,
              user_context: Dict[str, Any],
              item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply conditional rule to recommendations.
        
        Args:
            recommendations: List of recommendation items
            rule: Parsed conditional rule
            user_context: User context dictionary
            item_catalog: Item metadata lookup
            
        Returns:
            List of recommendations with conditional adjustments
        """
        if not recommendations:
            return []
        
        conditions = rule.conditions
        parameters = rule.parameters
        
        # Evaluate the main condition expression
        condition_met = self._evaluate_condition(
            conditions.get('expression'),
            user_context,
            item_catalog,
            recommendations
        )
        
        if not condition_met:
            logger.debug(f"Conditional rule '{rule.id}' condition not met, skipping")
            return recommendations
        
        # Condition met, apply the action
        action = rule.action
        logger.info(f"Conditional rule '{rule.id}' condition met, applying action: {action.value}")
        
        # Store evaluation result
        self.evaluation_history.append({
            'rule_id': rule.id,
            'condition_met': True,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_context.get('user_id')
        })
        
        # Apply action based on type
        if action == RuleAction.BOOST_SCORE:
            return self._apply_conditional_boost(recommendations, parameters, item_catalog)
        elif action == RuleAction.SET_SCORE:
            return self._apply_conditional_set_score(recommendations, parameters, item_catalog)
        elif action == RuleAction.EXCLUDE:
            return self._apply_conditional_exclude(recommendations, parameters, item_catalog)
        elif action == RuleAction.REORDER:
            return self._apply_conditional_reorder(recommendations, parameters, item_catalog)
        else:
            return recommendations
    
    def _evaluate_condition(self,
                           expression: Union[str, Dict[str, Any]],
                           user_context: Dict[str, Any],
                           item_catalog: Dict[str, Dict[str, Any]],
                           recommendations: List[Dict[str, Any]]) -> bool:
        """
        Evaluate a condition expression.
        
        Args:
            expression: Condition expression (string or dict)
            user_context: User context
            item_catalog: Item catalog
            recommendations: Current recommendations
            
        Returns:
            True if condition is met
        """
        if not expression:
            return False
        
        if isinstance(expression, str):
            return self._evaluate_string_expression(expression, user_context)
        
        if isinstance(expression, dict):
            return self._evaluate_dict_expression(expression, user_context, item_catalog, recommendations)
        
        return False
    
    def _evaluate_string_expression(self, 
                                    expression: str, 
                                    user_context: Dict[str, Any]) -> bool:
        """
        Evaluate a string-based condition expression.
        
        Supports simple expressions like:
        - "user_segment == 'vip'"
        - "total_purchases > 1000"
        - "days_since_last_purchase < 30"
        
        Args:
            expression: String expression
            user_context: User context
            
        Returns:
            Evaluation result
        """
        try:
            # Create safe evaluation context
            eval_context = {
                'user_segment': user_context.get('segment', ''),
                'user_tier': user_context.get('tier', ''),
                'total_purchases': user_context.get('total_purchases', 0),
                'total_spent': user_context.get('total_spent', 0),
                'days_since_last_purchase': user_context.get('days_since_last_purchase', 999),
                'is_new_user': user_context.get('is_new_user', False),
                'is_active': user_context.get('is_active', True),
            }
            
            # Safe evaluation using eval with restricted context
            # Note: In production, consider using a proper expression parser
            result = eval(expression, {"__builtins__": {}}, eval_context)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate expression '{expression}': {e}")
            return False
    
    def _evaluate_dict_expression(self,
                                  expression: Dict[str, Any],
                                  user_context: Dict[str, Any],
                                  item_catalog: Dict[str, Dict[str, Any]],
                                  recommendations: List[Dict[str, Any]]) -> bool:
        """
        Evaluate a dictionary-based condition expression.
        
        Supports structured conditions with operators:
        {
            "and": [...],
            "or": [...],
            "not": {...}
        }
        
        Args:
            expression: Dictionary expression
            user_context: User context
            item_catalog: Item catalog
            recommendations: Recommendations
            
        Returns:
            Evaluation result
        """
        if 'and' in expression:
            conditions = expression['and']
            return all(
                self._evaluate_dict_expression(cond, user_context, item_catalog, recommendations)
                for cond in conditions
            )
        
        if 'or' in expression:
            conditions = expression['or']
            return any(
                self._evaluate_dict_expression(cond, user_context, item_catalog, recommendations)
                for cond in conditions
            )
        
        if 'not' in expression:
            return not self._evaluate_dict_expression(
                expression['not'], user_context, item_catalog, recommendations
            )
        
        # Simple condition: {field: {operator: value}}
        for field, condition in expression.items():
            if field.startswith('user.'):
                user_field = field.replace('user.', '')
                user_value = user_context.get(user_field)
                if not self._check_condition(user_value, condition):
                    return False
            elif field.startswith('recommendation.'):
                rec_field = field.replace('recommendation.', '')
                # Check if any recommendation matches
                has_match = any(
                    self._check_condition(rec.get(rec_field), condition)
                    for rec in recommendations
                )
                if not has_match:
                    return False
        
        return True
    
    def _check_condition(self, value: Any, condition: Any) -> bool:
        """
        Check if a value satisfies a condition.
        
        Args:
            value: Value to check
            condition: Condition (can be dict with operators or simple value)
            
        Returns:
            True if condition is satisfied
        """
        if isinstance(condition, dict):
            if 'eq' in condition:
                return value == condition['eq']
            if 'neq' in condition:
                return value != condition['neq']
            if 'gt' in condition:
                return value is not None and value > condition['gt']
            if 'gte' in condition:
                return value is not None and value >= condition['gte']
            if 'lt' in condition:
                return value is not None and value < condition['lt']
            if 'lte' in condition:
                return value is not None and value <= condition['lte']
            if 'in' in condition:
                return value in condition['in']
            if 'notin' in condition:
                return value not in condition['notin']
            if 'contains' in condition:
                return value is not None and condition['contains'] in str(value)
        else:
            return value == condition
        
        return False
    
    def _apply_conditional_boost(self,
                                 recommendations: List[Dict[str, Any]],
                                 parameters: Dict[str, Any],
                                 item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply conditional boost to matching items."""
        boost_factor = float(parameters.get('boost_factor', 1.5))
        target_attribute = parameters.get('target_attribute')
        target_value = parameters.get('target_value')
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            
            should_boost = False
            if target_attribute and target_value:
                item_attr = item_data.get(target_attribute)
                if item_attr == target_value:
                    should_boost = True
            else:
                should_boost = True
            
            if should_boost:
                current_score = rec.get('score', 0)
                rec['score'] = current_score * boost_factor
                rec['conditional_boost_applied'] = True
                rec['boost_reason'] = f"conditional:{target_attribute}={target_value}"
        
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        return recommendations
    
    def _apply_conditional_set_score(self,
                                     recommendations: List[Dict[str, Any]],
                                     parameters: Dict[str, Any],
                                     item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply conditional score setting."""
        new_score = float(parameters.get('score', 1.0))
        target_items = parameters.get('target_items', [])
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            if not target_items or item_id in target_items:
                rec['score'] = new_score
                rec['score_set_by_rule'] = True
        
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        return recommendations
    
    def _apply_conditional_exclude(self,
                                   recommendations: List[Dict[str, Any]],
                                   parameters: Dict[str, Any],
                                   item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply conditional exclusion."""
        exclude_attribute = parameters.get('exclude_attribute')
        exclude_value = parameters.get('exclude_value')
        
        if not exclude_attribute:
            return recommendations
        
        filtered = []
        for rec in recommendations:
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            item_attr = item_data.get(exclude_attribute)
            
            if item_attr != exclude_value:
                filtered.append(rec)
        
        return filtered
    
    def _apply_conditional_reorder(self,
                                   recommendations: List[Dict[str, Any]],
                                   parameters: Dict[str, Any],
                                   item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply conditional reordering."""
        priority_attribute = parameters.get('priority_attribute')
        priority_values = parameters.get('priority_values', [])
        
        if not priority_attribute:
            return recommendations
        
        # Group items by priority
        priority_groups = {v: [] for v in priority_values}
        priority_groups['other'] = []
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            item_attr = item_data.get(priority_attribute)
            
            if item_attr in priority_values:
                priority_groups[item_attr].append(rec)
            else:
                priority_groups['other'].append(rec)
        
        # Flatten in priority order
        result = []
        for priority in priority_values:
            result.extend(priority_groups[priority])
        result.extend(priority_groups['other'])
        
        return result
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about rule evaluations."""
        total_evaluations = len(self.evaluation_history)
        conditions_met = sum(1 for e in self.evaluation_history if e.get('condition_met'))
        
        return {
            'total_evaluations': total_evaluations,
            'conditions_met': conditions_met,
            'conditions_not_met': total_evaluations - conditions_met,
            'met_rate': conditions_met / total_evaluations if total_evaluations > 0 else 0
        }
    
    def reset(self):
        """Reset evaluation state."""
        self.evaluation_history.clear()
