"""
Rule Parser - Parses raw YAML rules into internal rule objects.

Responsible for:
- Parsing raw rule dictionaries
- Validating rule parameters
- Creating typed rule objects
- Handling rule precedence and priorities
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from shared.logger import get_logger
from shared.exceptions import RuleEngineError

logger = get_logger(__name__)


class RuleType(Enum):
    """Enumeration of supported rule types."""
    FILTER = "filter"
    BOOST = "boost"
    CONDITIONAL = "conditional"
    CONTEXT = "context"
    CAMPAIGN = "campaign"
    BLOCK = "block"
    DIVERSITY = "diversity"
    FRESHNESS = "freshness"


class RuleAction(Enum):
    """Enumeration of rule actions."""
    EXCLUDE = "exclude"
    INCLUDE = "include"
    BOOST_SCORE = "boost_score"
    REDUCE_SCORE = "reduce_score"
    SET_SCORE = "set_score"
    REORDER = "reorder"
    INJECT = "inject"


@dataclass
class ParsedRule:
    """Represents a parsed rule with all parameters."""
    id: str
    type: RuleType
    name: str
    description: str
    enabled: bool
    priority: int
    action: RuleAction
    conditions: Dict[str, Any]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'priority': self.priority,
            'action': self.action.value if self.action else None,
            'conditions': self.conditions,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class RuleParser:
    """Parses raw rule configurations into typed rule objects."""
    
    def __init__(self):
        """Initialize RuleParser."""
        self.parsed_rules: List[ParsedRule] = []
        
    def parse_all(self, raw_rules: Dict[str, Any]) -> List[ParsedRule]:
        """
        Parse all rules from raw configuration.
        
        Args:
            raw_rules: Raw rule configuration dictionary
            
        Returns:
            List of parsed rule objects
        """
        rules_list = raw_rules.get('rules', [])
        self.parsed_rules = []
        
        for i, raw_rule in enumerate(rules_list):
            try:
                parsed = self.parse_single(raw_rule)
                if parsed and parsed.enabled:
                    self.parsed_rules.append(parsed)
            except Exception as e:
                logger.error(f"Failed to parse rule {i}: {e}")
                raise
        
        # Sort by priority (higher priority first)
        self.parsed_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Parsed {len(self.parsed_rules)} active rules")
        return self.parsed_rules
    
    def parse_single(self, raw_rule: Dict[str, Any]) -> Optional[ParsedRule]:
        """
        Parse a single rule from raw configuration.
        
        Args:
            raw_rule: Raw rule dictionary
            
        Returns:
            ParsedRule object or None if disabled
            
        Raises:
            RuleEngineError: If rule is invalid
        """
        # Extract basic fields
        rule_id = raw_rule.get('id')
        if not rule_id:
            raise RuleEngineError("Rule missing required 'id' field")
        
        rule_type_str = raw_rule.get('type')
        if not rule_type_str:
            raise RuleEngineError(f"Rule {rule_id} missing required 'type' field")
        
        try:
            rule_type = RuleType(rule_type_str.lower())
        except ValueError:
            logger.warning(f"Unknown rule type '{rule_type_str}' for rule {rule_id}, defaulting to FILTER")
            rule_type = RuleType.FILTER
        
        # Parse action
        action_str = raw_rule.get('action', 'boost_score')
        try:
            action = RuleAction(action_str.lower())
        except ValueError:
            action = RuleAction.BOOST_SCORE
        
        # Build rule object
        parsed = ParsedRule(
            id=rule_id,
            type=rule_type,
            name=raw_rule.get('name', rule_id),
            description=raw_rule.get('description', ''),
            enabled=raw_rule.get('enabled', True),
            priority=raw_rule.get('priority', 0),
            action=action,
            conditions=self._parse_conditions(raw_rule.get('conditions', {})),
            parameters=self._parse_parameters(raw_rule.get('parameters', {})),
            metadata=raw_rule.get('metadata', {})
        )
        
        logger.debug(f"Parsed rule: {parsed.id} ({parsed.type.value})")
        return parsed
    
    def _parse_conditions(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse rule conditions.
        
        Args:
            conditions: Raw conditions dictionary
            
        Returns:
            Parsed conditions dictionary
        """
        parsed = {}
        
        for key, value in conditions.items():
            # Handle special condition types
            if key == 'time_range':
                parsed[key] = self._parse_time_range(value)
            elif key == 'user_segments':
                parsed[key] = value if isinstance(value, list) else [value]
            elif key == 'item_attributes':
                parsed[key] = self._parse_item_attributes(value)
            elif key == 'expression':
                parsed[key] = value  # Keep as-is for evaluation later
            else:
                parsed[key] = value
        
        return parsed
    
    def _parse_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse rule parameters.
        
        Args:
            parameters: Raw parameters dictionary
            
        Returns:
            Parsed parameters dictionary
        """
        parsed = {}
        
        for key, value in parameters.items():
            if key in ['boost_factor', 'score_multiplier', 'weight']:
                # Ensure numeric
                try:
                    parsed[key] = float(value)
                except (ValueError, TypeError):
                    parsed[key] = 1.0
            elif key in ['max_items', 'limit', 'top_n']:
                # Ensure integer
                try:
                    parsed[key] = int(value)
                except (ValueError, TypeError):
                    parsed[key] = 10
            elif key == 'attributes':
                parsed[key] = value if isinstance(value, list) else [value]
            else:
                parsed[key] = value
        
        return parsed
    
    def _parse_time_range(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse time range conditions.
        
        Args:
            time_range: Raw time range dictionary
            
        Returns:
            Parsed time range with datetime objects
        """
        parsed = {}
        
        if 'start' in time_range:
            try:
                parsed['start'] = datetime.fromisoformat(time_range['start'])
            except (ValueError, TypeError):
                parsed['start'] = None
        
        if 'end' in time_range:
            try:
                parsed['end'] = datetime.fromisoformat(time_range['end'])
            except (ValueError, TypeError):
                parsed['end'] = None
        
        parsed['days_of_week'] = time_range.get('days_of_week', list(range(7)))
        parsed['hours'] = time_range.get('hours', list(range(24)))
        
        return parsed
    
    def _parse_item_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse item attribute conditions.
        
        Args:
            attributes: Raw attributes dictionary
            
        Returns:
            Parsed attributes dictionary
        """
        parsed = {}
        
        for attr, value in attributes.items():
            if isinstance(value, dict):
                # Handle operators like {'gte': 10, 'lte': 100}
                parsed[attr] = value
            elif isinstance(value, list):
                # Handle list of values (IN operator)
                parsed[attr] = {'in': value}
            else:
                # Handle single value (EQ operator)
                parsed[attr] = {'eq': value}
        
        return parsed
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[ParsedRule]:
        """
        Get parsed rules filtered by type.
        
        Args:
            rule_type: Type of rules to filter
            
        Returns:
            List of parsed rules matching the type
        """
        return [r for r in self.parsed_rules if r.type == rule_type]
    
    def get_rules_by_priority(self, min_priority: int = 0) -> List[ParsedRule]:
        """
        Get parsed rules with minimum priority.
        
        Args:
            min_priority: Minimum priority threshold
            
        Returns:
            List of parsed rules meeting priority threshold
        """
        return [r for r in self.parsed_rules if r.priority >= min_priority]
