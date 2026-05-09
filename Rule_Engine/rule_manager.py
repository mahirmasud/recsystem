"""Rule Manager - Centralized management of all business rules."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

class RuleManager:
    """Manages and applies business rules to recommendations."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = {'filter': [], 'boost': [], 'block': [], 'diversity': []}
        logger.info("RuleManager initialized")
        
    def add_rule(self, rule_type: str, rule_func, priority: int = 0):
        """Add a rule to the manager."""
        self.rules[rule_type].append({'func': rule_func, 'priority': priority})
        self.rules[rule_type].sort(key=lambda x: x['priority'], reverse=True)
        logger.info(f"Added {rule_type} rule with priority {priority}")
        
    def apply_all(self, candidates: List[Dict], user_context: Dict) -> List[Dict]:
        """Apply all rules in order: filter -> block -> boost -> diversity."""
        # Apply filters
        for rule in self.rules.get('filter', []):
            candidates = [c for c in candidates if rule['func'](c, user_context)]
        
        # Apply blocks
        blocked = set()
        for rule in self.rules.get('block', []):
            for c in candidates:
                if rule['func'](c, user_context):
                    blocked.add(c['item_id'])
        candidates = [c for c in candidates if c['item_id'] not in blocked]
        
        # Apply boosts
        for rule in self.rules.get('boost', []):
            for c in candidates:
                if rule['func'](c, user_context):
                    c['score'] *= 1.2
        
        # Apply diversity
        for rule in self.rules.get('diversity', []):
            candidates = rule['func'](candidates, user_context)
        
        return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
