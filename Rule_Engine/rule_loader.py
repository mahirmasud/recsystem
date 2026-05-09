"""
Rule Loader - Loads YAML rule configurations.

Responsible for:
- Loading rule configuration from YAML files
- Validating basic structure
- Providing raw rule data to parser
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from shared.logger import get_logger
from shared.exceptions import RuleEngineError

logger = get_logger(__name__)


class RuleLoader:
    """Loads and validates rule configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RuleLoader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.raw_rules: Dict[str, Any] = {}
        
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load rules from YAML file.
        
        Args:
            config_path: Optional path override
            
        Returns:
            Dictionary of raw rule configurations
            
        Raises:
            RuleEngineError: If file not found or invalid YAML
        """
        path = config_path or self.config_path
        
        if not path:
            logger.warning("No config path provided, returning empty rules")
            return {'rules': []}
            
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise RuleEngineError(f"Rule config file not found: {path}")
            
        try:
            with open(path_obj, 'r') as f:
                self.raw_rules = yaml.safe_load(f)
                
            if not self.raw_rules:
                logger.warning(f"Empty rule config: {path}")
                self.raw_rules = {'rules': []}
                
            logger.info(f"Loaded rule config from {path}")
            logger.info(f"Found {len(self.raw_rules.get('rules', []))} rules")
            
            return self.raw_rules
            
        except yaml.YAMLError as e:
            raise RuleEngineError(f"Invalid YAML in rule config: {e}")
        except Exception as e:
            raise RuleEngineError(f"Failed to load rule config: {e}")
    
    def load_from_dict(self, rules_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load rules directly from dictionary.
        
        Args:
            rules_dict: Dictionary containing rule configuration
            
        Returns:
            The same dictionary
        """
        self.raw_rules = rules_dict
        logger.info(f"Loaded {len(rules_dict.get('rules', []))} rules from dict")
        return self.raw_rules
    
    def get_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """
        Get all rules of a specific type.
        
        Args:
            rule_type: Type of rules to filter (filter, boost, conditional, etc.)
            
        Returns:
            List of rule configurations matching the type
        """
        rules = self.raw_rules.get('rules', [])
        return [r for r in rules if r.get('type') == rule_type]
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """
        Get all active rules (not disabled).
        
        Returns:
            List of active rule configurations
        """
        rules = self.raw_rules.get('rules', [])
        return [r for r in rules if r.get('enabled', True) is not False]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific rule by ID.
        
        Args:
            rule_id: Unique identifier of the rule
            
        Returns:
            Rule configuration or None if not found
        """
        rules = self.raw_rules.get('rules', [])
        for rule in rules:
            if rule.get('id') == rule_id:
                return rule
        return None
    
    def validate_structure(self) -> bool:
        """
        Validate basic structure of loaded rules.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if not self.raw_rules:
            raise RuleEngineError("No rules loaded")
        
        if 'rules' not in self.raw_rules:
            raise RuleEngineError("Missing 'rules' key in configuration")
        
        rules = self.raw_rules['rules']
        if not isinstance(rules, list):
            raise RuleEngineError("'rules' must be a list")
        
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise RuleEngineError(f"Rule {i} must be a dictionary")
            
            if 'id' not in rule:
                raise RuleEngineError(f"Rule {i} missing required 'id' field")
            
            if 'type' not in rule:
                raise RuleEngineError(f"Rule {rule['id']} missing required 'type' field")
        
        logger.info("Rule structure validation passed")
        return True
