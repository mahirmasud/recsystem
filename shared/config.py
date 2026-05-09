"""
Central configuration management for the recommendation system.
Handles loading and accessing configuration from rec_config.json.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager for the recommendation system.
    
    Provides centralized access to:
    - Trusted field mappings
    - Business meanings
    - Metric definitions
    - Canonical role assignments
    - Validation rules
    
    Thread-safe singleton pattern for consistent configuration access.
    """
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    _initialized: bool = False
    
    def __new__(cls, config_path: Optional[str] = None) -> 'Config':
        """Singleton pattern to ensure single configuration instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from rec_config.json.
        
        Args:
            config_path: Path to rec_config.json. Defaults to output/rec_config.json
        """
        if self._initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'output', 'rec_config.json'
        )
        self._load_config()
        Config._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                Config._config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}. "
                "Please ensure rec_config.json exists in output directory."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def reload(self) -> None:
        """Force reload configuration from disk."""
        self._load_config()
    
    @property
    def version(self) -> str:
        """Get configuration version."""
        return Config._config_data.get('version', '1.0.0')
    
    @property
    def domain(self) -> str:
        """Get domain type (e.g., ecommerce)."""
        return Config._config_data.get('domain', 'unknown')
    
    def get_trusted_mappings(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trusted field mappings.
        
        Args:
            entity_type: Specific entity type (users, items, transactions, interactions, categories)
                        If None, returns all mappings.
        
        Returns:
            Dictionary of column mappings
        """
        mappings = Config._config_data.get('trusted_mappings', {})
        if entity_type:
            return mappings.get(entity_type, {})
        return mappings
    
    def get_mapping(self, entity_type: str, canonical_field: str) -> Optional[Dict[str, Any]]:
        """
        Get specific field mapping.
        
        Args:
            entity_type: Entity type (users, items, etc.)
            canonical_field: Canonical field name
        
        Returns:
            Mapping details or None if not found
        """
        mappings = self.get_trusted_mappings(entity_type)
        return mappings.get(canonical_field)
    
    def get_business_meanings(self) -> Dict[str, Any]:
        """Get all business meaning definitions."""
        return Config._config_data.get('business_meanings', {})
    
    def get_metric_definitions(self) -> Dict[str, Any]:
        """Get all metric definitions."""
        return Config._config_data.get('metric_definitions', {})
    
    def get_canonical_roles(self) -> Dict[str, Any]:
        """Get canonical role assignments."""
        return Config._config_data.get('canonical_roles', {})
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules configuration."""
        return Config._config_data.get('validation_rules', {})
    
    def get_primary_keys(self) -> list:
        """Get list of primary key fields."""
        roles = self.get_canonical_roles()
        return roles.get('primary_key', [])
    
    def get_foreign_keys(self) -> Dict[str, str]:
        """Get foreign key relationships."""
        roles = self.get_canonical_roles()
        return roles.get('foreign_keys', {})
    
    def get_timestamp_fields(self) -> list:
        """Get list of timestamp fields."""
        roles = self.get_canonical_roles()
        return roles.get('timestamps', [])
    
    def get_required_fields(self) -> list:
        """Get list of required fields from validation rules."""
        rules = self.get_validation_rules()
        return rules.get('required_fields', [])
    
    def get_positive_value_fields(self) -> list:
        """Get list of fields that must have positive values."""
        rules = self.get_validation_rules()
        return rules.get('positive_values', [])
    
    def get_valid_ranges(self) -> Dict[str, list]:
        """Get valid value ranges for fields."""
        rules = self.get_validation_rules()
        return rules.get('valid_ranges', {})
    
    def get_entity_columns(self, entity_type: str) -> list:
        """
        Get list of canonical columns for an entity type.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            List of canonical column names
        """
        mappings = self.get_trusted_mappings(entity_type)
        return list(mappings.keys())
    
    def get_source_dtype(self, entity_type: str, canonical_field: str) -> str:
        """
        Get expected data type for a field.
        
        Args:
            entity_type: Entity type
            canonical_field: Canonical field name
        
        Returns:
            Data type string (e.g., 'int64', 'string', 'datetime64[ns]')
        """
        mapping = self.get_mapping(entity_type, canonical_field)
        if mapping:
            return mapping.get('dtype', 'object')
        return 'object'
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return Config._config_data.copy()
