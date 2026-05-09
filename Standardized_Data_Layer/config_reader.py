"""
Configuration reader for the standardization layer.
Reads and provides access to rec_config.json settings.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from shared.config import Config


logger = get_logger(__name__)


class ConfigReader:
    """
    Reads and provides structured access to rec_config.json.
    
    This class wraps the shared Config class and provides
    additional methods specific to the standardization layer.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration reader.
        
        Args:
            config_path: Path to rec_config.json file
        """
        self.config = Config(config_path)
        logger.info(f"Loaded configuration from {self.config.config_path}")
    
    def get_trusted_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all trusted field mappings.
        
        Returns:
            Dictionary of entity type to column mappings
        """
        return self.config.get_trusted_mappings()
    
    def get_entity_mapping(self, entity_type: str) -> Dict[str, Any]:
        """
        Get mapping for a specific entity type.
        
        Args:
            entity_type: Entity type name (users, items, etc.)
        
        Returns:
            Column mapping dictionary
        """
        return self.config.get_trusted_mappings(entity_type)
    
    def get_column_mapping(self, entity_type: str) -> Dict[str, str]:
        """
        Get source-to-canonical column mapping for an entity.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Dictionary mapping source columns to canonical names
        """
        mappings = self.get_entity_mapping(entity_type)
        return {
            details['source_column']: canonical_name
            for canonical_name, details in mappings.items()
        }
    
    def get_reverse_mapping(self, entity_type: str) -> Dict[str, str]:
        """
        Get canonical-to-source column mapping for an entity.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Dictionary mapping canonical names to source columns
        """
        mappings = self.get_entity_mapping(entity_type)
        return {
            canonical_name: details['source_column']
            for canonical_name, details in mappings.items()
        }
    
    def get_dtype_mapping(self, entity_type: str) -> Dict[str, str]:
        """
        Get data type mapping for an entity.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Dictionary mapping canonical columns to dtypes
        """
        mappings = self.get_entity_mapping(entity_type)
        return {
            canonical_name: details['dtype']
            for canonical_name, details in mappings.items()
        }
    
    def get_source_table(self, entity_type: str, canonical_field: str) -> Optional[str]:
        """
        Get source table for a canonical field.
        
        Args:
            entity_type: Entity type
            canonical_field: Canonical field name
        
        Returns:
            Source table name or None
        """
        mapping = self.config.get_mapping(entity_type, canonical_field)
        if mapping:
            return mapping.get('source_table')
        return None
    
    def get_business_meanings(self) -> Dict[str, Dict[str, str]]:
        """
        Get all business meaning definitions.
        
        Returns:
            Dictionary of metric name to definition
        """
        return self.config.get_business_meanings()
    
    def get_metric_definitions(self) -> Dict[str, Dict[str, str]]:
        """
        Get all metric definitions.
        
        Returns:
            Dictionary of metric name to aggregation info
        """
        return self.config.get_metric_definitions()
    
    def get_foreign_keys(self) -> Dict[str, str]:
        """
        Get foreign key relationships.
        
        Returns:
            Dictionary of FK relationships
        """
        return self.config.get_foreign_keys()
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields.
        
        Returns:
            List of required field names
        """
        return self.config.get_required_fields()
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules configuration.
        
        Returns:
            Validation rules dictionary
        """
        return self.config.get_validation_rules()
    
    def get_canonical_roles(self) -> Dict[str, Any]:
        """
        Get canonical role assignments.
        
        Returns:
            Canonical roles dictionary
        """
        return self.config.get_canonical_roles()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.to_dict()
