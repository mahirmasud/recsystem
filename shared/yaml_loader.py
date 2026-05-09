"""
YAML configuration loader for the recommendation system.
Supports loading rules and configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Optional, Union, Dict, Any
from shared.logger import get_logger


logger = get_logger(__name__)


class YAMLLoader:
    """
    YAML file loader with support for includes and validation.
    
    Features:
    - Load YAML files
    - Validate structure
    - Support for !include tags
    - Safe loading
    """
    
    @staticmethod
    def load(
        filepath: Union[str, Path],
        safe: bool = True
    ) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            filepath: Path to YAML file
            safe: Use safe loader (recommended)
        
        Returns:
            Parsed YAML content as dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"YAML file not found: {filepath}")
        
        logger.info(f"Loading YAML from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                if safe:
                    data = yaml.safe_load(f)
                else:
                    data = yaml.load(f, Loader=yaml.FullLoader)
            
            logger.info(f"Loaded YAML from {filepath}")
            return data or {}
        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filepath}: {e}")
    
    @staticmethod
    def save(
        data: Dict[str, Any],
        filepath: Union[str, Path],
        indent: int = 2
    ) -> None:
        """
        Save data to YAML file.
        
        Args:
            data: Data to save
            filepath: Output path
            indent: YAML indentation level
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving YAML to {filepath}")
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, indent=indent, default_flow_style=False)
        
        logger.info(f"Saved YAML to {filepath}")
    
    @staticmethod
    def validate_structure(
        data: Dict[str, Any],
        required_keys: list,
        optional_keys: Optional[list] = None
    ) -> bool:
        """
        Validate YAML structure has required keys.
        
        Args:
            data: Parsed YAML data
            required_keys: List of required top-level keys
            optional_keys: List of optional keys
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If required keys are missing
        """
        missing = [key for key in required_keys if key not in data]
        
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")
        
        logger.debug("YAML structure validation passed")
        return True
    
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
        
        Returns:
            Merged configuration
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = YAMLLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def load_rule_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate rule configuration file.
        
        Args:
            filepath: Path to rules YAML file
        
        Returns:
            Validated rule configuration
        """
        data = YAMLLoader.load(filepath)
        
        # Validate rule config structure
        YAMLLoader.validate_structure(
            data,
            required_keys=['rules'],
            optional_keys=['metadata', 'version']
        )
        
        logger.info(f"Loaded rule configuration with {len(data['rules'])} rules")
        return data
