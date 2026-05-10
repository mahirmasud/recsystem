"""
Feature config loader for loading recommendation feature configurations.
Loads feature assignments, interaction weights, and content features from config.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for a single feature."""
    feature_name: str
    source_column: str
    feature_type: str  # numeric, categorical, embedding, temporal
    transformation: Optional[str] = None
    normalization: Optional[str] = None
    is_target: bool = False
    weight: float = 1.0


@dataclass
class InteractionWeight:
    """Weight configuration for an interaction type."""
    interaction_type: str
    weight: float
    decay_factor: Optional[float] = None
    time_window: Optional[int] = None


class FeatureConfigLoader:
    """
    Loads and manages feature configurations for recommendations.
    
    Features:
    - Load feature assignments from config
    - Load interaction signal weights
    - Load content feature configurations
    - Handle ignored fields
    - Support custom feature definitions
    """
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize feature config loader.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
        self.feature_configs: Dict[str, List[FeatureConfig]] = {}
        self.interaction_weights: List[InteractionWeight] = []
        self.ignored_fields: Dict[str, List[str]] = {}
        self.custom_features: Dict[str, Any] = {}
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all feature-related configurations.
        
        Returns:
            Dictionary of loaded configurations
        """
        logger.info("Loading all feature configurations")
        
        config_dict = self.config_reader.to_dict()
        
        # Load feature assignments per entity
        self._load_feature_assignments(config_dict)
        
        # Load interaction weights
        self._load_interaction_weights(config_dict)
        
        # Load ignored fields
        self._load_ignored_fields(config_dict)
        
        # Load custom features
        self._load_custom_features(config_dict)
        
        return {
            'feature_configs': self.feature_configs,
            'interaction_weights': [iw.__dict__ for iw in self.interaction_weights],
            'ignored_fields': self.ignored_fields,
            'custom_features': self.custom_features
        }
    
    def _load_feature_assignments(self, config_dict: Dict[str, Any]) -> None:
        """Load feature assignments from trusted mappings."""
        trusted_mappings = config_dict.get('trusted_mappings', {})
        
        for entity_type, columns in trusted_mappings.items():
            features = []
            
            for canonical_col, details in columns.items():
                dtype = details.get('dtype', 'string')
                
                # Infer feature type from dtype
                if 'int' in dtype or 'float' in dtype:
                    feature_type = 'numeric'
                elif 'datetime' in dtype:
                    feature_type = 'temporal'
                else:
                    feature_type = 'categorical'
                
                feature = FeatureConfig(
                    feature_name=canonical_col,
                    source_column=details.get('source_column', canonical_col),
                    feature_type=feature_type,
                    weight=1.0
                )
                features.append(feature)
            
            self.feature_configs[entity_type] = features
            logger.debug(f"Loaded {len(features)} feature configs for {entity_type}")
    
    def _load_interaction_weights(self, config_dict: Dict[str, Any]) -> None:
        """Load interaction signal weights from config."""
        # Check for explicit interaction weights section
        interaction_weights = config_dict.get('interaction_weights', {})
        
        if not interaction_weights:
            # Generate default weights based on common interaction types
            default_weights = {
                'purchase': 5.0,
                'order': 5.0,
                'add_to_cart': 3.0,
                'cart_add': 3.0,
                'click': 1.0,
                'view': 0.5,
                'impression': 0.3,
                'search': 2.0,
                'engagement': 1.5
            }
            
            for int_type, weight in default_weights.items():
                self.interaction_weights.append(InteractionWeight(
                    interaction_type=int_type,
                    weight=weight
                ))
        else:
            for int_type, weight_config in interaction_weights.items():
                if isinstance(weight_config, dict):
                    self.interaction_weights.append(InteractionWeight(
                        interaction_type=int_type,
                        weight=weight_config.get('weight', 1.0),
                        decay_factor=weight_config.get('decay_factor'),
                        time_window=weight_config.get('time_window')
                    ))
                else:
                    self.interaction_weights.append(InteractionWeight(
                        interaction_type=int_type,
                        weight=float(weight_config)
                    ))
        
        logger.debug(f"Loaded {len(self.interaction_weights)} interaction weights")
    
    def _load_ignored_fields(self, config_dict: Dict[str, Any]) -> None:
        """Load list of fields to ignore during processing."""
        validation_rules = config_dict.get('validation_rules', {})
        ignored = validation_rules.get('ignored_fields', {})
        
        if isinstance(ignored, dict):
            for entity_type, fields in ignored.items():
                self.ignored_fields[entity_type] = fields if isinstance(fields, list) else [fields]
        elif isinstance(ignored, list):
            self.ignored_fields['_global'] = ignored
        
        logger.debug(f"Loaded ignored fields for {len(self.ignored_fields)} entities")
    
    def _load_custom_features(self, config_dict: Dict[str, Any]) -> None:
        """Load custom feature definitions."""
        self.custom_features = config_dict.get('custom_features', {})
        
        if self.custom_features:
            logger.debug(f"Loaded {len(self.custom_features)} custom feature definitions")
    
    def get_features_for_entity(self, entity_type: str) -> List[FeatureConfig]:
        """
        Get feature configurations for an entity type.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            List of feature configurations
        """
        return self.feature_configs.get(entity_type, [])
    
    def get_interaction_weight(self, interaction_type: str) -> float:
        """
        Get weight for an interaction type.
        
        Args:
            interaction_type: Interaction type name
        
        Returns:
            Weight value (default 1.0 if not found)
        """
        interaction_type_lower = interaction_type.lower()
        
        for iw in self.interaction_weights:
            if iw.interaction_type.lower() == interaction_type_lower:
                return iw.weight
        
        return 1.0  # Default weight
    
    def get_all_interaction_weights(self) -> Dict[str, float]:
        """Get all interaction weights as dictionary."""
        return {iw.interaction_type: iw.weight for iw in self.interaction_weights}
    
    def is_field_ignored(self, entity_type: str, field_name: str) -> bool:
        """
        Check if a field should be ignored.
        
        Args:
            entity_type: Entity type
            field_name: Field name
        
        Returns:
            True if field should be ignored
        """
        global_ignored = self.ignored_fields.get('_global', [])
        entity_ignored = self.ignored_fields.get(entity_type, [])
        
        return field_name in global_ignored or field_name in entity_ignored
    
    def get_feature_types_for_entity(
        self, 
        entity_type: str,
        feature_type: Optional[str] = None
    ) -> List[FeatureConfig]:
        """
        Get features filtered by type.
        
        Args:
            entity_type: Entity type
            feature_type: Filter by feature type (numeric, categorical, etc.)
        
        Returns:
            Filtered list of feature configurations
        """
        features = self.get_features_for_entity(entity_type)
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        return features
    
    def build_feature_matrix_config(
        self,
        entity_type: str,
        include_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Build configuration for feature matrix generation.
        
        Args:
            entity_type: Entity type
            include_weights: Whether to include feature weights
        
        Returns:
            Feature matrix configuration
        """
        features = self.get_features_for_entity(entity_type)
        
        config = {
            'entity_type': entity_type,
            'features': [
                {
                    'name': f.feature_name,
                    'source': f.source_column,
                    'type': f.feature_type,
                    'transformation': f.transformation,
                    'normalization': f.normalization
                }
                for f in features
            ]
        }
        
        if include_weights:
            for i, f in enumerate(config['features']):
                f['weight'] = features[i].weight
        
        return config
    
    def add_custom_feature(
        self,
        entity_type: str,
        feature: FeatureConfig
    ) -> None:
        """Add a custom feature configuration."""
        if entity_type not in self.feature_configs:
            self.feature_configs[entity_type] = []
        
        self.feature_configs[entity_type].append(feature)
        logger.debug(f"Added custom feature '{feature.feature_name}' to {entity_type}")
    
    def clear_configs(self) -> None:
        """Clear all loaded configurations."""
        self.feature_configs = {}
        self.interaction_weights = []
        self.ignored_fields = {}
        self.custom_features = {}
