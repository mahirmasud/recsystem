"""
Schema registry for managing and versioning schema definitions.
Provides schema evolution tracking and compatibility checking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


@dataclass
class SchemaVersion:
    """Represents a version of a schema."""
    version: str
    created_at: str
    columns: Dict[str, str]  # column_name -> dtype
    primary_key: Optional[str] = None
    description: str = ""
    is_compatible_with: List[str] = field(default_factory=list)


@dataclass
class SchemaDefinition:
    """Complete schema definition with version history."""
    entity_type: str
    current_version: str
    versions: Dict[str, SchemaVersion] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaRegistry:
    """
    Registry for managing schema definitions and versions.
    
    Features:
    - Schema versioning
    - Compatibility checking
    - Schema evolution tracking
    - Multi-domain support
    - Schema discovery
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize schema registry.
        
        Args:
            registry_path: Path to store registry files
        """
        self.registry_path = Path(registry_path) if registry_path else Constants.OUTPUT_DIR / 'schema_registry'
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.schemas: Dict[str, SchemaDefinition] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        registry_file = self.registry_path / 'registry.json'
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for entity_type, schema_data in data.items():
                    versions = {}
                    for ver, ver_data in schema_data.get('versions', {}).items():
                        versions[ver] = SchemaVersion(**ver_data)
                    
                    self.schemas[entity_type] = SchemaDefinition(
                        entity_type=entity_type,
                        current_version=schema_data.get('current_version', '1.0'),
                        versions=versions,
                        metadata=schema_data.get('metadata', {})
                    )
                
                logger.info(f"Loaded {len(self.schemas)} schemas from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
    
    def register_schema(
        self,
        entity_type: str,
        columns: Dict[str, str],
        primary_key: Optional[str] = None,
        version: str = '1.0',
        description: str = '',
        metadata: Optional[Dict[str, Any]] = None
    ) -> SchemaDefinition:
        """
        Register a new schema or update existing one.
        
        Args:
            entity_type: Entity type name
            columns: Column definitions (name -> dtype)
            primary_key: Primary key column
            version: Schema version
            description: Schema description
            metadata: Additional metadata
        
        Returns:
            Registered schema definition
        """
        logger.info(f"Registering schema for {entity_type} v{version}")
        
        if entity_type not in self.schemas:
            # New schema
            schema_def = SchemaDefinition(
                entity_type=entity_type,
                current_version=version,
                metadata=metadata or {}
            )
        else:
            schema_def = self.schemas[entity_type]
            schema_def.metadata.update(metadata or {})
        
        # Create version
        schema_version = SchemaVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            columns=columns,
            primary_key=primary_key,
            description=description
        )
        
        schema_def.versions[version] = schema_version
        schema_def.current_version = version
        
        self.schemas[entity_type] = schema_def
        self._save_registry()
        
        return schema_def
    
    def get_schema(
        self,
        entity_type: str,
        version: Optional[str] = None
    ) -> Optional[SchemaVersion]:
        """
        Get schema for an entity type.
        
        Args:
            entity_type: Entity type name
            version: Specific version (default: current)
        
        Returns:
            Schema version or None
        """
        if entity_type not in self.schemas:
            return None
        
        schema_def = self.schemas[entity_type]
        
        if version is None:
            version = schema_def.current_version
        
        return schema_def.versions.get(version)
    
    def get_columns(
        self,
        entity_type: str,
        version: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get column definitions for an entity.
        
        Args:
            entity_type: Entity type
            version: Schema version
        
        Returns:
            Column dictionary
        """
        schema = self.get_schema(entity_type, version)
        return schema.columns if schema else {}
    
    def check_compatibility(
        self,
        entity_type: str,
        new_columns: Dict[str, str],
        new_version: str
    ) -> Dict[str, Any]:
        """
        Check if a new schema version is compatible.
        
        Args:
            entity_type: Entity type
            new_columns: New column definitions
            new_version: New version string
        
        Returns:
            Compatibility report
        """
        current_schema = self.get_schema(entity_type)
        
        if not current_schema:
            return {
                'compatible': True,
                'reason': 'No existing schema, any columns are valid',
                'breaking_changes': [],
                'additive_changes': list(new_columns.keys())
            }
        
        current_columns = current_schema.columns
        breaking_changes = []
        additive_changes = []
        removed_columns = []
        
        # Check for type changes (breaking)
        for col, dtype in new_columns.items():
            if col in current_columns:
                if current_columns[col] != dtype:
                    breaking_changes.append({
                        'column': col,
                        'change': 'type_change',
                        'old_type': current_columns[col],
                        'new_type': dtype
                    })
            else:
                additive_changes.append(col)
        
        # Check for removed columns
        for col in current_columns:
            if col not in new_columns:
                removed_columns.append(col)
        
        # Removed required columns are breaking
        for col in removed_columns:
            breaking_changes.append({
                'column': col,
                'change': 'column_removed'
            })
        
        return {
            'compatible': len(breaking_changes) == 0,
            'breaking_changes': breaking_changes,
            'additive_changes': additive_changes,
            'removed_columns': removed_columns,
            'can_evolve': len(breaking_changes) == 0
        }
    
    def get_all_entity_types(self) -> List[str]:
        """Get all registered entity types."""
        return list(self.schemas.keys())
    
    def get_schema_history(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get version history for a schema.
        
        Args:
            entity_type: Entity type
        
        Returns:
            List of version records
        """
        if entity_type not in self.schemas:
            return []
        
        schema_def = self.schemas[entity_type]
        return [
            {
                'version': ver.version,
                'created_at': ver.created_at,
                'column_count': len(ver.columns),
                'description': ver.description
            }
            for ver in schema_def.versions.values()
        ]
    
    def validate_against_schema(
        self,
        entity_type: str,
        columns: Dict[str, str],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate columns against registered schema.
        
        Args:
            entity_type: Entity type
            columns: Columns to validate
            version: Schema version to validate against
        
        Returns:
            Validation report
        """
        schema = self.get_schema(entity_type, version)
        
        if not schema:
            return {
                'valid': False,
                'reason': f'Schema not found for {entity_type}'
            }
        
        missing_columns = []
        extra_columns = []
        type_mismatches = []
        
        for col, expected_dtype in schema.columns.items():
            if col not in columns:
                missing_columns.append(col)
            elif columns[col] != expected_dtype:
                type_mismatches.append({
                    'column': col,
                    'expected': expected_dtype,
                    'actual': columns[col]
                })
        
        for col in columns:
            if col not in schema.columns:
                extra_columns.append(col)
        
        return {
            'valid': len(missing_columns) == 0 and len(type_mismatches) == 0,
            'missing_columns': missing_columns,
            'extra_columns': extra_columns,
            'type_mismatches': type_mismatches,
            'schema_version': schema.version
        }
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / 'registry.json'
        
        data = {}
        for entity_type, schema_def in self.schemas.items():
            data[entity_type] = {
                'current_version': schema_def.current_version,
                'versions': {
                    ver: sv.__dict__ for ver, sv in schema_def.versions.items()
                },
                'metadata': schema_def.metadata
            }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved schema registry to {registry_file}")
    
    def export_schema_metadata(self, output_path: Optional[str] = None) -> Path:
        """
        Export complete schema metadata.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path) if output_path else self.registry_path / 'schema_metadata.json'
        
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'total_schemas': len(self.schemas),
            'schemas': {}
        }
        
        for entity_type, schema_def in self.schemas.items():
            metadata['schemas'][entity_type] = {
                'current_version': schema_def.current_version,
                'version_count': len(schema_def.versions),
                'columns': schema_def.versions[schema_def.current_version].columns if schema_def.versions else {},
                'primary_key': schema_def.versions[schema_def.current_version].primary_key if schema_def.versions else None,
                'history': self.get_schema_history(entity_type)
            }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported schema metadata to {output_path}")
        return output_path
    
    def clear(self) -> None:
        """Clear all registered schemas."""
        self.schemas = {}
        self._save_registry()
