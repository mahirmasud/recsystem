"""
Standardized Data Layer - Module 7

Production-grade standardized data transformation layer that converts ANY structured 
relational dataset into canonical ML-ready recommendation datasets using schema 
intelligence from rec_config.json.

This layer:
- Loads dynamic schema definitions from rec_config.json
- Interprets semantic mappings dynamically
- Applies canonical transformations
- Standardizes relational structures for any domain
- Creates ML-ready datasets (users, items, interactions, transactions, events, etc.)
- Generates derived metrics dynamically
- Maintains complete lineage tracking
- Preserves schema reproducibility
- Supports incremental synchronization
- Exports reusable parquet datasets

Supported domains:
- ecommerce, fintech, healthcare, edtech, SaaS, CRM, ERP
- analytics systems, recommendation systems, behavioral systems
- any structured relational dataset
"""

from .canonical_schema import CanonicalSchema, ColumnDefinition, EntitySchema
from .config_reader import ConfigReader
from .mapping_applier import MappingApplier
from .dataframe_standardizer import DataFrameStandardizer
from .metric_builder import MetricBuilder
from .lineage_manager import LineageManager
from .null_handler import NullHandler
from .schema_validator import SchemaValidator
from .dataset_writer import DatasetWriter
from .sync_manager import SyncManager
from .relationship_builder import RelationshipBuilder, Relationship
from .entity_resolver import EntityResolver, EntityMatch
from .semantic_transformer import SemanticTransformer, SemanticRule
from .role_mapper import RoleMapper, RoleAssignment
from .feature_config_loader import FeatureConfigLoader, FeatureConfig, InteractionWeight
from .schema_registry import SchemaRegistry, SchemaVersion, SchemaDefinition
from .transformation_registry import TransformationRegistry, TransformationRule, TransformationRecord
from .metadata_tracker import MetadataTracker, EntityMetadata, ColumnMetadata, ProcessingMetadata
from .incremental_processor import IncrementalProcessor, IncrementalState
from .run import StandardizationPipeline

__version__ = '1.0.0'

__all__ = [
    # Core components
    'CanonicalSchema',
    'ColumnDefinition', 
    'EntitySchema',
    'ConfigReader',
    'MappingApplier',
    'DataFrameStandardizer',
    
    # Transformation engines
    'MetricBuilder',
    'SemanticTransformer',
    'SemanticRule',
    'RoleMapper',
    'RoleAssignment',
    
    # Relationship & Identity
    'RelationshipBuilder',
    'Relationship',
    'EntityResolver',
    'EntityMatch',
    
    # Configuration & Features
    'FeatureConfigLoader',
    'FeatureConfig',
    'InteractionWeight',
    
    # Validation & Quality
    'SchemaValidator',
    'NullHandler',
    
    # Registry & Metadata
    'SchemaRegistry',
    'SchemaVersion',
    'SchemaDefinition',
    'TransformationRegistry',
    'TransformationRule',
    'TransformationRecord',
    'MetadataTracker',
    'EntityMetadata',
    'ColumnMetadata',
    'ProcessingMetadata',
    
    # Lineage & Tracking
    'LineageManager',
    'SyncManager',
    
    # Incremental Processing
    'IncrementalProcessor',
    'IncrementalState',
    
    # I/O
    'DatasetWriter',
    
    # Pipeline
    'StandardizationPipeline',
]
