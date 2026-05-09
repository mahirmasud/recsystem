"""
Standardized Data Layer - Module 7

Converts mapped client data into canonical recommendation-ready datasets.

This layer:
- Applies confirmed mappings from rec_config.json
- Renames columns to canonical names
- Creates canonical tables (users, items, transactions, interactions, categories)
- Handles missing optional fields
- Creates derived metrics
- Tracks lineage metadata
- Exports parquet datasets
"""

from .canonical_schema import CanonicalSchema
from .config_reader import ConfigReader
from .mapping_applier import MappingApplier
from .dataframe_standardizer import DataFrameStandardizer
from .metric_builder import MetricBuilder
from .lineage_manager import LineageManager
from .null_handler import NullHandler
from .schema_validator import SchemaValidator
from .dataset_writer import DatasetWriter
from .sync_manager import SyncManager
from .run import StandardizationPipeline

__all__ = [
    'CanonicalSchema',
    'ConfigReader',
    'MappingApplier',
    'DataFrameStandardizer',
    'MetricBuilder',
    'LineageManager',
    'NullHandler',
    'SchemaValidator',
    'DatasetWriter',
    'SyncManager',
    'StandardizationPipeline'
]
