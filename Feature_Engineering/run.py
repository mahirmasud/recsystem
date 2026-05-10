"""
Feature Engineering Pipeline Runner - Module 9 Entry Point

Production-grade automated feature engineering system using Featuretools.
Supports multi-domain schemas, relational datasets, temporal data, and dynamic mapping.

Usage:
    python -m Feature_Engineering.run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
import logging
import json
from typing import Dict, Any, Optional

from shared.config import ConfigLoader
from shared.logger import setup_logging
from shared.file_loader import FileLoader

# Import feature engineering components
from .entity_builder import EntityBuilder
from .dfs_pipeline import DFSPipeline
from .schema_analyzer import SchemaAnalyzer, RelationshipDetector
from .dynamic_feature_mapper import DynamicFeatureMapper
from .cutoff_manager import CutoffManager
from .feature_exporter import FeatureExporter
from .metadata_tracker import MetadataTracker
from .primitive_registry import PrimitiveRegistry
from .feature_store import FeatureStore
from .feature_versioning import FeatureVersioning

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Main orchestrator for the feature engineering pipeline."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "output/features"):
        self.config = config
        self.output_dir = output_dir
        
        # Initialize components
        self.entity_builder = EntityBuilder(config)
        self.dfs_pipeline = DFSPipeline(config)
        self.schema_analyzer = SchemaAnalyzer()
        self.dynamic_mapper = DynamicFeatureMapper(config)
        self.cutoff_manager = CutoffManager()
        self.feature_exporter = FeatureExporter(output_dir)
        self.metadata_tracker = MetadataTracker(output_dir)
        self.feature_store = FeatureStore(config, output_dir)
        self.feature_versioning = FeatureVersioning(output_dir)
        
        logger.info("FeatureEngineeringPipeline initialized")
    
    def run_full_pipeline(self,
                         dataframes: Dict[str, pd.DataFrame],
                         target_entity: Optional[str] = None,
                         use_dfs: bool = True,
                         max_depth: int = 2,
                         with_cutoffs: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run complete feature engineering pipeline.
        
        Args:
            dataframes: Dictionary of table_name -> DataFrame
            target_entity: Target entity for DFS (auto-detected if None)
            use_dfs: Whether to use Featuretools DFS
            max_depth: Maximum DFS depth
            with_cutoffs: Whether to use cutoff times
            
        Returns:
            Dictionary of generated features by entity type
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Analyze schemas
        logger.info("\n[Step 1] Analyzing schemas...")
        mappings = {}
        for table_name, df in dataframes.items():
            mapping = self.dynamic_mapper.analyze_and_map(df, table_name)
            mappings[table_name] = mapping
            
            # Track schema version
            self.metadata_tracker.track_schema_version(
                table_name=table_name,
                schema_info=mapping['schema_info']
            )
            
            domain = mapping.get('domain', 'generic')
            logger.info(f"  - {table_name}: domain={domain}, "
                       f"columns={mapping['schema_info']['column_count']}")
        
        # Step 2: Build EntitySet
        logger.info("\n[Step 2] Building EntitySet...")
        es = self.entity_builder.build_entityset(dataframes, name="main_entityset")
        es_info = self.entity_builder.export_entityset_info(es)
        logger.info(f"  Created EntitySet with {es_info['entity_count']} entities "
                   f"and {es_info['relationship_count']} relationships")
        
        # Step 3: Detect domain and configure primitives
        detected_domain = self.entity_builder.get_detected_domain()
        logger.info(f"\n[Step 3] Detected domain: {detected_domain}")
        self.dfs_pipeline.config['domain'] = detected_domain
        
        # Step 4: Run DFS or custom feature generation
        logger.info("\n[Step 4] Generating features...")
        
        if use_dfs and len(dataframes) > 0:
            # Determine target entity
            if not target_entity:
                target_entity = list(dataframes.keys())[0]
                logger.info(f"Auto-selected target entity: {target_entity}")
            
            # Prepare cutoff times if requested
            cutoff_time = None
            if with_cutoffs:
                target_df = dataframes.get(target_entity)
                if target_df is not None:
                    schema_info = self.schema_analyzer.analyze_dataframe(
                        target_df, target_entity
                    )
                    pks = schema_info.get('primary_keys', [])
                    timestamps = schema_info.get('timestamp_columns', [])
                    
                    if pks and timestamps:
                        cutoff_time = self.cutoff_manager.create_time_split_cutoffs(
                            df=target_df,
                            instance_col=pks[0],
                            time_col=timestamps[0],
                            split_date=datetime.now()
                        )
            
            # Run DFS
            feature_matrix, feature_defs = self.dfs_pipeline.run_dfs(
                entityset=es,
                target_dataframe_name=target_entity,
                max_depth=max_depth,
                cutoff_time=cutoff_time,
                verbose=True
            )
            
            results[target_entity] = feature_matrix
            
            # Export feature definitions
            self.feature_exporter.export_feature_definitions(
                feature_defs,
                os.path.join(self.output_dir, 'feature_registry.json')
            )
            
            # Track lineage
            self.metadata_tracker.track_feature_generation(
                entity_type=target_entity,
                features_df=feature_matrix,
                generation_method='dfs',
                source_tables=list(dataframes.keys()),
                parameters={'max_depth': max_depth, 'domain': detected_domain}
            )
            
            logger.info(f"  Generated {len(feature_matrix.columns)} DFS features "
                       f"for {len(feature_matrix)} records")
        
        # Step 5: Store and export features
        logger.info("\n[Step 5] Storing and exporting features...")
        
        for entity_type, features_df in results.items():
            self.feature_store.store_features(entity_type, features_df)
        
        # Export all artifacts
        exports = self.feature_exporter.export_all(
            features_dict=results,
            metadata=self.feature_store.get_feature_metadata(),
            feature_definitions={k: [] for k in results.keys()}  # Simplified
        )
        
        # Create version
        version_id = self.feature_versioning.create_version(results)
        logger.info(f"  Created feature version: {version_id}")
        
        # Export lineage report
        self.metadata_tracker.export_lineage_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("Feature Engineering Pipeline Complete")
        logger.info("=" * 60)
        
        return results


def run_feature_engineering(config_path: str = "output/rec_config.json",
                           data_dir: str = "output/validated",
                           output_dir: str = "output/features",
                           use_dfs: bool = True,
                           max_depth: int = 2) -> None:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing validated input data
        output_dir: Output directory for features
        use_dfs: Whether to use Featuretools DFS
        max_depth: Maximum DFS depth
    """
    # Setup logging
    setup_logging(log_dir="output/logs", log_file="feature_engineering.log")
    
    logger.info("Initializing Feature Engineering System")
    
    # Load configuration
    if os.path.exists(config_path):
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_config()
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = {}
    
    # Load input data from validated directory
    dataframes = {}
    file_loader = FileLoader()
    
    if os.path.exists(data_dir):
        logger.info(f"Loading data from {data_dir}")
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            try:
                if filename.endswith('.parquet'):
                    table_name = filename.replace('.parquet', '')
                    dataframes[table_name] = file_loader.load_parquet(filepath)
                    logger.info(f"  Loaded {table_name}: {len(dataframes[table_name])} rows")
                elif filename.endswith('.csv'):
                    table_name = filename.replace('.csv', '')
                    dataframes[table_name] = file_loader.load_csv(filepath)
                    logger.info(f"  Loaded {table_name}: {len(dataframes[table_name])} rows")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    else:
        logger.warning(f"Data directory not found: {data_dir}")
    
    # If no data loaded, create sample data for demonstration
    if not dataframes:
        logger.info("Creating sample data for demonstration...")
        dataframes = _create_sample_data()
    
    # Run pipeline
    pipeline = FeatureEngineeringPipeline(config, output_dir=output_dir)
    results = pipeline.run_full_pipeline(
        dataframes=dataframes,
        use_dfs=use_dfs,
        max_depth=max_depth
    )
    
    # Log summary
    for entity_type, df in results.items():
        logger.info(f"\n{entity_type} features:")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Features: {len(df.columns)}")
        logger.info(f"  Sample columns: {list(df.columns[:5])}")


def _create_sample_data() -> Dict[str, pd.DataFrame]:
    """Create sample data for demonstration."""
    import numpy as np
    
    n_users = 100
    n_items = 50
    n_transactions = 1000
    
    # Users table
    users_df = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'user_name': [f'user_{i}' for i in range(1, n_users + 1)],
        'signup_date': pd.date_range('2023-01-01', periods=n_users, freq='D'),
    })
    
    # Items table
    items_df = pd.DataFrame({
        'item_id': range(1, n_items + 1),
        'item_name': [f'item_{i}' for i in range(1, n_items + 1)],
        'category_id': np.random.choice(range(1, 6), n_items),
        'price': np.random.uniform(10, 100, n_items),
    })
    
    # Transactions table
    transactions_df = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'user_id': np.random.choice(range(1, n_users + 1), n_transactions),
        'item_id': np.random.choice(range(1, n_items + 1), n_transactions),
        'quantity': np.random.randint(1, 5, n_transactions),
        'net_sales': np.random.uniform(10, 500, n_transactions),
        'discount_amount': np.random.uniform(0, 50, n_transactions),
        'transaction_date': pd.date_range('2023-01-01', periods=n_transactions, freq='h'),
    })
    
    return {
        'users': users_df,
        'items': items_df,
        'transactions': transactions_df,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
    parser.add_argument("--config", default="output/rec_config.json", help="Config file path")
    parser.add_argument("--data-dir", default="output/validated", help="Input data directory")
    parser.add_argument("--output", default="output/features", help="Output directory")
    parser.add_argument("--no-dfs", action="store_true", help="Disable Featuretools DFS")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum DFS depth")
    
    args = parser.parse_args()
    
    run_feature_engineering(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output,
        use_dfs=not args.no_dfs,
        max_depth=args.max_depth
    )
