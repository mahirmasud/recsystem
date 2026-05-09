"""
Feature Engineering Pipeline Runner - Module 9 Entry Point

Usage:
    python -m Feature_Engineering.run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
import logging

from shared.config import ConfigLoader
from shared.logger import setup_logging
from .feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)


def run_feature_engineering(config_path: str = "output/rec_config.json",
                            transactions_path: str = "output/standardized/transactions.parquet",
                            interactions_path: str = "output/standardized/interactions.parquet",
                            output_dir: str = "output/features") -> None:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file
        transactions_path: Path to standardized transactions
        interactions_path: Path to standardized interactions
        output_dir: Output directory for features
    """
    # Setup logging
    setup_logging(log_dir="output/logs", log_file="feature_engineering.log")
    
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config = config_loader.get_config()
    
    # Load standardized data
    logger.info(f"Loading transactions from {transactions_path}")
    if os.path.exists(transactions_path):
        transactions_df = pd.read_parquet(transactions_path)
        logger.info(f"Loaded {len(transactions_df)} transactions")
    else:
        logger.warning(f"Transactions file not found: {transactions_path}")
        transactions_df = pd.DataFrame()
    
    # Load interactions if available
    interactions_df = None
    if os.path.exists(interactions_path):
        logger.info(f"Loading interactions from {interactions_path}")
        interactions_df = pd.read_parquet(interactions_path)
        logger.info(f"Loaded {len(interactions_df)} interactions")
    
    # Initialize feature generator
    feature_gen = FeatureGenerator(config, output_dir=output_dir)
    
    # Generate all features
    reference_date = datetime.now()
    features = feature_gen.generate_all_features(
        transactions_df=transactions_df,
        interactions_df=interactions_df,
        reference_date=reference_date
    )
    
    # Log feature summary
    for entity_type, df in features.items():
        if df is not None:
            logger.info(f"{entity_type} features: {df.shape[0]} records, {df.shape[1]} columns")
    
    # Get and log metadata
    metadata = feature_gen.get_feature_metadata()
    logger.info(f"Feature metadata saved: {metadata}")
    
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
    parser.add_argument("--config", default="output/rec_config.json", help="Config file path")
    parser.add_argument("--transactions", default="output/standardized/transactions.parquet", 
                        help="Transactions file path")
    parser.add_argument("--interactions", default="output/standardized/interactions.parquet",
                        help="Interactions file path")
    parser.add_argument("--output", default="output/features", help="Output directory")
    
    args = parser.parse_args()
    
    run_feature_engineering(
        config_path=args.config,
        transactions_path=args.transactions,
        interactions_path=args.interactions,
        output_dir=args.output
    )
