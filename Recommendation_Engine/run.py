"""
Recommendation Engine Run Module

CLI entry point for recommendation operations:
- Training models
- Generating recommendations
- Evaluation
"""

import argparse
import json
import sys
from pathlib import Path

from shared.logger import setup_logger, get_logger
from .recommendation_manager import RecommendationManager

logger = get_logger(__name__)


def train_models(args):
    """Train recommendation models."""
    logger.info("Starting model training...")
    
    manager = RecommendationManager(
        config_path=args.config,
        models_dir=args.models_dir,
        features_dir=args.features_dir
    )
    
    # Load data
    manager.load_data(
        user_features_path=args.user_features,
        item_features_path=args.item_features,
        interactions_path=args.interactions
    )
    
    # Parse model types
    model_types = args.model_types.split(',') if args.model_types else None
    
    # Train models
    trained = manager.train_models(
        model_types=model_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Export report
    if args.output_report:
        manager.export_model_report(args.output_report)
    
    logger.info(f"Training complete. Trained models: {trained}")
    print(json.dumps(trained, indent=2))


def generate_recommendations(args):
    """Generate recommendations for users."""
    logger.info("Generating recommendations...")
    
    manager = RecommendationManager(
        config_path=args.config,
        models_dir=args.models_dir,
        features_dir=args.features_dir
    )
    
    # Load data
    manager.load_data()
    
    if args.user_id:
        # Single user
        recs = manager.generate_recommendations(
            user_id=args.user_id,
            top_k=args.top_k,
            config={'diversity_boost': args.diversity}
        )
        print(json.dumps(recs, indent=2))
        
    elif args.batch_file:
        # Batch recommendations
        import pandas as pd
        users_df = pd.read_csv(args.batch_file)
        user_ids = users_df['user_id'].tolist()
        
        recs_df = manager.batch_recommendations(
            user_ids=user_ids,
            top_k=args.top_k,
            output_path=args.output,
            config={'diversity_boost': args.diversity}
        )
        
        if args.output:
            logger.info(f"Saved recommendations to {args.output}")
        else:
            print(recs_df.to_string())
    
    else:
        logger.error("Must specify --user-id or --batch-file")
        sys.exit(1)


def evaluate_models(args):
    """Evaluate model performance."""
    logger.info("Evaluating models...")
    
    manager = RecommendationManager(
        config_path=args.config,
        models_dir=args.models_dir
    )
    
    # Load data
    manager.load_data()
    
    # Parse metrics
    metrics = args.metrics.split(',') if args.metrics else None
    
    # Evaluate
    results = manager.evaluate_models(
        test_interactions=None,  # Use internal split
        metrics=metrics
    )
    
    print(json.dumps(results, indent=2))


def list_models(args):
    """List available models."""
    logger.info("Listing models...")
    
    manager = RecommendationManager(
        models_dir=args.models_dir
    )
    
    models = manager.list_available_models()
    
    if args.active_only:
        models = [m for m in models if m['model_type'] in manager.registry.active_models.values()]
    
    print(json.dumps(models, indent=2))


def set_active_model(args):
    """Set active model for a type."""
    logger.info(f"Setting active {args.model_type} model to {args.model_id}")
    
    manager = RecommendationManager(
        models_dir=args.models_dir
    )
    
    manager.set_active_model(args.model_type, args.model_id)
    logger.info("Active model updated")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Recommendation Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train recommendation models')
    train_parser.add_argument('--config', default='output/rec_config.json')
    train_parser.add_argument('--models-dir', default='output/models')
    train_parser.add_argument('--features-dir', default='output/features')
    train_parser.add_argument('--user-features', default=None)
    train_parser.add_argument('--item-features', default=None)
    train_parser.add_argument('--interactions', default=None)
    train_parser.add_argument('--model-types', default='xgboost,lightgbm')
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--batch-size', type=int, default=256)
    train_parser.add_argument('--learning-rate', type=float, default=0.001)
    train_parser.add_argument('--output-report', default=None)
    train_parser.set_defaults(func=train_models)
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate recommendations')
    recommend_parser.add_argument('--config', default='output/rec_config.json')
    recommend_parser.add_argument('--models-dir', default='output/models')
    recommend_parser.add_argument('--features-dir', default='output/features')
    recommend_parser.add_argument('--user-id', default=None)
    recommend_parser.add_argument('--batch-file', default=None)
    recommend_parser.add_argument('--top-k', type=int, default=10)
    recommend_parser.add_argument('--diversity', action='store_true')
    recommend_parser.add_argument('--output', default=None)
    recommend_parser.set_defaults(func=generate_recommendations)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--config', default='output/rec_config.json')
    eval_parser.add_argument('--models-dir', default='output/models')
    eval_parser.add_argument('--metrics', default='precision,recall,ndcg,map')
    eval_parser.set_defaults(func=evaluate_models)
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.add_argument('--models-dir', default='output/models')
    list_parser.add_argument('--active-only', action='store_true')
    list_parser.set_defaults(func=list_models)
    
    # Set active model command
    set_parser = subparsers.add_parser('set-active', help='Set active model')
    set_parser.add_argument('--models-dir', default='output/models')
    set_parser.add_argument('--model-type', required=True)
    set_parser.add_argument('--model-id', required=True)
    set_parser.set_defaults(func=set_active_model)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logger(log_file='output/logs/recommendation.log')
    
    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
