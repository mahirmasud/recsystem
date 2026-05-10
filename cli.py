#!/usr/bin/env python3
"""
Intelligent Schema Mapping & Business Meaning System - CLI Orchestrator

This is the main command-line interface for the entire recommendation system.
It orchestrates all modules (7-12) without containing business logic.

Usage:
    python cli.py standardize
    python cli.py validate
    python cli.py generate-features
    python cli.py train-model
    python cli.py recommend --user 1001
    python cli.py apply-rules
    python cli.py batch-recommend users.csv
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared across commands."""
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='output/rec_config.json',
        help='Path to rec_config.json (default: output/rec_config.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )


def cmd_standardize(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 7: Standardized Data Layer.
    
    Converts mapped client data into canonical recommendation-ready datasets.
    """
    from Standardized_Data_Layer.run import StandardizationPipeline, create_sample_data
    from shared.constants import Constants
    
    try:
        # Ensure directories exist
        Constants.ensure_directories()
        
        pipeline = StandardizationPipeline(config_path=args.config)
        
        # Check if sample data requested
        if args.sample:
            # Generate and process sample data
            print("Generating sample data from configuration...")
            sample_data = create_sample_data(pipeline.config_reader)
            
            if not sample_data:
                print("ERROR: Could not generate sample data. Check your rec_config.json", file=sys.stderr)
                return 1
            
            print(f"Generated sample data for {len(sample_data)} entities:")
            for entity_type, df in sample_data.items():
                print(f"  - {entity_type}: {len(df)} rows, {len(df.columns)} columns")
            
            print("\nRunning standardization pipeline...")
            results = pipeline.run(
                sample_data,
                validate=not args.no_validate,
                write_output=not args.no_write
            )
            
        elif args.files:
            # Parse file arguments
            file_paths = {}
            for file_arg in args.files:
                if '=' in file_arg:
                    entity_type, filepath = file_arg.split('=', 1)
                    file_paths[entity_type] = filepath
                else:
                    print(f"ERROR: Invalid file argument format: {file_arg}. Use entity_type=file_path", file=sys.stderr)
                    return 1
            
            print(f"Running standardization from files: {file_paths}")
            results = pipeline.run_from_files(
                file_paths,
                validate=not args.no_validate,
                write_output=not args.no_write
            )
            
        else:
            # Interactive mode - show instructions
            print("Standardization Pipeline Initialized")
            print(f"Config: {args.config}")
            print(f"Output directory: output/standardized/")
            print("\nTo run standardization, use one of the following options:\n")
            print("1. Process sample data (generated from config):")
            print("   python cli.py standardize --config output/rec_config.json --sample\n")
            print("2. Process actual data files:")
            print("   python cli.py standardize --config output/rec_config.json --files users=data/users.csv items=data/items.csv\n")
            print("3. Use Python API:")
            print("   from Standardized_Data_Layer.run import StandardizationPipeline")
            print("   pipeline = StandardizationPipeline('output/rec_config.json')")
            print("   pipeline.run_from_files({'users': 'data/users.csv', 'items': 'data/items.csv'})\n")
            return 0
        
        # Print results
        print("\n" + "="*60)
        print("STANDARDIZATION RESULTS")
        print("="*60)
        print(f"Success: {results.get('success', False)}")
        print(f"Entities processed: {len(results.get('entities_processed', []))}")
        print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
        
        if results.get('summary'):
            summary = results['summary']
            print(f"Total rows: {summary.get('total_rows', 0)}")
            print(f"Errors: {summary.get('error_count', 0)}")
            print(f"Warnings: {summary.get('warning_count', 0)}")
        
        if results.get('errors'):
            errors_list = results['errors']
            if errors_list:
                print("\nErrors encountered:")
                for error in errors_list[:5]:  # Show first 5 errors
                    print(f"  - {error}")
        
        if results.get('warnings'):
            warnings_list = results['warnings']
            if warnings_list:
                print("\nWarnings:")
                for warning in warnings_list[:5]:  # Show first 5 warnings
                    print(f"  - {warning}")
        
        print("\nOutput files written to:", Constants.STANDARDIZED_DIR)
        print("="*60)
        
        return 0 if results.get('success', False) else 1
        
    except Exception as e:
        print(f"Error during standardization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 8: Validation Layer.
    
    Validates standardized datasets before ML processing.
    """
    from Validation_Layer.run import ValidationPipeline
    
    try:
        pipeline = ValidationPipeline(config_path=args.config)
        
        print("Validation Pipeline Initialized")
        print(f"Config: {args.config}")
        print(f"Output directories: output/validated/, output/reports/")
        
        # Note: Actual execution requires standardized data
        # This would be called as: pipeline.run_from_files(file_paths={...})
        print("\nTo run validation with actual data, provide standardized file paths.")
        print("Example: pipeline.run_from_files({'users': 'output/standardized/users.parquet'})")
        
        return 0
        
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        return 1


def cmd_generate_features(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 9: Feature Engineering.
    
    Generates ML-ready features from validated datasets.
    """
    from Feature_Engineering.run import run_feature_engineering
    
    try:
        transactions_path = getattr(args, 'transactions', 'output/standardized/transactions.parquet')
        interactions_path = getattr(args, 'interactions', 'output/standardized/interactions.parquet')
        output_dir = getattr(args, 'output', 'output/features')
        
        print("Feature Engineering Pipeline Started")
        print(f"Config: {args.config}")
        print(f"Transactions: {transactions_path}")
        print(f"Interactions: {interactions_path}")
        print(f"Output: {output_dir}")
        
        run_feature_engineering(
            config_path=args.config,
            transactions_path=transactions_path,
            interactions_path=interactions_path,
            output_dir=output_dir
        )
        
        print("\nFeature engineering completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error during feature generation: {e}", file=sys.stderr)
        return 1


def cmd_train_model(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 10: Recommendation Engine - Training.
    
    Trains recommendation models (Three-Tower, DLRM, XGBoost, LightGBM).
    """
    from Recommendation_Engine.run import train_models
    
    try:
        # Create a namespace object with expected attributes
        train_args = argparse.Namespace(
            config=args.config,
            models_dir=getattr(args, 'models_dir', 'output/models'),
            features_dir=getattr(args, 'features_dir', 'output/features'),
            user_features=getattr(args, 'user_features', None),
            item_features=getattr(args, 'item_features', None),
            interactions=getattr(args, 'interactions', None),
            model_types=getattr(args, 'model_types', 'xgboost,lightgbm'),
            epochs=getattr(args, 'epochs', 10),
            batch_size=getattr(args, 'batch_size', 256),
            learning_rate=getattr(args, 'learning_rate', 0.001),
            output_report=getattr(args, 'output_report', None)
        )
        
        print("Model Training Pipeline Started")
        print(f"Config: {args.config}")
        print(f"Models directory: {train_args.models_dir}")
        print(f"Features directory: {train_args.features_dir}")
        print(f"Model types: {train_args.model_types}")
        
        train_models(train_args)
        
        print("\nModel training completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error during model training: {e}", file=sys.stderr)
        return 1


def cmd_recommend(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 10 & 12: Recommendation Generation.
    
    Generates personalized recommendations for a user.
    """
    from Recommendation_Serving.run import run_serving
    
    try:
        # Build arguments for the serving module
        serving_args = [
            '--config', args.config,
            'recommend',
            '--user', str(args.user),
            '--n', str(getattr(args, 'top_k', 10))
        ]
        
        if hasattr(args, 'session_id') and args.session_id:
            serving_args.extend(['--session-id', args.session_id])
        
        if hasattr(args, 'device_type') and args.device_type:
            serving_args.extend(['--device-type', args.device_type])
        
        if hasattr(args, 'output') and args.output:
            serving_args.extend(['--output', args.output])
        
        if hasattr(args, 'format') and args.format:
            serving_args.extend(['--format', args.format])
        
        if args.verbose:
            serving_args.append('--verbose')
        
        print(f"Generating recommendations for user {args.user}...")
        
        exit_code = run_serving(serving_args)
        
        if exit_code == 0:
            print("\nRecommendations generated successfully.")
        else:
            print("\nRecommendation generation encountered errors.", file=sys.stderr)
        
        return exit_code
        
    except Exception as e:
        print(f"Error during recommendation generation: {e}", file=sys.stderr)
        return 1


def cmd_apply_rules(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 11: Rule Engine.
    
    Applies business rules to ML-generated recommendations.
    """
    from Rule_Engine.run import run_rule_engine
    
    try:
        # Create a namespace object with expected attributes
        rule_args = argparse.Namespace(
            rules=getattr(args, 'rules', 'output/rules.yaml'),
            input=getattr(args, 'input', None),
            output=getattr(args, 'output', None),
            user_id=getattr(args, 'user_id', 'anonymous')
        )
        
        print("Rule Engine Started")
        print(f"Rules file: {rule_args.rules}")
        
        if rule_args.input:
            print(f"Input recommendations: {rule_args.input}")
        else:
            print("No input provided - will use sample recommendations")
        
        result = run_rule_engine(rule_args)
        
        print("\nRule application completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error during rule application: {e}", file=sys.stderr)
        return 1


def cmd_batch_recommend(args: argparse.Namespace) -> int:
    """
    Orchestrate Module 12: Batch Recommendation Serving.
    
    Generates recommendations for multiple users from a file.
    """
    from Recommendation_Serving.run import run_serving
    
    try:
        # Build arguments for the serving module
        serving_args = [
            '--config', args.config,
            'batch-recommend',
            '--input-file', args.input_file,
            '--n', str(getattr(args, 'top_k', 10))
        ]
        
        if hasattr(args, 'format') and args.format:
            serving_args.extend(['--format', args.format])
        
        if hasattr(args, 'output') and args.output:
            serving_args.extend(['--output', args.output])
        
        if args.verbose:
            serving_args.append('--verbose')
        
        print(f"Processing batch recommendations from {args.input_file}...")
        
        exit_code = run_serving(serving_args)
        
        if exit_code == 0:
            print("\nBatch recommendations generated successfully.")
        else:
            print("\nBatch recommendation generation encountered errors.", file=sys.stderr)
        
        return exit_code
        
    except Exception as e:
        print(f"Error during batch recommendation generation: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status and available resources."""
    from shared.constants import Constants
    from pathlib import Path
    
    print("=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    
    # Check config file
    config_path = Path(args.config)
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path} - {'✓ Exists' if config_path.exists() else '✗ Missing'}")
    
    # Check output directories
    print(f"\nOutput Directories:")
    dirs_to_check = [
        ('Standardized Data', Constants.STANDARDIZED_DIR),
        ('Validated Data', Constants.VALIDATED_DIR),
        ('Features', Constants.FEATURES_DIR),
        ('Models', Constants.MODELS_DIR),
        ('Recommendations', Constants.RECOMMENDATIONS_DIR),
        ('Reports', Constants.REPORTS_DIR),
        ('Logs', Constants.LOGS_DIR),
    ]
    
    for name, path in dirs_to_check:
        exists = Path(path).exists()
        print(f"  {name}: {path} - {'✓ Exists' if exists else '✗ Missing'}")
    
    # Check available modules
    print(f"\nAvailable Modules:")
    modules = [
        'Standardized_Data_Layer',
        'Validation_Layer',
        'Feature_Engineering',
        'Recommendation_Engine',
        'Rule_Engine',
        'Recommendation_Serving',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  {module}: ✓ Loaded")
        except ImportError as e:
            print(f"  {module}: ✗ Error - {e}")
    
    print("\n" + "=" * 60)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='intelligent-schema-mapping',
        description='Intelligent Schema Mapping & Business Meaning System - CLI',
        epilog='''
Examples:
  python cli.py standardize
  python cli.py validate
  python cli.py generate-features
  python cli.py train-model --model-types xgboost,lightgbm
  python cli.py recommend --user 1001 --top-k 10
  python cli.py apply-rules --rules rules.yaml --input recs.json
  python cli.py batch-recommend users.csv --top-k 20
  python cli.py status
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    add_common_arguments(parser)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command: standardize
    std_parser = subparsers.add_parser(
        'standardize',
        help='Module 7: Convert mapped data to canonical format',
        description='Convert mapped client data into canonical recommendation-ready datasets.'
    )
    add_common_arguments(std_parser)
    std_parser.add_argument(
        '--sample',
        action='store_true',
        help='Generate and process sample data from configuration'
    )
    std_parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        help='Source data files (format: entity_type=file_path)'
    )
    std_parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after standardization'
    )
    std_parser.add_argument(
        '--no-write',
        action='store_true',
        help='Skip writing output files'
    )
    std_parser.set_defaults(func=cmd_standardize)
    
    # Command: validate
    val_parser = subparsers.add_parser(
        'validate',
        help='Module 8: Validate standardized datasets',
        description='Validate standardized datasets before ML processing.'
    )
    add_common_arguments(val_parser)
    val_parser.set_defaults(func=cmd_validate)
    
    # Command: generate-features
    feat_parser = subparsers.add_parser(
        'generate-features',
        help='Module 9: Generate ML-ready features',
        description='Generate aggregation, temporal, behavioral, and monetary features.'
    )
    add_common_arguments(feat_parser)
    feat_parser.add_argument(
        '--transactions', '-t',
        type=str,
        default='output/standardized/transactions.parquet',
        help='Path to transactions parquet file'
    )
    feat_parser.add_argument(
        '--interactions', '-i',
        type=str,
        default='output/standardized/interactions.parquet',
        help='Path to interactions parquet file'
    )
    feat_parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/features',
        help='Output directory for features'
    )
    feat_parser.set_defaults(func=cmd_generate_features)
    
    # Command: train-model
    train_parser = subparsers.add_parser(
        'train-model',
        help='Module 10: Train recommendation models',
        description='Train Three-Tower, DLRM, XGBoost, and LightGBM models.'
    )
    add_common_arguments(train_parser)
    train_parser.add_argument(
        '--models-dir',
        type=str,
        default='output/models',
        help='Directory to save trained models'
    )
    train_parser.add_argument(
        '--features-dir',
        type=str,
        default='output/features',
        help='Directory containing feature files'
    )
    train_parser.add_argument(
        '--user-features',
        type=str,
        default=None,
        help='Path to user features parquet'
    )
    train_parser.add_argument(
        '--item-features',
        type=str,
        default=None,
        help='Path to item features parquet'
    )
    train_parser.add_argument(
        '--interactions',
        type=str,
        default=None,
        help='Path to interactions parquet'
    )
    train_parser.add_argument(
        '--model-types',
        type=str,
        default='xgboost,lightgbm',
        help='Comma-separated list of model types to train'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for deep models'
    )
    train_parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help='Path to export training report'
    )
    train_parser.set_defaults(func=cmd_train_model)
    
    # Command: recommend
    rec_parser = subparsers.add_parser(
        'recommend',
        help='Generate recommendations for a user',
        description='Generate personalized recommendations using trained models.'
    )
    add_common_arguments(rec_parser)
    rec_parser.add_argument(
        '--user', '-u',
        type=int,
        required=True,
        help='User ID to generate recommendations for'
    )
    rec_parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='Number of recommendations to return'
    )
    rec_parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID for context-aware recommendations'
    )
    rec_parser.add_argument(
        '--device-type',
        type=str,
        default=None,
        help='Device type for context'
    )
    rec_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path'
    )
    rec_parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['json', 'csv', 'parquet'],
        default='json',
        help='Output format'
    )
    rec_parser.set_defaults(func=cmd_recommend)
    
    # Command: apply-rules
    rules_parser = subparsers.add_parser(
        'apply-rules',
        help='Module 11: Apply business rules to recommendations',
        description='Apply filtering, boosting, and campaign rules to recommendations.'
    )
    add_common_arguments(rules_parser)
    rules_parser.add_argument(
        '--rules', '-r',
        type=str,
        default='output/rules.yaml',
        help='Path to YAML rules configuration'
    )
    rules_parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to input recommendations (JSON or Parquet)'
    )
    rules_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output results file'
    )
    rules_parser.add_argument(
        '--user-id',
        type=str,
        default='anonymous',
        help='User ID for context'
    )
    rules_parser.set_defaults(func=cmd_apply_rules)
    
    # Command: batch-recommend
    batch_parser = subparsers.add_parser(
        'batch-recommend',
        help='Generate recommendations for multiple users',
        description='Generate recommendations for multiple users from a CSV file.'
    )
    add_common_arguments(batch_parser)
    batch_parser.add_argument(
        'input_file',
        type=str,
        help='CSV file containing user IDs'
    )
    batch_parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='Number of recommendations per user'
    )
    batch_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path'
    )
    batch_parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['json', 'csv', 'parquet'],
        default='json',
        help='Output format'
    )
    batch_parser.set_defaults(func=cmd_batch_recommend)
    
    # Command: status
    status_parser = subparsers.add_parser(
        'status',
        help='Show system status and available resources',
        description='Display system configuration and module availability.'
    )
    add_common_arguments(status_parser)
    status_parser.set_defaults(func=cmd_status)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Ensure output directories exist
    from shared.constants import Constants
    Constants.ensure_directories()
    
    # Execute the command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
