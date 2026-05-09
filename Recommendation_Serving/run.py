"""
Run - CLI entry point for Recommendation Serving module.

Provides command-line interface for running recommendation serving operations.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from shared.logger import get_logger, Logger
from shared.constants import Constants
from .recommendation_service import RecommendationService
from .realtime_recommendation import RealtimeRecommendation
from .batch_recommendation import BatchRecommendation
from .export_manager import ExportManager

logger = get_logger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    Logger.set_log_directory(Constants.LOGS_DIR)
    logger.setLevel(level)


def create_service(config_path: Optional[str] = None) -> RecommendationService:
    """Create a RecommendationService instance."""
    config = {}
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    service = RecommendationService(
        config=config,
        cache_ttl=3600,
        enable_caching=True
    )
    
    return service


def cmd_recommend(args: argparse.Namespace) -> int:
    """Handle recommend command."""
    setup_logging(args.verbose)
    
    service = create_service(args.config)
    realtime = RealtimeRecommendation(service)
    
    logger.info(f"Getting recommendations for user {args.user}")
    
    context = {}
    if args.session_id:
        context['session_id'] = args.session_id
    if args.device_type:
        context['device_type'] = args.device_type
    
    result = realtime.get_recommendations(
        user_id=args.user,
        n=args.n,
        context=context
    )
    
    # Output results
    if args.output:
        export_manager = ExportManager()
        output_path = export_manager.export_user_recommendations(
            user_id=args.user,
            recommendations=result.get('recommendations', []),
            format=args.format,
            metadata=result.get('metadata')
        )
        print(f"Recommendations exported to: {output_path}")
    else:
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            # Human-readable output
            for rec in result.get('recommendations', []):
                rank = rec.get('rank', 0)
                item_id = rec.get('item_id')
                score = rec.get('score', 0)
                reason = rec.get('reason', '')
                explanation = rec.get('explanation', '')
                
                print(f"{rank}. Item {item_id} (score: {score:.4f})")
                if reason:
                    print(f"   Reason: {reason}")
                if explanation:
                    print(f"   Explanation: {explanation}")
    
    return 0 if 'error' not in result else 1


def cmd_batch_recommend(args: argparse.Namespace) -> int:
    """Handle batch-recommend command."""
    setup_logging(args.verbose)
    
    service = create_service(args.config)
    batch = BatchRecommendation(service)
    
    logger.info(f"Processing batch from {args.input_file}")
    
    try:
        output_path = batch.generate_from_file(
            input_file=args.input_file,
            n=args.n,
            output_format=args.format
        )
        print(f"Batch recommendations exported to: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_similar_items(args: argparse.Namespace) -> int:
    """Handle similar-items command."""
    setup_logging(args.verbose)
    
    service = create_service(args.config)
    realtime = RealtimeRecommendation(service)
    
    logger.info(f"Getting similar items for item {args.item}")
    
    result = realtime.get_similar_items(
        item_id=args.item,
        n=args.n
    )
    
    if args.output:
        export_manager = ExportManager()
        output_path = export_manager.export_user_recommendations(
            user_id=0,  # Not user-specific
            recommendations=result.get('recommendations', []),
            format=args.format
        )
        print(f"Similar items exported to: {output_path}")
    else:
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            for rec in result.get('recommendations', []):
                rank = rec.get('rank', 0)
                item_id = rec.get('item_id')
                score = rec.get('score', 0)
                print(f"{rank}. Item {item_id} (score: {score:.4f})")
    
    return 0 if 'error' not in result else 1


def cmd_export(args: argparse.Namespace) -> int:
    """Handle export command."""
    setup_logging(args.verbose)
    
    export_manager = ExportManager(args.output_dir)
    
    if args.action == 'list':
        files = export_manager.list_exports(pattern=args.pattern)
        print(f"Found {len(files)} export files:")
        for f in files:
            print(f"  {f['filename']} ({f['size_bytes']} bytes)")
    
    elif args.action == 'cleanup':
        deleted = export_manager.cleanup_old_exports(
            max_age_days=args.days,
            pattern=args.pattern
        )
        print(f"Cleaned up {deleted} old export files")
    
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Handle stats command."""
    setup_logging(args.verbose)
    
    service = create_service(args.config)
    stats = service.get_service_stats()
    
    print(json.dumps(stats, indent=2, default=str))
    return 0


def run_serving(argv: Optional[list] = None) -> int:
    """
    Main entry point for Recommendation Serving CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog='recommendation-serving',
        description='CLI for recommendation serving operations'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file',
        default='output/rec_config.json'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Recommend command
    recommend_parser = subparsers.add_parser(
        'recommend',
        help='Get recommendations for a user'
    )
    recommend_parser.add_argument('--user', '-u', type=int, required=True,
                                   help='User ID')
    recommend_parser.add_argument('--n', '-n', type=int, default=10,
                                   help='Number of recommendations')
    recommend_parser.add_argument('--session-id', help='Session ID')
    recommend_parser.add_argument('--device-type', help='Device type')
    recommend_parser.add_argument('--output', '-o', help='Output file path')
    recommend_parser.add_argument('--format', '-f', choices=['json', 'csv', 'parquet'],
                                   default='json', help='Output format')
    recommend_parser.set_defaults(func=cmd_recommend)
    
    # Batch recommend command
    batch_parser = subparsers.add_parser(
        'batch-recommend',
        help='Generate recommendations for multiple users'
    )
    batch_parser.add_argument('--input-file', '-i', required=True,
                               help='Input file with user IDs (CSV or JSON)')
    batch_parser.add_argument('--n', '-n', type=int, default=10,
                               help='Number of recommendations per user')
    batch_parser.add_argument('--format', '-f', choices=['json', 'csv', 'parquet'],
                               default='json', help='Output format')
    batch_parser.set_defaults(func=cmd_batch_recommend)
    
    # Similar items command
    similar_parser = subparsers.add_parser(
        'similar-items',
        help='Get similar items for an item'
    )
    similar_parser.add_argument('--item', '-i', type=int, required=True,
                                 help='Item ID')
    similar_parser.add_argument('--n', '-n', type=int, default=10,
                                 help='Number of recommendations')
    similar_parser.add_argument('--output', '-o', help='Output file path')
    similar_parser.add_argument('--format', '-f', choices=['json', 'csv', 'parquet'],
                                 default='json', help='Output format')
    similar_parser.set_defaults(func=cmd_similar_items)
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Manage export files'
    )
    export_parser.add_argument('--action', '-a', choices=['list', 'cleanup'],
                                required=True, help='Export action')
    export_parser.add_argument('--output-dir', help='Output directory')
    export_parser.add_argument('--pattern', help='Filename pattern')
    export_parser.add_argument('--days', type=int, default=7,
                                help='Max age in days for cleanup')
    export_parser.set_defaults(func=cmd_export)
    
    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Show service statistics'
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # Parse and execute
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(run_serving())
