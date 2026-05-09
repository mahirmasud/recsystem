"""
Rule Engine Run Module - CLI entry point for rule execution.

Usage:
    python -m Rule_Engine.run --rules rules.yaml --input recommendations.json --output result.json
    python -m Rule_Engine.run --test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from shared.logger import setup_logger, get_logger
from shared.file_loader import FileLoader

from .rule_loader import RuleLoader
from .rule_parser import RuleParser
from .rule_executor import RuleExecutor
from .chain_executor import ChainExecutor, RuleChain, ChainStep, ChainCondition
from .explanation_logger import ExplanationLogger


def load_recommendations(input_path: str) -> List[Dict[str, Any]]:
    """Load recommendations from file."""
    loader = FileLoader()
    
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
            return data.get('recommendations', data) if isinstance(data, dict) else data
    elif input_path.endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(input_path)
            return df.to_dict('records')
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet: {e}")
    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif output_path.endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.DataFrame(results.get('recommendations', []))
            df.to_parquet(path, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save parquet: {e}")
    else:
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


def run_rule_engine(args):
    """Main function to run the rule engine."""
    logger = get_logger(__name__)
    
    # Setup
    logger.info("=" * 60)
    logger.info("RULE ENGINE EXECUTION")
    logger.info("=" * 60)
    
    # Load rules
    logger.info(f"Loading rules from: {args.rules}")
    rule_loader = RuleLoader(args.rules)
    raw_rules = rule_loader.load()
    rule_loader.validate_structure()
    
    # Parse rules
    logger.info("Parsing rules...")
    rule_parser = RuleParser()
    parsed_rules = rule_parser.parse_all(raw_rules)
    logger.info(f"Parsed {len(parsed_rules)} active rules")
    
    # Load recommendations
    if args.input:
        logger.info(f"Loading recommendations from: {args.input}")
        recommendations = load_recommendations(args.input)
        logger.info(f"Loaded {len(recommendations)} recommendations")
    else:
        # Generate sample recommendations for testing
        logger.info("No input provided, generating sample recommendations...")
        recommendations = [
            {'item_id': f'item_{i}', 'score': 0.9 - i * 0.05, 'category': f'cat_{i % 3}'}
            for i in range(20)
        ]
    
    # Build item catalog (mock for now)
    item_catalog = {
        rec['item_id']: {
            'category': rec.get('category', 'unknown'),
            'price': 10.0 + rec['score'] * 100,
            'in_stock': True,
            'profit_margin': 0.2 + rec['score'] * 0.3,
            'is_trending': rec['score'] > 0.7,
        }
        for rec in recommendations
    }
    
    # User context
    user_context = {
        'user_id': args.user_id or 'anonymous',
        'segment': 'standard',
        'purchased_items': set(),
    }
    
    # Execute rules
    logger.info("Executing rules...")
    executor = RuleExecutor()
    result = executor.execute(
        recommendations=recommendations,
        rules=parsed_rules,
        user_context=user_context,
        item_catalog=item_catalog
    )
    
    # Prepare output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'status': result.status.value,
        'input_count': len(recommendations),
        'output_count': len(result.recommendations),
        'rules_applied': result.rules_applied,
        'rules_skipped': result.rules_skipped,
        'rules_failed': result.rules_failed,
        'execution_time_ms': result.execution_time_ms,
        'recommendations': result.recommendations,
        'explanations': result.explanations
    }
    
    # Save results
    if args.output:
        logger.info(f"Saving results to: {args.output}")
        save_results(output_data, args.output)
    else:
        # Print summary
        print("\n" + "=" * 60)
        print("RULE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status: {result.status.value}")
        print(f"Input items: {len(recommendations)}")
        print(f"Output items: {len(result.recommendations)}")
        print(f"Rules applied: {len(result.rules_applied)}")
        print(f"Rules skipped: {len(result.rules_skipped)}")
        print(f"Rules failed: {len(result.rules_failed)}")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")
        print("=" * 60)
        
        if result.recommendations:
            print("\nTOP 5 RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                score = rec.get('score', 0)
                item_id = rec.get('item_id', 'unknown')
                reason = rec.get('boost_reason', rec.get('campaign_applied', ''))
                print(f"  {i}. {item_id} (score: {score:.4f}) {reason}")
    
    # Get execution stats
    stats = executor.get_execution_stats()
    logger.info(f"Execution stats: {stats}")
    
    logger.info("Rule engine execution completed")
    
    return output_data


def run_test():
    """Run basic tests."""
    logger = get_logger(__name__)
    
    print("\n" + "=" * 60)
    print("RULE ENGINE TESTS")
    print("=" * 60)
    
    # Test 1: Rule loading
    print("\n[Test 1] Loading rules...")
    test_rules = {
        'rules': [
            {
                'id': 'test_filter_1',
                'type': 'filter',
                'name': 'Test Filter',
                'enabled': True,
                'priority': 10,
                'conditions': {'out_of_stock': True},
                'parameters': {}
            },
            {
                'id': 'test_boost_1',
                'type': 'boost',
                'name': 'Test Boost',
                'enabled': True,
                'priority': 5,
                'conditions': {'high_margin': True},
                'parameters': {'boost_factor': 1.5}
            }
        ]
    }
    
    loader = RuleLoader()
    raw_rules = loader.load_from_dict(test_rules)
    assert len(raw_rules['rules']) == 2
    print("  ✓ Rules loaded successfully")
    
    # Test 2: Rule parsing
    print("\n[Test 2] Parsing rules...")
    parser = RuleParser()
    parsed_rules = parser.parse_all(raw_rules)
    assert len(parsed_rules) == 2
    print("  ✓ Rules parsed successfully")
    
    # Test 3: Rule execution
    print("\n[Test 3] Executing rules...")
    recommendations = [
        {'item_id': 'item_1', 'score': 0.8},
        {'item_id': 'item_2', 'score': 0.7},
        {'item_id': 'item_3', 'score': 0.6},
    ]
    
    item_catalog = {
        'item_1': {'in_stock': True, 'profit_margin': 0.4},
        'item_2': {'in_stock': False, 'profit_margin': 0.3},
        'item_3': {'in_stock': True, 'profit_margin': 0.2},
    }
    
    user_context = {'user_id': 'test_user'}
    
    executor = RuleExecutor()
    result = executor.execute(
        recommendations=recommendations,
        rules=parsed_rules,
        user_context=user_context,
        item_catalog=item_catalog
    )
    
    assert result.status.value == 'success'
    assert len(result.recommendations) <= len(recommendations)
    print(f"  ✓ Rules executed successfully")
    print(f"    Input: {len(recommendations)} items")
    print(f"    Output: {len(result.recommendations)} items")
    print(f"    Rules applied: {result.rules_applied}")
    
    # Test 4: Chain execution
    print("\n[Test 4] Testing chain execution...")
    chain = RuleChain(
        chain_id='test_chain',
        name='Test Chain',
        steps=[
            ChainStep(
                step_id='step1',
                rule_ids=['test_filter_1'],
                condition=ChainCondition.ALWAYS
            ),
            ChainStep(
                step_id='step2',
                rule_ids=['test_boost_1'],
                condition=ChainCondition.IF_ITEMS_REMAIN
            )
        ]
    )
    
    chain_executor = ChainExecutor()
    chain_executor.register_chain(chain)
    chain_executor.register_rules(parsed_rules)
    
    chain_result = chain_executor.execute_chain(
        chain_id='test_chain',
        recommendations=recommendations,
        user_context=user_context,
        item_catalog=item_catalog
    )
    
    assert chain_result.status.value == 'success'
    print("  ✓ Chain executed successfully")
    
    # Test 5: Explanation logging
    print("\n[Test 5] Testing explanation logging...")
    expl_logger = ExplanationLogger()
    expl_logger.log_boost(
        item_id='item_1',
        rule={'id': 'test_boost_1', 'name': 'Test Boost'},
        reason='high_margin',
        score_before=0.8,
        score_after=1.2
    )
    assert len(expl_logger) == 1
    print("  ✓ Explanation logging works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60 + "\n")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rule Engine - Apply business rules to recommendations'
    )
    
    parser.add_argument(
        '--rules', '-r',
        type=str,
        help='Path to YAML rules configuration file'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input recommendations file (JSON or Parquet)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output results file'
    )
    
    parser.add_argument(
        '--user-id', '-u',
        type=str,
        default='anonymous',
        help='User ID for context'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run tests instead of execution'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level=args.log_level)
    
    if args.test:
        success = run_test()
        sys.exit(0 if success else 1)
    else:
        try:
            run_rule_engine(args)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
