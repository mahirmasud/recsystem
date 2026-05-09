"""
Test Rules - Unit tests for Rule Engine module.

Run with: python -m Rule_Engine.test_rules
"""

import unittest
import json
from datetime import datetime, timedelta
from pathlib import Path

from .rule_loader import RuleLoader
from .rule_parser import RuleParser, RuleType, RuleAction, ParsedRule
from .filter_rules import FilterRules
from .boost_rules import BoostRules
from .conditional_rules import ConditionalRules
from .context_rules import ContextRules
from .campaign_rules import CampaignRules, Campaign
from .rule_executor import RuleExecutor, ExecutionStatus
from .chain_executor import ChainExecutor, RuleChain, ChainStep, ChainCondition
from .explanation_logger import ExplanationLogger


class TestRuleLoader(unittest.TestCase):
    """Tests for RuleLoader."""
    
    def test_load_from_dict(self):
        """Test loading rules from dictionary."""
        rules_dict = {
            'rules': [
                {'id': 'rule1', 'type': 'filter', 'enabled': True},
                {'id': 'rule2', 'type': 'boost', 'enabled': True}
            ]
        }
        
        loader = RuleLoader()
        result = loader.load_from_dict(rules_dict)
        
        self.assertEqual(len(result['rules']), 2)
        self.assertEqual(result['rules'][0]['id'], 'rule1')
    
    def test_get_rules_by_type(self):
        """Test filtering rules by type."""
        rules_dict = {
            'rules': [
                {'id': 'f1', 'type': 'filter'},
                {'id': 'b1', 'type': 'boost'},
                {'id': 'f2', 'type': 'filter'}
            ]
        }
        
        loader = RuleLoader()
        loader.load_from_dict(rules_dict)
        
        filters = loader.get_rules_by_type('filter')
        self.assertEqual(len(filters), 2)
    
    def test_validate_structure(self):
        """Test rule structure validation."""
        valid_rules = {'rules': [{'id': 'r1', 'type': 'filter'}]}
        invalid_rules = {'rules': [{'id': 'r1'}]}  # Missing type
        
        loader = RuleLoader()
        loader.load_from_dict(valid_rules)
        self.assertTrue(loader.validate_structure())
        
        loader.load_from_dict(invalid_rules)
        with self.assertRaises(Exception):
            loader.validate_structure()


class TestRuleParser(unittest.TestCase):
    """Tests for RuleParser."""
    
    def test_parse_single_rule(self):
        """Test parsing a single rule."""
        raw_rule = {
            'id': 'test_rule',
            'type': 'boost',
            'name': 'Test Boost Rule',
            'priority': 10,
            'conditions': {'high_margin': True},
            'parameters': {'boost_factor': 1.5}
        }
        
        parser = RuleParser()
        parsed = parser.parse_single(raw_rule)
        
        self.assertEqual(parsed.id, 'test_rule')
        self.assertEqual(parsed.type, RuleType.BOOST)
        self.assertEqual(parsed.priority, 10)
    
    def test_parse_all_rules(self):
        """Test parsing multiple rules."""
        raw_rules = {
            'rules': [
                {'id': 'r1', 'type': 'filter', 'priority': 5},
                {'id': 'r2', 'type': 'boost', 'priority': 10},
                {'id': 'r3', 'type': 'conditional', 'priority': 7}
            ]
        }
        
        parser = RuleParser()
        parsed = parser.parse_all(raw_rules)
        
        self.assertEqual(len(parsed), 3)
        # Should be sorted by priority (descending)
        self.assertEqual(parsed[0].id, 'r2')  # priority 10
        self.assertEqual(parsed[1].id, 'r3')  # priority 7
        self.assertEqual(parsed[2].id, 'r1')  # priority 5
    
    def test_parse_disabled_rule(self):
        """Test that disabled rules are excluded."""
        raw_rules = {
            'rules': [
                {'id': 'r1', 'type': 'filter', 'enabled': True},
                {'id': 'r2', 'type': 'boost', 'enabled': False}
            ]
        }
        
        parser = RuleParser()
        parsed = parser.parse_all(raw_rules)
        
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].id, 'r1')


class TestFilterRules(unittest.TestCase):
    """Tests for FilterRules."""
    
    def test_filter_out_of_stock(self):
        """Test filtering out of stock items."""
        recommendations = [
            {'item_id': 'i1', 'score': 0.9},
            {'item_id': 'i2', 'score': 0.8},
            {'item_id': 'i3', 'score': 0.7}
        ]
        
        item_catalog = {
            'i1': {'in_stock': True},
            'i2': {'in_stock': False},
            'i3': {'in_stock': True}
        }
        
        rule = ParsedRule(
            id='filter_oos',
            type=RuleType.FILTER,
            name='Filter OOS',
            description='',
            enabled=True,
            priority=0,
            action=RuleAction.EXCLUDE,
            conditions={'out_of_stock': True},
            parameters={},
            metadata={}
        )
        
        filter_rules = FilterRules()
        result = filter_rules.apply(recommendations, rule, {}, item_catalog)
        
        self.assertEqual(len(result), 2)
        self.assertNotIn('i2', [r['item_id'] for r in result])
    
    def test_filter_no_matches(self):
        """Test filtering when no items match filter criteria."""
        recommendations = [{'item_id': 'i1', 'score': 0.9}]
        item_catalog = {'i1': {'in_stock': True}}
        
        rule = ParsedRule(
            id='filter_oos',
            type=RuleType.FILTER,
            name='Filter OOS',
            description='',
            enabled=True,
            priority=0,
            action=RuleAction.EXCLUDE,
            conditions={'out_of_stock': True},
            parameters={},
            metadata={}
        )
        
        filter_rules = FilterRules()
        result = filter_rules.apply(recommendations, rule, {}, item_catalog)
        
        self.assertEqual(len(result), 1)


class TestBoostRules(unittest.TestCase):
    """Tests for BoostRules."""
    
    def test_boost_high_margin(self):
        """Test boosting high margin items."""
        recommendations = [
            {'item_id': 'i1', 'score': 0.5},
            {'item_id': 'i2', 'score': 0.8}
        ]
        
        item_catalog = {
            'i1': {'profit_margin': 0.5},  # High margin
            'i2': {'profit_margin': 0.1}   # Low margin
        }
        
        rule = ParsedRule(
            id='boost_margin',
            type=RuleType.BOOST,
            name='Boost Margin',
            description='',
            enabled=True,
            priority=0,
            action=RuleAction.BOOST_SCORE,
            conditions={'high_margin': True, 'margin_threshold': 0.3},
            parameters={'boost_factor': 2.0},
            metadata={}
        )
        
        boost_rules = BoostRules()
        result = boost_rules.apply(recommendations, rule, {}, item_catalog)
        
        # i1 should be boosted and now have higher score
        i1_result = next(r for r in result if r['item_id'] == 'i1')
        self.assertGreater(i1_result['score'], 0.5)


class TestRuleExecutor(unittest.TestCase):
    """Tests for RuleExecutor."""
    
    def test_execute_multiple_rules(self):
        """Test executing multiple rules in sequence."""
        recommendations = [
            {'item_id': 'i1', 'score': 0.9},
            {'item_id': 'i2', 'score': 0.8},
            {'item_id': 'i3', 'score': 0.7}
        ]
        
        item_catalog = {
            'i1': {'in_stock': True, 'profit_margin': 0.4},
            'i2': {'in_stock': False, 'profit_margin': 0.3},
            'i3': {'in_stock': True, 'profit_margin': 0.2}
        }
        
        rules = [
            ParsedRule(
                id='filter_oos',
                type=RuleType.FILTER,
                name='Filter OOS',
                description='',
                enabled=True,
                priority=10,
                action=RuleAction.EXCLUDE,
                conditions={'out_of_stock': True},
                parameters={},
                metadata={}
            ),
            ParsedRule(
                id='boost_margin',
                type=RuleType.BOOST,
                name='Boost Margin',
                description='',
                enabled=True,
                priority=5,
                action=RuleAction.BOOST_SCORE,
                conditions={'high_margin': True, 'margin_threshold': 0.3},
                parameters={'boost_factor': 1.5},
                metadata={}
            )
        ]
        
        executor = RuleExecutor()
        result = executor.execute(
            recommendations=recommendations,
            rules=rules,
            user_context={},
            item_catalog=item_catalog
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertLess(len(result.recommendations), len(recommendations))
        self.assertIn('filter_oos', result.rules_applied)


class TestChainExecutor(unittest.TestCase):
    """Tests for ChainExecutor."""
    
    def test_execute_chain(self):
        """Test executing a rule chain."""
        chain = RuleChain(
            chain_id='test_chain',
            name='Test Chain',
            steps=[
                ChainStep(
                    step_id='step1',
                    rule_ids=['filter_oos'],
                    condition=ChainCondition.ALWAYS
                ),
                ChainStep(
                    step_id='step2',
                    rule_ids=['boost_margin'],
                    condition=ChainCondition.IF_ITEMS_REMAIN
                )
            ]
        )
        
        rules = [
            ParsedRule(
                id='filter_oos',
                type=RuleType.FILTER,
                name='Filter OOS',
                description='',
                enabled=True,
                priority=10,
                action=RuleAction.EXCLUDE,
                conditions={'out_of_stock': True},
                parameters={},
                metadata={}
            ),
            ParsedRule(
                id='boost_margin',
                type=RuleType.BOOST,
                name='Boost Margin',
                description='',
                enabled=True,
                priority=5,
                action=RuleAction.BOOST_SCORE,
                conditions={'high_margin': True},
                parameters={'boost_factor': 1.5},
                metadata={}
            )
        ]
        
        recommendations = [
            {'item_id': 'i1', 'score': 0.9},
            {'item_id': 'i2', 'score': 0.8}
        ]
        
        item_catalog = {
            'i1': {'in_stock': True, 'profit_margin': 0.4},
            'i2': {'in_stock': False, 'profit_margin': 0.3}
        }
        
        executor = ChainExecutor()
        executor.register_chain(chain)
        executor.register_rules(rules)
        
        result = executor.execute_chain(
            chain_id='test_chain',
            recommendations=recommendations,
            user_context={},
            item_catalog=item_catalog
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)


class TestExplanationLogger(unittest.TestCase):
    """Tests for ExplanationLogger."""
    
    def test_log_explanation(self):
        """Test logging explanations."""
        logger = ExplanationLogger()
        
        logger.log(
            item_id='item_1',
            rule_id='rule_1',
            rule_name='Test Rule',
            rule_type='boost',
            action='boost',
            reason='high_margin',
            score_before=0.5,
            score_after=0.75
        )
        
        self.assertEqual(len(logger), 1)
        
        explanations = logger.get_item_explanations('item_1')
        self.assertEqual(len(explanations), 1)
        self.assertEqual(explanations[0].rule_id, 'rule_1')
    
    def test_generate_summary(self):
        """Test generating explanation summary."""
        logger = ExplanationLogger()
        
        logger.log(
            item_id='i1',
            rule_id='r1',
            rule_name='Rule 1',
            rule_type='boost',
            action='boost',
            reason='test',
            score_before=0.5,
            score_after=0.75
        )
        
        summary = logger.get_summary()
        
        self.assertEqual(summary['total_explanations'], 1)
        self.assertEqual(summary['by_action']['boost'], 1)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRuleLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleParser))
    suite.addTests(loader.loadTestsFromTestCase(TestFilterRules))
    suite.addTests(loader.loadTestsFromTestCase(TestBoostRules))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestChainExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestExplanationLogger))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
