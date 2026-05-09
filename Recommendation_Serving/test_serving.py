"""
Test Serving - Unit tests for Recommendation Serving module.

Tests all components of the recommendation serving layer.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
import os
from pathlib import Path


class TestRequestParser(unittest.TestCase):
    """Tests for RequestParser."""
    
    def setUp(self):
        from .request_parser import RequestParser
        self.parser = RequestParser()
    
    def test_parse_user_request(self):
        """Test parsing a user request."""
        request = self.parser.parse_user_request(
            user_id=123,
            n=10,
            context={'device_type': 'mobile'}
        )
        
        self.assertEqual(request.user_id, 123)
        self.assertEqual(request.n_recommendations, 10)
        self.assertEqual(request.device_type, 'mobile')
    
    def test_parse_item_request(self):
        """Test parsing an item request."""
        request = self.parser.parse_item_request(
            item_id=456,
            n=5
        )
        
        self.assertEqual(request.item_id, 456)
        self.assertEqual(request.n_recommendations, 5)
    
    def test_validate_request_valid(self):
        """Test validation of valid request."""
        from .request_parser import RecommendationRequest
        
        request = RecommendationRequest(user_id=123, n_recommendations=10)
        result = self.parser.validate_request(request)
        
        self.assertTrue(result)
    
    def test_validate_request_invalid(self):
        """Test validation catches invalid requests."""
        from .request_parser import RecommendationRequest
        
        # Missing both user_id and item_id
        request = RecommendationRequest(n_recommendations=10)
        
        with self.assertRaises(ValueError):
            self.parser.validate_request(request)


class TestCacheHandler(unittest.TestCase):
    """Tests for CacheHandler."""
    
    def setUp(self):
        from .cache_handler import CacheHandler
        self.cache = CacheHandler(ttl_seconds=60, max_entries=100)
    
    def test_set_and_get(self):
        """Test setting and getting cache values."""
        self.cache.set(
            value=['item1', 'item2'],
            user_id=123
        )
        
        result = self.cache.get(user_id=123)
        self.assertEqual(result, ['item1', 'item2'])
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get(user_id=999)
        self.assertIsNone(result)
    
    def test_invalidate_user(self):
        """Test invalidating user cache."""
        self.cache.set(value=['recs'], user_id=123)
        self.cache.set(value=['recs'], user_id=456)
        
        count = self.cache.invalidate(user_id=123)
        self.assertEqual(count, 1)
        
        self.assertIsNone(self.cache.get(user_id=123))
        self.assertIsNotNone(self.cache.get(user_id=456))
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        self.cache.set(value=['recs'], user_id=123)
        self.cache.get(user_id=123)
        self.cache.get(user_id=999)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)


class TestSessionHandler(unittest.TestCase):
    """Tests for SessionHandler."""
    
    def setUp(self):
        from .session_handler import SessionHandler
        self.handler = SessionHandler(ttl_minutes=30)
    
    def test_create_session(self):
        """Test creating a session."""
        session = self.handler.create_session(user_id=123)
        
        self.assertEqual(session.user_id, 123)
        self.assertTrue(session.is_active())
    
    def test_get_session(self):
        """Test retrieving a session."""
        created = self.handler.create_session(session_id='test123')
        retrieved = self.handler.get_session('test123')
        
        self.assertEqual(retrieved.session_id, 'test123')
    
    def test_add_interaction(self):
        """Test adding interaction to session."""
        session = self.handler.create_session(user_id=123)
        
        self.handler.add_session_interaction(
            session_id=session.session_id,
            interaction_type='view',
            item_id=456
        )
        
        signals = self.handler.get_session_signals(session.session_id)
        self.assertIn(456, signals['viewed_items'])
    
    def test_get_session_signals(self):
        """Test extracting session signals."""
        session = self.handler.create_session(user_id=123)
        
        self.handler.add_session_interaction(
            session_id=session.session_id,
            interaction_type='view',
            item_id=100
        )
        self.handler.add_session_interaction(
            session_id=session.session_id,
            interaction_type='add_to_cart',
            item_id=200
        )
        
        signals = self.handler.get_session_signals(session.session_id)
        
        self.assertEqual(len(signals['viewed_items']), 1)
        self.assertEqual(len(signals['cart_items']), 1)


class TestRecommendationFormatter(unittest.TestCase):
    """Tests for RecommendationFormatter."""
    
    def setUp(self):
        from .recommendation_formatter import RecommendationFormatter
        self.formatter = RecommendationFormatter()
    
    def test_format_json(self):
        """Test JSON formatting."""
        recs = [
            {'rank': 1, 'item_id': 100, 'score': 0.95},
            {'rank': 2, 'item_id': 200, 'score': 0.85}
        ]
        
        result = self.formatter.format_json(recs)
        
        self.assertIn('recommendations', result)
        self.assertIn('generated_at', result)
        self.assertEqual(result['count'], 2)
    
    def test_format_csv_rows(self):
        """Test CSV row formatting."""
        recs = [
            {'rank': 1, 'item_id': 100, 'score': 0.95, 'reason': 'Popular'}
        ]
        
        rows = self.formatter.format_csv_rows(recs, user_id=123)
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['user_id'], 123)
        self.assertEqual(rows[0]['item_id'], 100)
    
    def test_format_human_readable(self):
        """Test human-readable formatting."""
        recs = [
            {
                'rank': 1, 
                'item_id': 100, 
                'item_name': 'Test Item',
                'score': 0.95,
                'price': 29.99,
                'reason': 'Recommended'
            }
        ]
        
        output = self.formatter.format_human_readable(recs)
        
        self.assertIn('Test Item', output)
        self.assertIn('$29.99', output)


class TestExplanationGenerator(unittest.TestCase):
    """Tests for ExplanationGenerator."""
    
    def setUp(self):
        from .explanation_generator import ExplanationGenerator
        self.generator = ExplanationGenerator()
    
    def test_generate_explanation_collaborative(self):
        """Test collaborative explanation."""
        rec = {'source': 'collaborative', 'item_id': 100}
        explanation = self.generator.generate_explanation(rec)
        
        self.assertIn('similar', explanation.lower())
    
    def test_generate_explanation_popularity(self):
        """Test popularity explanation."""
        rec = {'source': 'popularity', 'item_id': 100}
        explanation = self.generator.generate_explanation(rec)
        
        self.assertIn('trending', explanation.lower())
    
    def test_generate_batch_explanations(self):
        """Test batch explanation generation."""
        recs = [
            {'source': 'collaborative', 'item_id': 100},
            {'source': 'popularity', 'item_id': 200}
        ]
        
        explanations = self.generator.generate_batch_explanations(recs)
        
        self.assertEqual(len(explanations), 2)
        self.assertIn(100, explanations)
        self.assertIn(200, explanations)


class TestExportManager(unittest.TestCase):
    """Tests for ExportManager."""
    
    def setUp(self):
        from .export_manager import ExportManager
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ExportManager(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_user_json(self):
        """Test JSON export."""
        recs = [{'rank': 1, 'item_id': 100, 'score': 0.95}]
        
        path = self.manager.export_user_recommendations(
            user_id=123,
            recommendations=recs,
            format='json'
        )
        
        self.assertTrue(Path(path).exists())
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['user_id'], 123)
        self.assertEqual(len(data['recommendations']), 1)
    
    def test_export_user_csv(self):
        """Test CSV export."""
        recs = [{'rank': 1, 'item_id': 100, 'score': 0.95, 'reason': 'Test'}]
        
        path = self.manager.export_user_recommendations(
            user_id=123,
            recommendations=recs,
            format='csv'
        )
        
        self.assertTrue(Path(path).exists())
    
    def test_list_exports(self):
        """Test listing exports."""
        recs = [{'rank': 1, 'item_id': 100, 'score': 0.95}]
        self.manager.export_user_recommendations(123, recs, 'json')
        
        files = self.manager.list_exports()
        
        self.assertGreater(len(files), 0)


class TestTraceLogger(unittest.TestCase):
    """Tests for TraceLogger."""
    
    def setUp(self):
        from .trace_logger import TraceLogger
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TraceLogger(
            log_dir=self.temp_dir,
            enabled=True,
            persist_to_file=False
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_start_and_end_trace(self):
        """Test trace lifecycle."""
        self.logger.start_trace('trace1', {'user_id': 123})
        self.logger.log_stage('trace1', 'generation', {'candidates': 100})
        self.logger.end_trace('trace1', {'recommendations': 10})
        
        trace = self.logger.get_trace('trace1')
        
        self.assertIsNotNone(trace)
        self.assertFalse(trace['is_complete'] is False)
    
    def test_get_performance_stats(self):
        """Test performance statistics."""
        self.logger.start_trace('trace1', {'user_id': 123})
        self.logger.end_trace('trace1', {})
        
        stats = self.logger.get_performance_stats()
        
        self.assertIn('total_traces', stats)


class TestRealtimeRecommendation(unittest.TestCase):
    """Tests for RealtimeRecommendation."""
    
    def setUp(self):
        from .recommendation_service import RecommendationService
        from .realtime_recommendation import RealtimeRecommendation
        
        self.service = Mock(spec=RecommendationService)
        self.service.get_recommendations.return_value = {
            'recommendations': [
                {'item_id': 100, 'score': 0.95},
                {'item_id': 200, 'score': 0.85}
            ],
            'metadata': {}
        }
        self.service.session_handler = None
        
        self.realtime = RealtimeRecommendation(
            self.service,
            enable_tracing=False
        )
    
    def test_get_recommendations(self):
        """Test getting real-time recommendations."""
        result = self.realtime.get_recommendations(user_id=123, n=10)
        
        self.assertIn('recommendations', result)
        self.assertEqual(len(result['recommendations']), 2)
        self.assertIn('latency_ms', result)


class TestBatchRecommendation(unittest.TestCase):
    """Tests for BatchRecommendation."""
    
    def setUp(self):
        from .recommendation_service import RecommendationService
        from .batch_recommendation import BatchRecommendation
        
        self.service = Mock(spec=RecommendationService)
        self.service.get_recommendations.return_value = {
            'recommendations': [{'item_id': 100, 'score': 0.9}],
            'metadata': {}
        }
        
        self.batch = BatchRecommendation(self.service)
    
    def test_generate_for_users(self):
        """Test batch generation for multiple users."""
        result = self.batch.generate_for_users([1, 2, 3], n=5)
        
        self.assertEqual(result['total_users'], 3)
        self.assertEqual(result['successful'], 3)
        self.assertEqual(result['failed'], 0)
    
    def test_get_batch_stats(self):
        """Test batch statistics."""
        batch_result = {
            'total_users': 10,
            'successful': 9,
            'failed': 1,
            'results': [
                {'user_id': 1, 'recommendations': [1, 2, 3]}
            ],
            'processing_time_ms': 1000
        }
        
        stats = self.batch.get_batch_stats(batch_result)
        
        self.assertIn('success_rate', stats)
        self.assertIn('avg_recommendations_per_user', stats)


if __name__ == '__main__':
    unittest.main()
