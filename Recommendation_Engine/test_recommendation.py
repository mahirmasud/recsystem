"""
Test Recommendation Engine

Unit tests for:
- Model registry
- Trainer
- Inference
- Recommendation manager
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelRegistry(unittest.TestCase):
    """Tests for ModelRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.test_dir) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        from Recommendation_Engine.model_registry import ModelRegistry
        self.registry = ModelRegistry(str(self.models_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        self.assertTrue(self.registry.models_dir.exists())
        self.assertTrue(self.registry.pytorch_dir.exists())
        self.assertTrue(self.registry.xgboost_dir.exists())
        self.assertTrue(self.registry.embeddings_dir.exists())
    
    def test_register_and_list_model(self):
        """Test model registration and listing."""
        # Create a simple mock model (numpy array)
        mock_model = np.array([1, 2, 3])
        
        model_id = self.registry.register_model(
            model=mock_model,
            model_type='sklearn',
            description="Test model"
        )
        
        self.assertIsNotNone(model_id)
        self.assertTrue(model_id.startswith('sklearn_'))
        
        # List models
        models = self.registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['model_id'], model_id)
    
    def test_save_and_load_embeddings(self):
        """Test embedding save and load."""
        embeddings = np.random.randn(100, 64)
        ids = [f"item_{i}" for i in range(100)]
        
        path = self.registry.save_embeddings(
            embeddings=embeddings,
            embedding_type='item',
            ids=ids
        )
        
        self.assertTrue(path.exists())
        
        # Load embeddings
        loaded_emb, loaded_ids = self.registry.load_embeddings('item')
        
        self.assertEqual(loaded_emb.shape, embeddings.shape)
        self.assertEqual(len(loaded_ids), len(ids))


class TestRecommendationTrainer(unittest.TestCase):
    """Tests for RecommendationTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.test_dir) / "models"
        
        from Recommendation_Engine.trainer import RecommendationTrainer
        self.trainer = RecommendationTrainer(str(self.models_dir), device='cpu')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Create sample data
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 11, 10, 12, 11],
            'rating': [5, 4, 3, 5, 4]
        })
        
        user_features = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'total_purchases': [10, 20, 15]
        })
        
        item_features = pd.DataFrame({
            'item_id': [10, 11, 12],
            'price': [100, 200, 150],
            'popularity': [0.8, 0.9, 0.7]
        })
        
        feature_columns = {
            'user': ['age', 'total_purchases'],
            'item': ['price', 'popularity']
        }
        
        user_feat, item_feat, labels = self.trainer.prepare_training_data(
            interactions_df=interactions,
            user_features_df=user_features,
            item_features_df=item_features,
            feature_columns=feature_columns
        )
        
        self.assertEqual(len(user_feat), 5)
        self.assertEqual(user_feat.shape[1], 2)
        self.assertEqual(len(labels), 5)


class TestRecommendationInference(unittest.TestCase):
    """Tests for RecommendationInference class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.test_dir) / "models"
        
        from Recommendation_Engine.inference import RecommendationInference
        self.inference = RecommendationInference(str(self.models_dir), device='cpu')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_retrieve_candidates_without_embeddings(self):
        """Test candidate retrieval with no embeddings."""
        user_embedding = np.random.randn(64)
        
        candidates = self.inference.retrieve_candidates(
            user_embedding=user_embedding,
            top_k=10
        )
        
        # Should return empty list when no embeddings available
        self.assertEqual(len(candidates), 0)
    
    def test_reranking(self):
        """Test re-ranking application."""
        ranked_candidates = [
            ("item_1", 0.9),
            ("item_2", 0.8),
            ("item_3", 0.7),
            ("item_4", 0.6),
            ("item_5", 0.5)
        ]
        
        recommendations = self.inference.apply_reranking(
            ranked_candidates=ranked_candidates,
            user_id="user_1",
            config={}
        )
        
        self.assertEqual(len(recommendations), 5)
        self.assertIn('item_id', recommendations[0])
        self.assertIn('ranking_score', recommendations[0])


class TestRecommendationManager(unittest.TestCase):
    """Tests for RecommendationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal config
        config_path = self.output_dir / "rec_config.json"
        with open(config_path, 'w') as f:
            f.write('{}')
        
        from Recommendation_Engine.recommendation_manager import RecommendationManager
        self.manager = RecommendationManager(
            config_path=str(config_path),
            models_dir=str(self.output_dir / "models"),
            features_dir=str(self.output_dir / "features")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        self.assertIsNotNone(self.manager.registry)
        self.assertIsNotNone(self.manager.trainer)
        self.assertIsNotNone(self.manager.inference)
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = self.manager.list_available_models()
        self.assertIsInstance(models, list)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationInference))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
