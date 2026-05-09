"""
Recommendation Manager

Orchestrates the full recommendation pipeline:
- Model loading and management
- Feature preparation
- Candidate generation
- Ranking and re-ranking
- Result formatting and export
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from shared.logger import get_logger
from shared.config import Config
from .model_registry import ModelRegistry
from .trainer import RecommendationTrainer
from .inference import RecommendationInference

logger = get_logger(__name__)


class RecommendationManager:
    """
    High-level manager for recommendation operations.
    
    Provides simplified interface for:
    - Training models
    - Generating recommendations
    - Managing model lifecycle
    - Exporting results
    """
    
    def __init__(
        self,
        config_path: str = "output/rec_config.json",
        models_dir: str = "output/models",
        features_dir: str = "output/features",
        recommendations_dir: str = "output/recommendations"
    ):
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        self.recommendations_dir = Path(recommendations_dir)
        
        # Ensure directories exist
        for dir_path in [self.models_dir, self.features_dir, self.recommendations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.registry = ModelRegistry(str(self.models_dir))
        self.trainer = RecommendationTrainer(str(self.models_dir))
        self.inference = RecommendationInference(str(self.models_dir))
        
        # Loaded data
        self.user_features_df = None
        self.item_features_df = None
        self.interactions_df = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def load_data(
        self,
        user_features_path: Optional[str] = None,
        item_features_path: Optional[str] = None,
        interactions_path: Optional[str] = None
    ):
        """
        Load feature and interaction data.
        
        Args:
            user_features_path: Path to user features parquet
            item_features_path: Path to item features parquet
            interactions_path: Path to interactions parquet
        """
        # Default paths
        if user_features_path is None:
            user_features_path = self.features_dir / "user_features.parquet"
        if item_features_path is None:
            item_features_path = self.features_dir / "item_features.parquet"
        if interactions_path is None:
            interactions_path = self.features_dir / "interaction_features.parquet"
        
        # Load user features
        if Path(user_features_path).exists():
            self.user_features_df = pd.read_parquet(user_features_path)
            logger.info(f"Loaded user features: {self.user_features_df.shape}")
        else:
            logger.warning(f"User features not found: {user_features_path}")
        
        # Load item features
        if Path(item_features_path).exists():
            self.item_features_df = pd.read_parquet(item_features_path)
            logger.info(f"Loaded item features: {self.item_features_df.shape}")
        else:
            logger.warning(f"Item features not found: {item_features_path}")
        
        # Load interactions
        if Path(interactions_path).exists():
            self.interactions_df = pd.read_parquet(interactions_path)
            logger.info(f"Loaded interactions: {self.interactions_df.shape}")
        else:
            logger.warning(f"Interactions not found: {interactions_path}")
    
    def train_models(
        self,
        model_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Train recommendation models.
        
        Args:
            model_types: List of model types to train
                         ['three_tower', 'dlrm', 'xgboost', 'lightgbm']
            **kwargs: Additional training parameters
            
        Returns:
            Dict mapping model type to model ID
        """
        if self.user_features_df is None or self.item_features_df is None:
            raise ValueError("Must load data before training")
        
        model_types = model_types or ['xgboost', 'lightgbm']
        trained_models = {}
        
        # Prepare training data
        feature_columns = self.config.get('feature_columns', {
            'user': ['age', 'gender_encoded', 'total_purchases', 'avg_order_value'],
            'item': ['price', 'category_encoded', 'popularity_score']
        })
        
        user_features, item_features, labels = self.trainer.prepare_training_data(
            interactions_df=self.interactions_df,
            user_features_df=self.user_features_df,
            item_features_df=self.item_features_df,
            feature_columns=feature_columns
        )
        
        # Train requested models
        for model_type in model_types:
            try:
                if model_type == 'three_tower':
                    model_id = self.trainer.train_three_tower(
                        user_features=user_features,
                        item_features=item_features,
                        labels=labels,
                        **kwargs
                    )
                    trained_models['three_tower'] = model_id
                    
                elif model_type == 'dlrm':
                    model_id = self.trainer.train_dlrm(
                        user_features=user_features,
                        item_features=item_features,
                        labels=labels,
                        **kwargs
                    )
                    trained_models['dlrm'] = model_id
                    
                elif model_type == 'xgboost':
                    model_id = self.trainer.train_xgboost(
                        user_features=user_features,
                        item_features=item_features,
                        labels=labels,
                        **kwargs
                    )
                    trained_models['xgboost'] = model_id
                    
                elif model_type == 'lightgbm':
                    model_id = self.trainer.train_lightgbm(
                        user_features=user_features,
                        item_features=item_features,
                        labels=labels,
                        **kwargs
                    )
                    trained_models['lightgbm'] = model_id
                    
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
        
        # Set active models
        for model_type, model_id in trained_models.items():
            self.registry.set_active_model(model_type, model_id)
        
        logger.info(f"Trained models: {list(trained_models.keys())}")
        return trained_models
    
    def generate_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_items: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a single user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_items: Items to exclude
            config: Re-ranking configuration
            
        Returns:
            List of recommendation dictionaries
        """
        if self.user_features_df is None or self.item_features_df is None:
            raise ValueError("Must load data before generating recommendations")
        
        # Get user features
        user_row = self.user_features_df[self.user_features_df['user_id'] == user_id]
        if user_row.empty:
            logger.warning(f"User {user_id} not found")
            return []
        
        user_features = user_row.drop(columns=['user_id']).values[0]
        
        # Build item features dict
        item_features_dict = {}
        for _, row in self.item_features_df.iterrows():
            item_id = row['item_id']
            features = row.drop(columns=['item_id']).values
            item_features_dict[item_id] = features
        
        # Load models if needed
        if self.inference.item_embeddings is None:
            self.inference.load_models()
        
        # Generate recommendations
        recommendations = self.inference.recommend(
            user_id=user_id,
            user_features=user_features,
            item_features_dict=item_features_dict,
            exclude_items=exclude_items,
            top_k=top_k,
            config=config
        )
        
        return recommendations
    
    def batch_recommendations(
        self,
        user_ids: Optional[List[str]] = None,
        top_k: int = 10,
        output_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs (uses all if None)
            top_k: Number of recommendations per user
            output_path: Path to save results
            config: Re-ranking configuration
            
        Returns:
            DataFrame with all recommendations
        """
        if self.user_features_df is None or self.item_features_df is None:
            raise ValueError("Must load data before generating recommendations")
        
        # Filter users if specified
        users_df = self.user_features_df
        if user_ids:
            users_df = users_df[users_df['user_id'].isin(user_ids)]
        
        # Build item features dict
        item_features_dict = {}
        for _, row in self.item_features_df.iterrows():
            item_id = row['item_id']
            features = row.drop(columns=['item_id']).values
            item_features_dict[item_id] = features
        
        # Load models if needed
        if self.inference.item_embeddings is None:
            self.inference.load_models()
        
        # Generate batch recommendations
        recommendations_df = self.inference.batch_recommend(
            users_df=users_df,
            item_features_dict=item_features_dict,
            top_k=top_k,
            config=config
        )
        
        # Save results
        if output_path:
            self.inference.export_recommendations(
                recommendations=recommendations_df.to_dict('records'),
                output_path=output_path,
                format='parquet'
            )
        
        logger.info(f"Generated {len(recommendations_df)} recommendations")
        return recommendations_df
    
    def evaluate_models(
        self,
        test_interactions: Optional[pd.DataFrame] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_interactions: Test set interactions
            metrics: List of metrics to compute
            
        Returns:
            Dict of metric name to value
        """
        from .Evaluation.evaluator import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        # Use provided test data or split from loaded data
        if test_interactions is None:
            if self.interactions_df is None:
                raise ValueError("No test data available")
            # Simple train/test split
            test_interactions = self.interactions_df.sample(frac=0.2)
        
        # Compute metrics
        results = evaluator.evaluate(
            predictions=[],  # Would need actual predictions
            ground_truth=test_interactions,
            metrics=metrics or ['precision', 'recall', 'ndcg', 'map']
        )
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def export_model_report(self, output_path: str):
        """Export comprehensive model report."""
        report = {
            'exported_at': datetime.now().isoformat(),
            'models': self.registry.list_models(),
            'active_models': self.registry.active_models,
            'training_history': self.trainer.training_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported model report to {output_path}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return self.registry.list_models()
    
    def get_active_model(self, model_type: str) -> Optional[str]:
        """Get active model ID for a type."""
        return self.registry.get_active_model(model_type)
    
    def set_active_model(self, model_type: str, model_id: str):
        """Set active model for a type."""
        self.registry.set_active_model(model_type, model_id)
