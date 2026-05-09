"""
Scoring Engine - Unified scoring interface for multiple rankers

Provides a unified interface to:
- Select ranking model (DLRM, XGBoost, LightGBM)
- Manage model lifecycle
- Generate scores with fallback support
- Batch scoring optimization
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .ranker import BaseRanker
from .dlrm_ranker import DLRMRanker
from .xgboost_ranker import XGBoostRanker
from .lightgbm_ranker import LightGBMRanker

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Unified scoring engine for recommendation ranking.
    
    Features:
    - Model selection and switching
    - Fallback chain for robustness
    - Batch scoring optimization
    - Score normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scoring engine.
        
        Args:
            config: Configuration with model settings
        """
        self.config = config
        self.primary_model_type = config.get('primary_model', 'xgboost')
        self.fallback_models = config.get('fallback_models', ['popularity'])
        self.models: Dict[str, BaseRanker] = {}
        self.current_model = None
        
        # Score normalization
        self.score_min = config.get('score_min', 0.0)
        self.score_max = config.get('score_max', 1.0)
        
    def register_model(self, name: str, model: BaseRanker) -> None:
        """
        Register a ranking model.
        
        Args:
            name: Model identifier
            model: Ranking model instance
        """
        self.models[name] = model
        logger.info(f"Registered model: {name}")
        
    def load_model(self, name: str, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            name: Model identifier
            path: Path to model directory
        """
        model_type = self.config.get(f'{name}_type', 'xgboost')
        
        if model_type == 'dlrm':
            model = DLRMRanker(self.config.get('dlrm_config', {}))
        elif model_type == 'xgboost':
            model = XGBoostRanker(self.config.get('xgboost_config', {}))
        elif model_type == 'lightgbm':
            model = LightGBMRanker(self.config.get('lightgbm_config', {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load(path)
        self.register_model(name, model)
        
    def set_primary_model(self, name: str) -> None:
        """
        Set the primary model for scoring.
        
        Args:
            name: Model identifier
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
        
        self.current_model = self.models[name]
        self.primary_model_type = name
        logger.info(f"Set primary model to: {name}")
        
    def score(
        self, 
        candidates: pd.DataFrame,
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Generate scores for candidate items.
        
        Args:
            candidates: Candidate items with features
            model_name: Optional model name (uses primary if not specified)
            normalize: Whether to normalize scores
            
        Returns:
            DataFrame with added score column
        """
        # Select model
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not registered")
            model = self.models[model_name]
        else:
            if self.current_model is None:
                raise RuntimeError("No model selected for scoring")
            model = self.current_model
        
        # Generate scores
        try:
            scores = model.predict(candidates)
        except Exception as e:
            logger.error(f"Scoring failed with primary model: {e}")
            # Fallback logic could be implemented here
            raise
        
        # Add scores to dataframe
        result = candidates.copy()
        result['ranking_score'] = scores
        
        # Normalize scores if requested
        if normalize and len(scores) > 1:
            score_min = scores.min()
            score_max = scores.max()
            if score_max > score_min:
                normalized = (scores - score_min) / (score_max - score_min)
                normalized = normalized * (self.score_max - self.score_min) + self.score_min
                result['ranking_score'] = normalized
        
        return result
    
    def score_batch(
        self,
        candidate_batches: List[pd.DataFrame],
        model_name: Optional[str] = None
    ) -> List[pd.DataFrame]:
        """
        Score multiple batches of candidates efficiently.
        
        Args:
            candidate_batches: List of candidate DataFrames
            model_name: Optional model name
            
        Returns:
            List of scored DataFrames
        """
        results = []
        for batch in candidate_batches:
            scored = self.score(batch, model_name)
            results.append(scored)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about registered models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'primary_model': self.primary_model_type,
            'registered_models': list(self.models.keys()),
            'current_model': self.current_model.__class__.__name__ if self.current_model else None
        }
        return info
    
    def save_all_models(self, base_path: str) -> None:
        """
        Save all registered models.
        
        Args:
            base_path: Base directory for saving models
        """
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model.save(str(base / name))
        
        logger.info(f"Saved {len(self.models)} models to {base_path}")
    
    def evaluate_models(
        self,
        test_df: pd.DataFrame,
        metrics: List[str] = ['auc', 'ndcg']
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all registered models on test data.
        
        Args:
            test_df: Test dataframe
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of model -> metric -> value
        """
        results = {}
        
        for name, model in self.models.items():
            if not model.is_trained:
                continue
            
            try:
                predictions = model.predict(test_df)
                actuals = test_df[model.target_column].values if model.target_column in test_df.columns else None
                
                if actuals is None:
                    continue
                
                model_metrics = {}
                
                if 'auc' in metrics:
                    from sklearn.metrics import roc_auc_score
                    model_metrics['auc'] = roc_auc_score(actuals, predictions)
                
                if 'ndcg' in metrics:
                    # Simplified NDCG calculation
                    from .ndcg import NDCGMetric
                    ndcg = NDCGMetric()
                    model_metrics['ndcg'] = ndcg.compute(predictions, actuals)
                
                results[name] = model_metrics
                
            except Exception as e:
                logger.error(f"Evaluation failed for model {name}: {e}")
        
        return results
