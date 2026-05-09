"""
Base Ranker - Abstract base class for all ranking models

Provides common interface for DLRM, XGBoost, LightGBM rankers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseRanker(ABC):
    """
    Abstract base class for recommendation ranking models.
    
    All ranking models must implement:
    - fit(): Train the ranking model
    - predict(): Generate relevance scores
    - save(): Persist model artifacts
    - load(): Restore model from disk
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base ranker.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self.feature_columns = []
        self.target_column = config.get('target_column', 'label')
        self.is_trained = False
        
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Train the ranking model.
        
        Args:
            train_df: Training dataframe with features and labels
            val_df: Optional validation dataframe
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate relevance scores for candidate items.
        
        Args:
            features: Feature dataframe for scoring
            
        Returns:
            Array of relevance scores
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass
    
    def rank(self, candidates: pd.DataFrame, top_k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Rank candidates by predicted relevance.
        
        Args:
            candidates: Candidate items with features
            top_k: Number of top items to return
            
        Returns:
            Tuple of (ranked dataframe, scores)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before ranking")
        
        scores = self.predict(candidates[self.feature_columns])
        candidates = candidates.copy()
        candidates['ranking_score'] = scores
        
        # Sort by score descending
        ranked = candidates.sort_values('ranking_score', ascending=False)
        
        return ranked.head(top_k), scores
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance values
        """
        logger.warning("Feature importance not implemented for this ranker")
        return None
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that required features are present.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all features present
        """
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing features: {missing}")
            return False
        return True
