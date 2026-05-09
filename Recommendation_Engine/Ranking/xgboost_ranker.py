"""
XGBoost Ranker - Gradient Boosting for Recommendation Ranking

Implements XGBoost-based ranking with:
- Learning-to-rank objectives (LambdaMART)
- Feature importance analysis
- Fast training and inference
- Robust handling of missing values
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from .ranker import BaseRanker

logger = logging.getLogger(__name__)


class XGBoostRanker(BaseRanker):
    """
    XGBoost-based ranking model for personalized recommendations.
    
    Features:
    - LambdaMART objective for learning-to-rank
    - Handles large feature sets efficiently
    - Provides feature importance scores
    - Supports group-wise ranking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost ranker.
        
        Args:
            config: Configuration with XGBoost hyperparameters
        """
        super().__init__(config)
        
        # XGBoost hyperparameters
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.min_child_weight = config.get('min_child_weight', 1)
        self.gamma = config.get('gamma', 0)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.objective = config.get('objective', 'rank:pairwise')
        self.eval_metric = config.get('eval_metric', 'ndcg')
        
        # Training parameters
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
        self.num_boost_round = config.get('num_boost_round', 1000)
        self.verbose_eval = config.get('verbose_eval', 10)
        
        # Model instance
        self.model = None
        self.feature_importance_df = None
        
    def _create_dmatrix(
        self, 
        df: pd.DataFrame, 
        qid_column: Optional[str] = None
    ) -> xgb.DMatrix:
        """
        Create DMatrix for XGBoost.
        
        Args:
            df: Input dataframe
            qid_column: Query/group ID column for ranking
            
        Returns:
            XGBoost DMatrix
        """
        features = df[self.feature_columns].fillna(-1)
        
        if self.target_column in df.columns:
            labels = df[self.target_column].values
        else:
            labels = np.zeros(len(df))
        
        dmatrix = xgb.DMatrix(features, label=labels)
        
        # Set query groups for ranking
        if qid_column and qid_column in df.columns:
            qid = df[qid_column].values
            # Calculate group sizes
            _, group_sizes = np.unique(qid, return_counts=True)
            dmatrix.set_group(group_sizes)
        
        return dmatrix
    
    def fit(
        self, 
        train_df: pd.DataFrame, 
        val_df: Optional[pd.DataFrame] = None,
        qid_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the XGBoost ranking model.
        
        Args:
            train_df: Training dataframe
            val_df: Optional validation dataframe
            qid_column: Query/group ID column for ranking
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training XGBoost ranker on {len(train_df)} samples")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        # Auto-detect feature columns if not set
        if not self.feature_columns:
            exclude_cols = [self.target_column, qid_column] if qid_column else [self.target_column]
            self.feature_columns = [c for c in train_df.columns if c not in exclude_cols]
        
        # Prepare training data
        dtrain = self._create_dmatrix(train_df, qid_column)
        
        # XGBoost parameters
        params = {
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'verbosity': 1
        }
        
        # Training callbacks
        evals = [(dtrain, 'train')]
        if val_df is not None:
            dval = self._create_dmatrix(val_df, qid_column)
            evals.append((dval, 'val'))
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval
        )
        
        self.is_trained = True
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Get best iteration score
        best_score = self.model.best_score if hasattr(self.model, 'best_score') else 0.0
        best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        
        metrics = {
            'best_score': best_score,
            'best_iteration': best_iteration,
            'num_features': len(self.feature_columns)
        }
        
        logger.info(f"XGBoost training completed. Best score: {best_score:.4f}")
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate relevance scores.
        
        Args:
            features: Feature dataframe
            
        Returns:
            Array of relevance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_columns) - set(features.columns)
        if missing_cols:
            logger.warning(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                features[col] = 0
        
        dmatrix = xgb.DMatrix(features[self.feature_columns].fillna(-1))
        scores = self.model.predict(dmatrix)
        
        return scores
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if self.model is None:
            return
        
        importance_dict = self.model.get_score(importance_type='gain')
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.feature_importance_df = importance_df
        logger.info(f"Top 5 features: {importance_df.head(5)['feature'].tolist()}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance values
        """
        if self.feature_importance_df is None:
            self._calculate_feature_importance()
        return self.feature_importance_df
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(save_path / 'xgboost_ranker.json'))
        
        # Save configuration
        config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsamples_bytree': self.colsample_bytree,
        }
        joblib.dump(config, save_path / 'xgboost_config.pkl')
        
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        load_path = Path(path)
        
        # Load configuration
        config = joblib.load(load_path / 'xgboost_config.pkl')
        self.feature_columns = config['feature_columns']
        self.target_column = config['target_column']
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.learning_rate = config['learning_rate']
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(load_path / 'xgboost_ranker.json'))
        
        # Load feature importance if available
        importance_path = load_path / 'feature_importance.csv'
        if importance_path.exists():
            self.feature_importance_df = pd.read_csv(importance_path)
        
        self.is_trained = True
        logger.info(f"XGBoost model loaded from {path}")
