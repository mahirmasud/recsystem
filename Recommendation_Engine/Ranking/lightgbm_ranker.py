"""
LightGBM Ranker - Gradient Boosting with Leaf-wise Growth

Implements LightGBM-based ranking with:
- Efficient leaf-wise tree growth
- Learning-to-rank objectives
- Fast training on large datasets
- Native categorical feature support
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path

from .ranker import BaseRanker

logger = logging.getLogger(__name__)


class LightGBMRanker(BaseRanker):
    """
    LightGBM-based ranking model for personalized recommendations.
    
    Features:
    - LambdaMART and LambdaRank objectives
    - Leaf-wise tree growth for better accuracy
    - Efficient handling of large datasets
    - Native categorical feature support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM ranker.
        
        Args:
            config: Configuration with LightGBM hyperparameters
        """
        super().__init__(config)
        
        # LightGBM hyperparameters
        self.n_estimators = config.get('n_estimators', 100)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.num_leaves = config.get('num_leaves', 31)
        self.max_depth = config.get('max_depth', -1)
        self.min_child_samples = config.get('min_child_samples', 20)
        self.subsample = config.get('subsample', 1.0)
        self.colsample_bytree = config.get('colsample_bytree', 1.0)
        self.reg_alpha = config.get('reg_alpha', 0.0)
        self.reg_lambda = config.get('reg_lambda', 1.0)
        self.objective = config.get('objective', 'lambdarank')
        self.metric = config.get('metric', 'ndcg')
        
        # Training parameters
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
        self.verbose_eval = config.get('verbose_eval', 10)
        self.n_jobs = config.get('n_jobs', -1)
        
        # Categorical features
        self.categorical_features = config.get('categorical_features', [])
        
        # Model instance
        self.model = None
        self.feature_importance_df = None
        
    def fit(
        self, 
        train_df: pd.DataFrame, 
        val_df: Optional[pd.DataFrame] = None,
        qid_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the LightGBM ranking model.
        
        Args:
            train_df: Training dataframe
            val_df: Optional validation dataframe
            qid_column: Query/group ID column for ranking
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training LightGBM ranker on {len(train_df)} samples")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        # Auto-detect feature columns if not set
        if not self.feature_columns:
            exclude_cols = [self.target_column, qid_column] if qid_column else [self.target_column]
            exclude_cols.extend(self.categorical_features)
            self.feature_columns = [c for c in train_df.columns if c not in exclude_cols]
            # Add categorical features that are not in exclude_cols
            for cat_feat in self.categorical_features:
                if cat_feat in train_df.columns and cat_feat not in self.feature_columns:
                    self.feature_columns.append(cat_feat)
        
        # Prepare training data
        X_train = train_df[self.feature_columns].copy()
        y_train = train_df[self.target_column].values if self.target_column in train_df.columns else np.zeros(len(train_df))
        
        # Handle categorical features
        cat_indices = []
        for i, col in enumerate(self.feature_columns):
            if col in self.categorical_features:
                cat_indices.append(i)
                X_train[col] = X_train[col].astype('category').cat.codes
        
        # Create dataset
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_indices if cat_indices else 'auto',
            feature_name=self.feature_columns
        )
        
        # Set query groups for ranking
        if qid_column and qid_column in train_df.columns:
            qid = train_df[qid_column].values
            _, group_sizes = np.unique(qid, return_counts=True)
            train_data.set_group(group_sizes)
        
        # Validation data
        evals = [(train_data, 'train')]
        if val_df is not None:
            X_val = val_df[self.feature_columns].copy()
            y_val = val_df[self.target_column].values if self.target_column in val_df.columns else np.zeros(len(val_df))
            
            # Handle categorical features
            for col in self.categorical_features:
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype('category').cat.codes
            
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=cat_indices if cat_indices else 'auto',
                feature_name=self.feature_columns,
                reference=train_data
            )
            
            if qid_column and qid_column in val_df.columns:
                qid = val_df[qid_column].values
                _, group_sizes = np.unique(qid, return_counts=True)
                val_data.set_group(group_sizes)
            
            evals.append((val_data, 'val'))
        
        # LightGBM parameters
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'n_jobs': self.n_jobs,
            'verbosity': -1
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval
        )
        
        self.is_trained = True
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Get best iteration
        best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        
        metrics = {
            'best_iteration': best_iteration,
            'num_features': len(self.feature_columns)
        }
        
        logger.info(f"LightGBM training completed. Best iteration: {best_iteration}")
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
        X = features.copy()
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            logger.warning(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                X[col] = 0
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category').cat.codes
        
        scores = self.model.predict(X[self.feature_columns])
        return scores
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if self.model is None:
            return
        
        importance = self.model.feature_importance(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        })
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
        self.model.save_model(str(save_path / 'lightgbm_ranker.txt'))
        
        # Save configuration
        config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'categorical_features': self.categorical_features,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
        }
        joblib.dump(config, save_path / 'lightgbm_config.pkl')
        
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        logger.info(f"LightGBM model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        load_path = Path(path)
        
        # Load configuration
        config = joblib.load(load_path / 'lightgbm_config.pkl')
        self.feature_columns = config['feature_columns']
        self.target_column = config['target_column']
        self.categorical_features = config['categorical_features']
        self.n_estimators = config['n_estimators']
        self.learning_rate = config['learning_rate']
        self.num_leaves = config['num_leaves']
        self.max_depth = config['max_depth']
        
        # Load model
        self.model = lgb.Booster(model_file=str(load_path / 'lightgbm_ranker.txt'))
        
        # Load feature importance if available
        importance_path = load_path / 'feature_importance.csv'
        if importance_path.exists():
            self.feature_importance_df = pd.read_csv(importance_path)
        
        self.is_trained = True
        logger.info(f"LightGBM model loaded from {path}")
