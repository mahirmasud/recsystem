"""
Feature Selector - Selects optimal features for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects optimal features for machine learning models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_features = {}
        self.feature_importance = {}
        logger.info("FeatureSelector initialized")
    
    def select_by_variance(self, features_df: pd.DataFrame, 
                           threshold: float = 0.01,
                           exclude_cols: Optional[List[str]] = None) -> List[str]:
        """Select features based on variance threshold."""
        logger.info(f"Selecting features by variance (threshold={threshold})")
        
        df = features_df.copy()
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        
        # Only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)
        
        selected = numeric_df.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected)} features by variance")
        
        return selected
    
    def select_by_importance(self, features_df: pd.DataFrame, 
                             target_col: str,
                             n_features: int = 50,
                             model_type: str = 'random_forest') -> List[str]:
        """Select features based on importance scores."""
        logger.info(f"Selecting top {n_features} features by importance")
        
        df = features_df.copy()
        
        # Handle missing target
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return []
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ['int64', 'float64']]
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            is_classification = True
        else:
            is_classification = y.nunique() < 20
        
        # Calculate importance
        if model_type == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            importances = model.feature_importances_
        else:
            # Use mutual information
            if is_classification:
                importances = mutual_info_classif(X, y, random_state=42)
            else:
                importances = mutual_info_regression(X, y, random_state=42)
        
        # Get top features
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(n_features)['feature'].tolist()
        self.feature_importance = importance_df.set_index('feature')['importance'].to_dict()
        
        logger.info(f"Selected {len(top_features)} features by importance")
        
        return top_features
    
    def select_by_correlation(self, features_df: pd.DataFrame,
                              threshold: float = 0.95,
                              exclude_cols: Optional[List[str]] = None) -> List[str]:
        """Remove highly correlated features."""
        logger.info(f"Removing correlated features (threshold={threshold})")
        
        df = features_df.copy()
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        
        # Only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Mark features to drop
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        
        selected = [c for c in numeric_df.columns if c not in to_drop]
        logger.info(f"Removed {len(to_drop)} correlated features, kept {len(selected)}")
        
        return selected
    
    def get_selected_features(self, entity_type: str) -> List[str]:
        """Get previously selected features for an entity type."""
        return self.selected_features.get(entity_type, [])
    
    def store_selection(self, entity_type: str, features: List[str]) -> None:
        """Store selected features for an entity type."""
        self.selected_features[entity_type] = features
        logger.info(f"Stored {len(features)} selected features for {entity_type}")
