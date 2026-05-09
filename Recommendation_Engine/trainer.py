"""
Recommendation Model Trainer

Training pipelines for:
- Three-Tower retrieval model
- DLRM ranking model
- XGBoost/LightGBM fallback models
- Embedding generation
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from shared.logger import get_logger
from .model_registry import ModelRegistry

logger = get_logger(__name__)


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation training."""
    
    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        labels: np.ndarray,
        user_ids: Optional[np.ndarray] = None,
        item_ids: Optional[np.ndarray] = None
    ):
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
        self.labels = torch.FloatTensor(labels)
        self.user_ids = user_ids
        self.item_ids = item_ids
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'user_features': self.user_features[idx],
            'item_features': self.item_features[idx],
            'label': self.labels[idx]
        }
        if self.user_ids is not None:
            sample['user_id'] = self.user_ids[idx]
        if self.item_ids is not None:
            sample['item_id'] = self.item_ids[idx]
        return sample


class RecommendationTrainer:
    """
    Trainer for recommendation models.
    
    Supports:
    - Three-Tower model training
    - DLRM training
    - XGBoost/LightGBM training
    - Embedding generation
    """
    
    def __init__(
        self,
        models_dir: str = "output/models",
        device: str = "cpu"
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.registry = ModelRegistry(models_dir)
        self.training_history: List[Dict[str, Any]] = []
    
    def prepare_training_data(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        feature_columns: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from standardized datasets.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            user_features_df: DataFrame with user features
            item_features_df: DataFrame with item features
            feature_columns: Dict specifying which columns to use
            
        Returns:
            Tuple of (user_features, item_features, labels)
        """
        logger.info("Preparing training data...")
        
        # Merge interactions with user features
        merged = interactions_df.merge(
            user_features_df,
            on='user_id',
            how='left'
        )
        
        # Merge with item features
        merged = merged.merge(
            item_features_df,
            on='item_id',
            how='left'
        )
        
        # Extract feature matrices
        user_feature_cols = feature_columns.get('user', [])
        item_feature_cols = feature_columns.get('item', [])
        
        user_features = merged[user_feature_cols].fillna(0).values
        item_features = merged[item_feature_cols].fillna(0).values
        
        # Create labels (1 for positive interaction, 0 for negative)
        if 'label' in merged.columns:
            labels = merged['label'].values
        elif 'rating' in merged.columns:
            labels = (merged['rating'] > 3).astype(float).values
        else:
            # Assume all interactions are positive
            labels = np.ones(len(merged))
        
        logger.info(f"Prepared {len(labels)} training samples")
        return user_features, item_features, labels
    
    def train_three_tower(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        labels: np.ndarray,
        user_ids: Optional[np.ndarray] = None,
        item_ids: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        embedding_dim: int = 64
    ) -> str:
        """
        Train Three-Tower retrieval model.
        
        Args:
            user_features: User feature matrix
            item_features: Item feature matrix
            labels: Interaction labels
            user_ids: Optional user IDs
            item_ids: Optional item IDs
            config: Model configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            embedding_dim: Embedding dimension
            
        Returns:
            model_id: Registered model ID
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Three-Tower training")
        
        logger.info("Training Three-Tower model...")
        
        from .Candidate_Generation.three_tower_model import ThreeTowerModel
        
        # Get unique counts
        n_users = len(np.unique(user_ids)) if user_ids is not None else len(user_features)
        n_items = len(np.unique(item_ids)) if item_ids is not None else len(item_features)
        
        user_feature_dim = user_features.shape[1]
        item_feature_dim = item_features.shape[1]
        
        # Initialize model
        model = ThreeTowerModel(
            n_users=n_users,
            n_items=n_items,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            context_feature_dim=10,  # Default context dim
            embedding_dim=embedding_dim
        ).to(self.device)
        
        # Prepare dataset
        dataset = RecommendationDataset(
            user_features=user_features,
            item_features=item_features,
            labels=labels,
            user_ids=user_ids,
            item_ids=item_ids
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                user_feat = batch['user_features'].to(self.device)
                item_feat = batch['item_features'].to(self.device)
                
                # Simple context (could be extended)
                context_feat = torch.zeros(user_feat.size(0), 10).to(self.device)
                
                labels_batch = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                user_emb, item_emb, _ = model(user_feat, item_feat, context_feat)
                
                # Compute similarity score
                scores = (user_emb * item_emb).sum(dim=1)
                
                loss = criterion(scores.unsqueeze(1), labels_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            history.append({'epoch': epoch + 1, 'loss': avg_loss})
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Register model
        metrics = {'final_loss': history[-1]['loss']}
        hyperparameters = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'embedding_dim': embedding_dim
        }
        
        model_id = self.registry.register_model(
            model=model,
            model_type='three_tower',
            metrics=metrics,
            hyperparameters=hyperparameters,
            description="Three-Tower retrieval model"
        )
        
        # Save embeddings
        self._save_embeddings(model, user_features, item_features, model_id)
        
        self.training_history.append({
            'model_id': model_id,
            'model_type': 'three_tower',
            'timestamp': datetime.now().isoformat(),
            'history': history
        })
        
        logger.info(f"Three-Tower model trained and registered: {model_id}")
        return model_id
    
    def _save_embeddings(
        self,
        model: nn.Module,
        user_features: np.ndarray,
        item_features: np.ndarray,
        model_id: str
    ):
        """Generate and save embeddings from trained model."""
        logger.info("Generating embeddings...")
        
        model.eval()
        with torch.no_grad():
            user_tensor = torch.FloatTensor(user_features).to(self.device)
            item_tensor = torch.FloatTensor(item_features).to(self.device)
            context_tensor = torch.zeros(user_tensor.size(0), 10).to(self.device)
            
            user_emb, item_emb, _ = model(user_tensor, item_tensor, context_tensor)
            
            user_embeddings = user_emb.cpu().numpy()
            item_embeddings = item_emb.cpu().numpy()
        
        # Save embeddings
        user_ids = [f"user_{i}" for i in range(len(user_embeddings))]
        item_ids = [f"item_{i}" for i in range(len(item_embeddings))]
        
        self.registry.save_embeddings(
            user_embeddings, 'user', user_ids, model_id
        )
        self.registry.save_embeddings(
            item_embeddings, 'item', item_ids, model_id
        )
    
    def train_dlrm(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        labels: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ) -> str:
        """
        Train DLRM ranking model.
        
        Args:
            user_features: User feature matrix
            item_features: Item feature matrix
            labels: Interaction labels
            config: Model configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            model_id: Registered model ID
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DLRM training")
        
        logger.info("Training DLRM model...")
        
        from .Ranking.dlrm_ranker import DLRMRanker
        
        feature_dim = user_features.shape[1] + item_features.shape[1]
        
        # Initialize model
        model = DLRMRanker(
            dense_feature_dim=feature_dim,
            sparse_feature_dims=[],  # Can be extended for categorical features
            embedding_dim=64,
            hidden_dims=[256, 128, 64]
        ).to(self.device)
        
        # Prepare dataset
        combined_features = np.hstack([user_features, item_features])
        dataset = RecommendationDataset(
            user_features=combined_features,
            item_features=np.zeros_like(combined_features),  # Not used in DLRM
            labels=labels
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                features = batch['user_features'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                scores = model(features)
                
                loss = criterion(scores, labels_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            history.append({'epoch': epoch + 1, 'loss': avg_loss})
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Register model
        metrics = {'final_loss': history[-1]['loss']}
        hyperparameters = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'feature_dim': feature_dim
        }
        
        model_id = self.registry.register_model(
            model=model,
            model_type='dlrm',
            metrics=metrics,
            hyperparameters=hyperparameters,
            description="DLRM ranking model"
        )
        
        self.training_history.append({
            'model_id': model_id,
            'model_type': 'dlrm',
            'timestamp': datetime.now().isoformat(),
            'history': history
        })
        
        logger.info(f"DLRM model trained and registered: {model_id}")
        return model_id
    
    def train_xgboost(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        labels: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Train XGBoost ranking model.
        
        Args:
            user_features: User feature matrix
            item_features: Item feature matrix
            labels: Interaction labels
            config: Model configuration
            
        Returns:
            model_id: Registered model ID
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required for XGBoost training")
        
        logger.info("Training XGBoost model...")
        
        # Combine features
        X = np.hstack([user_features, item_features])
        y = labels
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Default parameters
        params = config or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100
        )
        
        # Register model
        metrics = {}  # Could add validation metrics
        hyperparameters = params
        
        model_id = self.registry.register_model(
            model=model,
            model_type='xgboost',
            metrics=metrics,
            hyperparameters=hyperparameters,
            description="XGBoost ranking model"
        )
        
        logger.info(f"XGBoost model trained and registered: {model_id}")
        return model_id
    
    def train_lightgbm(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        labels: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Train LightGBM ranking model.
        
        Args:
            user_features: User feature matrix
            item_features: Item feature matrix
            labels: Interaction labels
            config: Model configuration
            
        Returns:
            model_id: Registered model ID
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required for LightGBM training")
        
        logger.info("Training LightGBM model...")
        
        # Combine features
        X = np.hstack([user_features, item_features])
        y = labels
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Default parameters
        params = config or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )
        
        # Register model
        metrics = {}
        hyperparameters = params
        
        model_id = self.registry.register_model(
            model=model,
            model_type='lightgbm',
            metrics=metrics,
            hyperparameters=hyperparameters,
            description="LightGBM ranking model"
        )
        
        logger.info(f"LightGBM model trained and registered: {model_id}")
        return model_id
    
    def export_training_report(self, output_path: str):
        """Export training history to JSON."""
        report = {
            'exported_at': datetime.now().isoformat(),
            'total_models_trained': len(self.training_history),
            'training_history': self.training_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported training report to {output_path}")
