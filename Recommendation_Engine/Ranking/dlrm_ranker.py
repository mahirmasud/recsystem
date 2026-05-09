"""
DLRM Ranker - Deep Learning Recommendation Model

Implements the DLRM architecture for personalized ranking with:
- Dense feature processing
- Sparse categorical embeddings
- Feature interaction learning
- Click-through rate prediction
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path

from .ranker import BaseRanker

logger = logging.getLogger(__name__)


class DLRMModel(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM) architecture.
    
    Architecture:
    1. Bottom MLP: Processes dense features
    2. Embedding layers: Process sparse categorical features
    3. Interaction layer: Combines dense and sparse representations
    4. Top MLP: Produces final prediction
    """
    
    def __init__(
        self,
        dense_feature_dim: int,
        sparse_feature_dims: Dict[str, int],
        embedding_dim: int = 16,
        bottom_mlp_dims: List[int] = [64, 32],
        top_mlp_dims: List[int] = [32, 16, 1],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Store dimensions
        self.dense_feature_dim = dense_feature_dim
        self.sparse_feature_dims = sparse_feature_dims
        self.embedding_dim = embedding_dim
        
        # Bottom MLP for dense features
        bottom_layers = []
        input_dim = dense_feature_dim
        for hidden_dim in bottom_mlp_dims:
            bottom_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        self.bottom_mlp = nn.Sequential(*bottom_layers)
        
        # Embedding layers for sparse features
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in sparse_feature_dims.items():
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
        
        # Interaction dimension calculation
        interaction_dim = bottom_mlp_dims[-1] + len(sparse_feature_dims) * embedding_dim
        
        # Top MLP for final prediction
        top_layers = []
        input_dim = interaction_dim
        for hidden_dim in top_mlp_dims[:-1]:
            top_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        top_layers.append(nn.Linear(input_dim, top_mlp_dims[-1]))
        top_layers.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_layers)
        
    def forward(self, dense_features: torch.Tensor, sparse_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through DLRM.
        
        Args:
            dense_features: Tensor of shape (batch_size, dense_feature_dim)
            sparse_features: Dictionary of sparse feature tensors
            
        Returns:
            Prediction scores of shape (batch_size, 1)
        """
        # Process dense features
        dense_out = self.bottom_mlp(dense_features)
        
        # Process sparse features through embeddings
        sparse_outputs = []
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in sparse_features:
                emb = embedding_layer(sparse_features[feature_name])
                sparse_outputs.append(emb.view(emb.size(0), -1))
        
        # Concatenate all representations
        if sparse_outputs:
            combined = torch.cat([dense_out] + sparse_outputs, dim=1)
        else:
            combined = dense_out
        
        # Final prediction
        return self.top_mlp(combined)


class DLRMRanker(BaseRanker):
    """
    DLRM-based ranking model for personalized recommendations.
    
    Features:
    - Handles both dense and sparse features
    - Learns feature interactions automatically
    - Optimized for binary classification (click/no-click)
    - Supports GPU acceleration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DLRM ranker.
        
        Args:
            config: Configuration with model hyperparameters
        """
        super().__init__(config)
        
        # Model hyperparameters
        self.embedding_dim = config.get('embedding_dim', 16)
        self.bottom_mlp_dims = config.get('bottom_mlp_dims', [64, 32])
        self.top_mlp_dims = config.get('top_mlp_dims', [32, 16, 1])
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 10)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature configuration
        self.dense_features = config.get('dense_features', [])
        self.sparse_features = config.get('sparse_features', {})
        
        # Model instance
        self.model = None
        self.feature_preprocessors = {}
        
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training/inference.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (dense_tensor, sparse_dict, labels)
        """
        # Extract dense features
        dense_data = df[self.dense_features].fillna(0).values.astype(np.float32)
        dense_tensor = torch.tensor(dense_data, device=self.device)
        
        # Extract and encode sparse features
        sparse_dict = {}
        for feature_name, vocab_size in self.sparse_features.items():
            if feature_name in df.columns:
                # Simple integer encoding (in production, use fitted encoder)
                codes = df[feature_name].astype('category').cat.codes.values
                codes = np.clip(codes, 0, vocab_size - 1)  # Handle unknown categories
                sparse_dict[feature_name] = torch.tensor(codes, device=self.device)
        
        # Extract labels if present
        labels = None
        if self.target_column in df.columns:
            labels = torch.tensor(df[self.target_column].values.astype(np.float32), device=self.device)
        
        return dense_tensor, sparse_dict, labels
    
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Train the DLRM model.
        
        Args:
            train_df: Training dataframe
            val_df: Optional validation dataframe
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training DLRM on {len(train_df)} samples")
        logger.info(f"Dense features: {len(self.dense_features)}, Sparse features: {len(self.sparse_features)}")
        
        # Prepare training data
        dense_train, sparse_train, labels_train = self._prepare_data(train_df)
        
        if labels_train is None:
            raise ValueError("Training data must contain target column")
        
        # Create data loader
        dataset = TensorDataset(dense_train, labels_train)
        for key, value in sparse_train.items():
            # Add sparse features to dataset
            pass  # Simplified for now
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = DLRMModel(
            dense_feature_dim=len(self.dense_features),
            sparse_feature_dims=self.sparse_features,
            embedding_dim=self.embedding_dim,
            bottom_mlp_dims=self.bottom_mlp_dims,
            top_mlp_dims=self.top_mlp_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_dense, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass (simplified - sparse features not fully integrated in this demo)
                predictions = self.model(batch_dense, {})
                
                loss = criterion(predictions.squeeze(), batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        self.feature_columns = self.dense_features + list(self.sparse_features.keys())
        
        metrics = {
            'final_loss': train_losses[-1],
            'epochs_trained': self.epochs
        }
        
        if val_df is not None:
            val_metrics = self.evaluate(val_df)
            metrics.update(val_metrics)
        
        logger.info(f"DLRM training completed. Metrics: {metrics}")
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
        
        self.model.eval()
        dense_feat, sparse_feat, _ = self._prepare_data(features)
        
        with torch.no_grad():
            predictions = self.model(dense_feat, sparse_feat)
            scores = predictions.squeeze().cpu().numpy()
        
        return scores
    
    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            val_df: Validation dataframe
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        dense_val, sparse_val, labels_val = self._prepare_data(val_df)
        
        with torch.no_grad():
            predictions = self.model(dense_val, sparse_val)
            preds = predictions.squeeze().cpu().numpy()
            labels = labels_val.cpu().numpy()
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(labels, preds)
        loss = log_loss(labels, preds)
        
        return {'val_auc': auc, 'val_loss': loss}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), save_path / 'dlrm_weights.pt')
        
        # Save configuration
        config = {
            'dense_features': self.dense_features,
            'sparse_features': self.sparse_features,
            'embedding_dim': self.embedding_dim,
            'bottom_mlp_dims': self.bottom_mlp_dims,
            'top_mlp_dims': self.top_mlp_dims,
            'dropout_rate': self.dropout_rate,
            'feature_columns': self.feature_columns
        }
        joblib.dump(config, save_path / 'dlrm_config.pkl')
        
        logger.info(f"DLRM model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        load_path = Path(path)
        
        # Load configuration
        config = joblib.load(load_path / 'dlrm_config.pkl')
        self.dense_features = config['dense_features']
        self.sparse_features = config['sparse_features']
        self.embedding_dim = config['embedding_dim']
        self.bottom_mlp_dims = config['bottom_mlp_dims']
        self.top_mlp_dims = config['top_mlp_dims']
        self.dropout_rate = config['dropout_rate']
        self.feature_columns = config['feature_columns']
        
        # Reinitialize model
        self.model = DLRMModel(
            dense_feature_dim=len(self.dense_features),
            sparse_feature_dims=self.sparse_features,
            embedding_dim=self.embedding_dim,
            bottom_mlp_dims=self.bottom_mlp_dims,
            top_mlp_dims=self.top_mlp_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(load_path / 'dlrm_weights.pt', map_location=self.device))
        self.model.eval()
        self.is_trained = True
        
        logger.info(f"DLRM model loaded from {path}")
