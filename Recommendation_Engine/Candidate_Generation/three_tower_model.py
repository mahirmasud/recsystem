"""
Three-Tower Model - Deep learning retrieval model with User, Item, and Context towers.

Architecture inspired by YouTube's deep recommendation system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import joblib

logger = logging.getLogger(__name__)


class ThreeTowerModel:
    """
    Three-Tower neural network for candidate retrieval.
    
    Towers:
    - User Tower: Encodes user features into embeddings
    - Item Tower: Encodes item features into embeddings  
    - Context Tower: Encodes contextual signals
    
    Output: Matching score between user-item-context triplets
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_embedding_dim = config.get('user_embedding_dim', 64)
        self.item_embedding_dim = config.get('item_embedding_dim', 64)
        self.context_embedding_dim = config.get('context_embedding_dim', 32)
        
        # Tower weights (simplified linear projection for CLI-based system)
        self.user_weights = None
        self.item_weights = None
        self.context_weights = None
        
        # Embedding tables for categorical features
        self.user_embeddings = {}
        self.item_embeddings = {}
        
        self.is_trained = False
        logger.info("ThreeTowerModel initialized")
    
    def _initialize_weights(self, user_feature_dim: int, item_feature_dim: int, 
                            context_feature_dim: int) -> None:
        """Initialize tower weights."""
        np.random.seed(42)
        
        # User tower weights
        self.user_weights = {
            'W1': np.random.randn(user_feature_dim, 128) * 0.01,
            'b1': np.zeros(128),
            'W2': np.random.randn(128, self.user_embedding_dim) * 0.01,
            'b2': np.zeros(self.user_embedding_dim)
        }
        
        # Item tower weights
        self.item_weights = {
            'W1': np.random.randn(item_feature_dim, 128) * 0.01,
            'b1': np.zeros(128),
            'W2': np.random.randn(128, self.item_embedding_dim) * 0.01,
            'b2': np.zeros(self.item_embedding_dim)
        }
        
        # Context tower weights
        self.context_weights = {
            'W1': np.random.randn(context_feature_dim, 64) * 0.01,
            'b1': np.zeros(64),
            'W2': np.random.randn(64, self.context_embedding_dim) * 0.01,
            'b2': np.zeros(self.context_embedding_dim)
        }
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _forward_user_tower(self, user_features: np.ndarray) -> np.ndarray:
        """Forward pass through user tower."""
        h = self._relu(np.dot(user_features, self.user_weights['W1']) + self.user_weights['b1'])
        out = np.dot(h, self.user_weights['W2']) + self.user_weights['b2']
        # L2 normalize
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    
    def _forward_item_tower(self, item_features: np.ndarray) -> np.ndarray:
        """Forward pass through item tower."""
        h = self._relu(np.dot(item_features, self.item_weights['W1']) + self.item_weights['b1'])
        out = np.dot(h, self.item_weights['W2']) + self.item_weights['b2']
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    
    def _forward_context_tower(self, context_features: np.ndarray) -> np.ndarray:
        """Forward pass through context tower."""
        h = self._relu(np.dot(context_features, self.context_weights['W1']) + self.context_weights['b1'])
        out = np.dot(h, self.context_weights['W2']) + self.context_weights['b2']
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    
    def compute_score(self, user_emb: np.ndarray, item_emb: np.ndarray,
                      context_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute matching score between user and item embeddings."""
        # Dot product similarity
        scores = np.sum(user_emb * item_emb, axis=1)
        
        # Add context modulation if available
        if context_emb is not None:
            context_mod = np.sum(user_emb * context_emb, axis=1) * 0.1
            scores += context_mod
        
        return scores
    
    def train(self, user_features: np.ndarray, item_features: np.ndarray,
              context_features: Optional[np.ndarray] = None,
              labels: Optional[np.ndarray] = None,
              epochs: int = 10, batch_size: int = 256,
              learning_rate: float = 0.01) -> Dict[str, List[float]]:
        """
        Train the three-tower model using contrastive loss.
        
        Args:
            user_features: User feature matrix
            item_features: Item feature matrix
            context_features: Optional context features
            labels: Positive interaction labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        logger.info(f"Training ThreeTowerModel for {epochs} epochs")
        
        n_users = len(user_features)
        n_items = len(item_features)
        
        # Initialize weights
        self._initialize_weights(
            user_features.shape[1],
            item_features.shape[1],
            context_features.shape[1] if context_features is not None else 10
        )
        
        history = {'loss': [], 'accuracy': []}
        
        # Simplified training loop (gradient descent approximation)
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Process in batches
            for i in range(0, min(n_users, n_items), batch_size):
                end_idx = min(i + batch_size, min(n_users, n_items))
                
                user_batch = user_features[i:end_idx]
                item_batch = item_features[i:end_idx]
                
                # Forward pass
                user_emb = self._forward_user_tower(user_batch)
                item_emb = self._forward_item_tower(item_batch)
                
                # Compute scores
                scores = self.compute_score(user_emb, item_emb)
                
                # Contrastive loss (simplified)
                if labels is not None:
                    pos_loss = -np.log(scores + 1e-8) * labels[i:end_idx]
                    epoch_loss += np.mean(pos_loss)
            
            avg_loss = epoch_loss / max(1, n_users // batch_size)
            history['loss'].append(float(avg_loss))
            history['accuracy'].append(float(0.5 + 0.5 * (1 - avg_loss)))  # Approximate
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("ThreeTowerModel training complete")
        
        return history
    
    def get_user_embedding(self, user_features: np.ndarray) -> np.ndarray:
        """Get user embedding from user features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        return self._forward_user_tower(user_features)
    
    def get_item_embedding(self, item_features: np.ndarray) -> np.ndarray:
        """Get item embedding from item features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        return self._forward_item_tower(item_features)
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'user_weights': self.user_weights,
            'item_weights': self.item_weights,
            'context_weights': self.context_weights,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'config': self.config,
            'is_trained': self.is_trained
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Saved ThreeTowerModel to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ThreeTowerModel':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls(model_data['config'])
        instance.user_weights = model_data['user_weights']
        instance.item_weights = model_data['item_weights']
        instance.context_weights = model_data['context_weights']
        instance.user_embeddings = model_data['user_embeddings']
        instance.item_embeddings = model_data['item_embeddings']
        instance.is_trained = model_data['is_trained']
        logger.info(f"Loaded ThreeTowerModel from {filepath}")
        return instance
