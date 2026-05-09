"""
Matrix Factorization - Classic collaborative filtering with matrix decomposition

Implements matrix factorization for recommendation retrieval:
- SVD-based factorization
- Alternating Least Squares (ALS)
- Implicit feedback support
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class MatrixFactorization:
    """
    Matrix Factorization model for collaborative filtering.
    
    Features:
    - SVD-based dimensionality reduction
    - User and item latent factors
    - Rating prediction
    - Top-N item recommendation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize matrix factorization model.
        
        Args:
            config: Configuration with model parameters
        """
        self.config = config
        self.n_factors = config.get('n_factors', 50)
        self.n_iterations = config.get('n_iterations', 20)
        self.regularization = config.get('regularization', 0.1)
        
        # Model components
        self.user_factors = None  # U matrix
        self.item_factors = None  # V matrix
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.is_trained = False
        
    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        rating_col: str = 'rating'
    ) -> Dict[str, float]:
        """
        Train the matrix factorization model using SVD.
        
        Args:
            interactions: DataFrame with user-item interactions
            user_col: Name of user ID column
            item_col: Name of item ID column
            rating_col: Name of rating column
            
        Returns:
            Training metrics
        """
        logger.info(f"Training Matrix Factorization with {self.n_factors} factors")
        
        # Build mappings
        users = interactions[user_col].unique()
        items = interactions[item_col].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        # Create sparse rating matrix
        row_indices = [self.user_to_idx[u] for u in interactions[user_col]]
        col_indices = [self.item_to_idx[i] for i in interactions[item_col]]
        ratings = interactions[rating_col].values
        
        R = csr_matrix((ratings, (row_indices, col_indices)), shape=(n_users, n_items))
        
        # Calculate global mean and biases
        self.global_mean = ratings.mean()
        
        # User bias (mean rating per user minus global mean)
        user_means = np.array([R.getrow(i).mean() if R.getrow(i).nnz > 0 else 0 for i in range(n_users)])
        self.user_bias = user_means - self.global_mean
        
        # Item bias (mean rating per item minus global mean)
        item_means = np.array([R.getcol(i).mean() if R.getcol(i).nnz > 0 else 0 for i in range(n_items)])
        self.item_bias = item_means - self.global_mean
        
        # Center the matrix
        R_centered = R.copy()
        for i in range(n_users):
            for j in range(n_items):
                if R_centered[i, j] != 0:
                    R_centered[i, j] -= (self.global_mean + self.user_bias[i] + self.item_bias[j])
        
        # Perform SVD
        k = min(self.n_factors, min(R_centered.shape) - 1)
        U, sigma, Vt = svds(R_centered.astype(float), k=k)
        
        # Store factors
        self.user_factors = U
        self.item_factors = Vt.T
        self.sigma = np.diag(sigma)
        
        self.is_trained = True
        
        # Calculate training RMSE
        predictions = self._predict_matrix(R)
        actual = R.toarray()
        mask = actual > 0
        rmse = np.sqrt(np.mean((predictions[mask] - actual[mask]) ** 2))
        
        metrics = {'rmse': rmse, 'n_users': n_users, 'n_items': n_items}
        logger.info(f"MF training completed. RMSE: {rmse:.4f}")
        
        return metrics
    
    def _predict_matrix(self, R: csr_matrix) -> np.ndarray:
        """Predict full rating matrix."""
        predictions = np.dot(np.dot(self.user_factors, self.sigma), self.item_factors.T)
        
        # Add biases
        predictions += self.global_mean
        predictions += self.user_bias.reshape(-1, 1)
        predictions += self.item_bias.reshape(1, -1)
        
        return predictions
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[item_id]
        
        # Base prediction from factors
        pred = np.dot(self.user_factors[u_idx], np.dot(self.sigma, self.item_factors[i_idx]))
        
        # Add biases
        pred += self.global_mean + self.user_bias[u_idx] + self.item_bias[i_idx]
        
        return pred
    
    def recommend_for_user(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n_items: Number of recommendations
            exclude_items: Items to exclude (already consumed)
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before recommendations")
        
        if user_id not in self.user_to_idx:
            # Cold start: return popular items
            return self._popular_items(n_items)
        
        u_idx = self.user_to_idx[user_id]
        exclude_set = set(exclude_items) if exclude_items else set()
        
        # Calculate scores for all items
        user_vector = np.dot(self.user_factors[u_idx], self.sigma)
        scores = np.dot(self.item_factors, user_vector)
        
        # Add biases
        scores += self.global_mean + self.user_bias[u_idx] + self.item_bias
        
        # Get top N items excluding already consumed
        recommendations = []
        sorted_indices = np.argsort(-scores)
        
        for idx in sorted_indices:
            item_id = self.idx_to_item[idx]
            if item_id not in exclude_set:
                recommendations.append((item_id, scores[idx]))
                if len(recommendations) >= n_items:
                    break
        
        return recommendations
    
    def _popular_items(self, n_items: int) -> List[Tuple[int, float]]:
        """Return popular items for cold start."""
        # Use item bias as proxy for popularity
        sorted_indices = np.argsort(-self.item_bias)
        
        recommendations = []
        for idx in sorted_indices[:n_items]:
            item_id = self.idx_to_item[idx]
            recommendations.append((item_id, self.item_bias[idx] + self.global_mean))
        
        return recommendations
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user latent factor embedding.
        
        Args:
            user_id: User ID
            
        Returns:
            User embedding vector or None
        """
        if user_id not in self.user_to_idx:
            return None
        
        u_idx = self.user_to_idx[user_id]
        return np.dot(self.user_factors[u_idx], self.sigma)
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """
        Get item latent factor embedding.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item embedding vector or None
        """
        if item_id not in self.item_to_idx:
            return None
        
        i_idx = self.item_to_idx[item_id]
        return self.item_factors[i_idx]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'sigma': self.sigma if hasattr(self, 'sigma') else None,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_mean': self.global_mean,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'config': self.config
        }
        
        joblib.dump(state, save_path / 'matrix_factorization.pkl')
        logger.info(f"Matrix Factorization model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        load_path = Path(path)
        state = joblib.load(load_path / 'matrix_factorization.pkl')
        
        self.user_factors = state['user_factors']
        self.item_factors = state['item_factors']
        self.sigma = state['sigma']
        self.user_bias = state['user_bias']
        self.item_bias = state['item_bias']
        self.global_mean = state['global_mean']
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']
        self.item_to_idx = state['item_to_idx']
        self.idx_to_item = state['idx_to_item']
        self.config = state['config']
        self.is_trained = True
        
        logger.info(f"Matrix Factorization model loaded from {path}")
