"""Matrix Factorization - Classic MF for collaborative filtering."""
import numpy as np
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class MatrixFactorization:
    """Matrix factorization using SGD."""
    def __init__(self, config: Dict[str, Any], n_factors: int = 50):
        self.config = config
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0
        
    def fit(self, interactions: List[tuple], n_users: int, n_items: int, 
            epochs: int = 20, lr: float = 0.01, reg: float = 0.02):
        """Train matrix factorization model."""
        logger.info(f"Training MF with {self.n_factors} factors")
        
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        ratings = [r for _, _, r in interactions if r > 0]
        self.global_mean = np.mean(ratings) if ratings else 3.0
        
        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(interactions)
            
            for user_id, item_id, rating in interactions[:min(10000, len(interactions))]:
                pred = self.global_mean + self.user_bias[user_id] + self.item_bias[item_id]
                pred += np.dot(self.user_factors[user_id], self.item_factors[item_id])
                
                error = rating - pred
                total_loss += error ** 2
                
                # Update biases
                self.user_bias[user_id] += lr * (error - reg * self.user_bias[user_id])
                self.item_bias[item_id] += lr * (error - reg * self.item_bias[item_id])
                
                # Update factors
                self.user_factors[user_id] += lr * (error * self.item_factors[item_id] - reg * self.user_factors[user_id])
                self.item_factors[item_id] += lr * (error * self.user_factors[user_id] - reg * self.item_factors[item_id])
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(interactions):.4f}")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        pred = self.global_mean + self.user_bias[user_id] + self.item_bias[item_id]
        pred += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return float(np.clip(pred, 0.5, 5.0))
    
    def recommend(self, user_id: int, n: int = 10, exclude: List[int] = None) -> List[int]:
        """Get top-n recommendations for a user."""
        exclude = exclude or []
        scores = self.global_mean + self.user_bias[user_id] + self.item_bias
        scores += np.dot(self.user_factors[user_id], self.item_factors.T)
        
        for item_id in exclude:
            scores[item_id] = -float('inf')
        
        return list(np.argsort(scores)[-n:][::-1])
