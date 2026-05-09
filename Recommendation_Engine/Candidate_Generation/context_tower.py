"""Context Tower - Encodes contextual signals."""
import numpy as np
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

class ContextTower:
    """Neural network tower for context encoding."""
    def __init__(self, config: Dict[str, Any], embedding_dim: int = 32):
        self.config = config
        self.embedding_dim = embedding_dim
        self.weights = None
        
    def build(self, input_dim: int):
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(input_dim, 64) * 0.01,
            'b1': np.zeros(64),
            'W2': np.random.randn(64, self.embedding_dim) * 0.01,
            'b2': np.zeros(self.embedding_dim)
        }
        return self
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, np.dot(x, self.weights['W1']) + self.weights['b1'])
        out = np.dot(h, self.weights['W2']) + self.weights['b2']
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
