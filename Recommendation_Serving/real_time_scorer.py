"""Real-Time Scorer - Scores candidates in real-time."""
import numpy as np
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class RealTimeScorer:
    """Scores recommendation candidates in real-time."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def set_model(self, model):
        """Set scoring model."""
        self.model = model
        
    def score_candidates(self, user_features: np.ndarray, 
                         candidate_features: np.ndarray) -> np.ndarray:
        """Score candidates for a user."""
        if self.model is None:
            # Fallback: simple dot product
            return np.sum(user_features * candidate_features, axis=1)
        
        # Use model for scoring
        return self.model.predict(user_features, candidate_features)
    
    def rank_candidates(self, candidates: List[Dict], user_context: Dict) -> List[Dict]:
        """Rank candidates by score."""
        for c in candidates:
            c['original_score'] = c.get('score', 0)
            # Apply real-time adjustments based on context
            if user_context.get('is_mobile'):
                c['score'] *= 1.05  # Mobile boost
            if user_context.get('time_of_day') == 'evening':
                c['score'] *= 1.02
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
