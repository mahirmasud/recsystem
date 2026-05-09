"""Batch Generator - Pre-computes recommendations in batches."""
from typing import Dict, Any, List
import logging
import numpy as np
logger = logging.getLogger(__name__)

class BatchGenerator:
    """Generates batch recommendations offline."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_recommendations = {}
        
    def generate_batch(self, user_ids: List[int], engine, n: int = 50):
        """Pre-generate recommendations for a batch of users."""
        logger.info(f"Generating batch recommendations for {len(user_ids)} users")
        
        for user_id in user_ids:
            try:
                recs = engine.get_recommendations(user_id, n=n)
                self.batch_recommendations[user_id] = recs
            except Exception as e:
                logger.error(f"Error generating recs for user {user_id}: {e}")
        
        return len(self.batch_recommendations)
    
    def get_batch_recs(self, user_id: int) -> List[Dict]:
        """Get pre-computed recommendations."""
        return self.batch_recommendations.get(user_id, [])
    
    def clear_batch(self):
        """Clear batch cache."""
        self.batch_recommendations = {}
