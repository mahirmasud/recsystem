"""Serving API - REST API for recommendations."""
from typing import Dict, Any, List, Optional
import logging
import json
logger = logging.getLogger(__name__)

class RecommendationAPI:
    """REST API interface for recommendation serving."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recommendation_engine = None
        logger.info("RecommendationAPI initialized")
    
    def set_engine(self, engine):
        """Set the recommendation engine."""
        self.recommendation_engine = engine
    
    def get_recommendations(self, user_id: int, n: int = 10, 
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get recommendations for a user."""
        if not self.recommendation_engine:
            return {'error': 'Engine not configured'}
        
        recommendations = self.recommendation_engine.get_recommendations(
            user_id=user_id, n=n, context=context or {}
        )
        
        return {
            'user_id': user_id,
            'recommendations': [{'item_id': r['item_id'], 'score': r['score']} for r in recommendations],
            'count': len(recommendations)
        }
    
    def get_similar_items(self, item_id: int, n: int = 10) -> Dict[str, Any]:
        """Get similar items."""
        if not self.recommendation_engine:
            return {'error': 'Engine not configured'}
        
        similar = self.recommendation_engine.get_similar_items(item_id, n)
        
        return {
            'item_id': item_id,
            'similar_items': similar,
            'count': len(similar)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """API health check."""
        return {'status': 'healthy', 'engine_ready': self.recommendation_engine is not None}
