"""
Realtime Recommendation - Real-time recommendation generation.

Handles real-time, on-demand recommendation requests with
low-latency requirements.
"""

from typing import Dict, Any, Optional, List
import time
import logging
from shared.logger import get_logger
from .request_parser import RequestParser
from .recommendation_service import RecommendationService
from .recommendation_formatter import RecommendationFormatter
from .explanation_generator import ExplanationGenerator
from .trace_logger import TraceLogger

logger = get_logger(__name__)


class RealtimeRecommendation:
    """
    Handles real-time recommendation requests.
    
    Features:
    - Low-latency recommendation generation
    - Real-time context incorporation
    - Session-aware recommendations
    - Performance tracking
    """
    
    def __init__(
        self,
        service: RecommendationService,
        enable_tracing: bool = True
    ):
        """
        Initialize realtime recommendation handler.
        
        Args:
            service: RecommendationService instance
            enable_tracing: Whether to enable request tracing
        """
        self.service = service
        self.parser = RequestParser()
        self.formatter = RecommendationFormatter()
        self.explainer = ExplanationGenerator()
        self.trace_logger = TraceLogger(enabled=enable_tracing)
        
        logger.info("RealtimeRecommendation initialized")
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get real-time recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            context: Optional context dictionary
            
        Returns:
            Recommendation response dictionary
        """
        start_time = time.time()
        trace_id = f"rt_{user_id}_{int(start_time)}"
        
        # Start trace
        self.trace_logger.start_trace(trace_id, {
            'user_id': user_id,
            'n': n,
            'context': context,
            'type': 'realtime'
        })
        
        try:
            # Get recommendations from service
            result = self.service.get_recommendations(
                user_id=user_id,
                n=n,
                context=context or {}
            )
            
            # Add explanations
            if result.get('recommendations'):
                explanations = self.explainer.generate_batch_explanations(
                    result['recommendations'],
                    user_context=context
                )
                for rec in result['recommendations']:
                    item_id = rec.get('item_id')
                    if item_id in explanations:
                        rec['explanation'] = explanations[item_id]
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log stages
            self.trace_logger.log_stage(
                trace_id, 
                'generation', 
                {'latency_ms': latency_ms}
            )
            
            # End trace
            self.trace_logger.end_trace(trace_id, {
                'recommendation_count': len(result.get('recommendations', [])),
                'latency_ms': latency_ms,
                'success': 'error' not in result
            })
            
            # Add metadata
            result['latency_ms'] = round(latency_ms, 2)
            result['request_type'] = 'realtime'
            
            logger.info(
                f"Real-time recommendations for user {user_id}: "
                f"{len(result.get('recommendations', []))} items in {latency_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time recommendation failed: {e}", exc_info=True)
            
            self.trace_logger.log_stage(trace_id, 'error', {'error': str(e)})
            self.trace_logger.end_trace(trace_id, {'error': str(e)})
            
            return {
                'user_id': user_id,
                'recommendations': [],
                'error': str(e),
                'request_type': 'realtime'
            }
    
    def get_similar_items(
        self,
        item_id: int,
        n: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get similar items in real-time.
        
        Args:
            item_id: Item ID
            n: Number of recommendations
            context: Optional context dictionary
            
        Returns:
            Recommendation response dictionary
        """
        start_time = time.time()
        trace_id = f"rt_item_{item_id}_{int(start_time)}"
        
        self.trace_logger.start_trace(trace_id, {
            'item_id': item_id,
            'n': n,
            'type': 'similar_items'
        })
        
        try:
            result = self.service.get_recommendations(
                item_id=item_id,
                n=n,
                context=context or {}
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.trace_logger.end_trace(trace_id, {
                'recommendation_count': len(result.get('recommendations', [])),
                'latency_ms': latency_ms
            })
            
            result['latency_ms'] = round(latency_ms, 2)
            result['request_type'] = 'similar_items'
            
            logger.info(
                f"Similar items for item {item_id}: "
                f"{len(result.get('recommendations', []))} items in {latency_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Similar items failed: {e}", exc_info=True)
            return {
                'item_id': item_id,
                'recommendations': [],
                'error': str(e)
            }
    
    def get_session_recommendations(
        self,
        session_id: str,
        n: int = 10
    ) -> Dict[str, Any]:
        """
        Get recommendations based on session activity.
        
        Args:
            session_id: Session ID
            n: Number of recommendations
            
        Returns:
            Recommendation response dictionary
        """
        if not self.service.session_handler:
            return {'error': 'Session handler not available'}
        
        session = self.service.session_handler.get_session(session_id)
        if not session:
            return {'error': 'Session not found or expired'}
        
        # Get session signals
        signals = self.service.session_handler.get_session_signals(session_id)
        
        # Use viewed items for context
        viewed_items = signals.get('viewed_items', [])
        cart_items = signals.get('cart_items', [])
        
        context = {
            'session_id': session_id,
            'viewed_items': viewed_items,
            'cart_items': cart_items,
            'device_type': signals.get('device_type')
        }
        
        # Get recommendations for the session's user
        user_id = session.user_id
        if user_id is None:
            return {'error': 'Anonymous session'}
        
        return self.get_recommendations(user_id, n, context)
    
    def record_interaction(
        self,
        session_id: str,
        interaction_type: str,
        item_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a user interaction for session-based personalization.
        
        Args:
            session_id: Session ID
            interaction_type: Type of interaction (view, click, cart, etc.)
            item_id: Item ID
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        if not self.service.session_handler:
            return False
        
        return self.service.session_handler.add_session_interaction(
            session_id=session_id,
            interaction_type=interaction_type,
            item_id=item_id,
            metadata=metadata
        )
