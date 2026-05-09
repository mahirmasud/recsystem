"""
Recommendation Service - Core service for generating recommendations.

Orchestrates the recommendation pipeline including candidate generation,
ranking, re-ranking, and result formatting.
"""

from typing import Dict, Any, Optional, List
import logging
from shared.logger import get_logger
from .request_parser import RecommendationRequest, RequestParser
from .cache_handler import CacheHandler
from .session_handler import SessionHandler

logger = get_logger(__name__)


class RecommendationService:
    """
    Core service for generating recommendations.
    
    Responsibilities:
    - Orchestrate recommendation pipeline
    - Manage candidate generation
    - Apply ranking models
    - Execute re-ranking strategies
    - Handle caching
    - Track recommendation metadata
    
    Pipeline stages:
    1. Parse request
    2. Check cache
    3. Generate candidates
    4. Rank candidates
    5. Re-rank results
    6. Format output
    7. Cache results
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_ttl: int = 3600,
        enable_caching: bool = True
    ):
        """
        Initialize the recommendation service.
        
        Args:
            config: Configuration dictionary
            cache_ttl: Cache TTL in seconds
            enable_caching: Whether to enable caching
        """
        self.config = config or {}
        self.enable_caching = enable_caching
        
        # Initialize components
        self.request_parser = RequestParser(config)
        self.cache = CacheHandler(ttl_seconds=cache_ttl) if enable_caching else None
        self.session_handler = SessionHandler()
        
        # These will be set by external modules
        self.candidate_generator = None
        self.ranker = None
        self.reranker = None
        self.rule_engine = None
        
        logger.info("RecommendationService initialized")
    
    def set_candidate_generator(self, generator) -> None:
        """Set the candidate generation engine."""
        self.candidate_generator = generator
        logger.info("Candidate generator configured")
    
    def set_ranker(self, ranker) -> None:
        """Set the ranking model."""
        self.ranker = ranker
        logger.info("Ranker configured")
    
    def set_reranker(self, reranker) -> None:
        """Set the re-ranking engine."""
        self.reranker = reranker
        logger.info("Re-ranker configured")
    
    def set_rule_engine(self, rule_engine) -> None:
        """Set the business rule engine."""
        self.rule_engine = rule_engine
        logger.info("Rule engine configured")
    
    def get_recommendations(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        n: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get recommendations for a user or item.
        
        Args:
            user_id: User ID (for personalized recommendations)
            item_id: Item ID (for similar items)
            n: Number of recommendations
            context: Optional context dictionary
            
        Returns:
            Dictionary with recommendations and metadata
        """
        # Parse request
        if user_id is not None:
            request = self.request_parser.parse_user_request(user_id, n, context)
        elif item_id is not None:
            request = self.request_parser.parse_item_request(item_id, n, context)
        else:
            raise ValueError("Either user_id or item_id must be provided")
        
        return self._execute_pipeline(request)
    
    def _execute_pipeline(
        self, 
        request: RecommendationRequest
    ) -> Dict[str, Any]:
        """
        Execute the full recommendation pipeline.
        
        Args:
            request: Parsed recommendation request
            
        Returns:
            Complete recommendation response
        """
        response = {
            'request_id': request.request_id or self._generate_request_id(),
            'user_id': request.user_id,
            'item_id': request.item_id,
            'recommendations': [],
            'metadata': {
                'candidates_generated': 0,
                'candidates_ranked': 0,
                'final_count': 0,
                'cache_hit': False,
                'applied_rules': [],
                'processing_stages': []
            }
        }
        
        try:
            # Stage 1: Check cache
            if self.enable_caching and self.cache:
                cached = self._check_cache(request)
                if cached:
                    response['recommendations'] = cached
                    response['metadata']['cache_hit'] = True
                    response['metadata']['final_count'] = len(cached)
                    logger.info(f"Cache hit for request {response['request_id']}")
                    return response
            
            response['metadata']['processing_stages'].append('cache_miss')
            
            # Stage 2: Generate candidates
            candidates = self._generate_candidates(request)
            response['metadata']['candidates_generated'] = len(candidates)
            response['metadata']['processing_stages'].append('candidate_generation')
            
            if not candidates:
                logger.warning("No candidates generated")
                return response
            
            # Stage 3: Rank candidates
            ranked = self._rank_candidates(request, candidates)
            response['metadata']['candidates_ranked'] = len(ranked)
            response['metadata']['processing_stages'].append('ranking')
            
            # Stage 4: Re-rank
            reranked = self._rerank_candidates(request, ranked)
            response['metadata']['processing_stages'].append('reranking')
            
            # Stage 5: Apply business rules
            if self.rule_engine and request.apply_business_boost:
                reranked = self._apply_business_rules(request, reranked)
                response['metadata']['processing_stages'].append('business_rules')
            
            # Stage 6: Select top-N
            final_recommendations = reranked[:request.n_recommendations]
            
            # Stage 7: Format results
            formatted = self._format_results(final_recommendations, request)
            response['recommendations'] = formatted
            response['metadata']['final_count'] = len(formatted)
            
            # Stage 8: Cache results
            if self.enable_caching and self.cache:
                self._cache_results(request, formatted)
            
            logger.info(
                f"Generated {len(formatted)} recommendations for "
                f"user={request.user_id}, item={request.item_id}"
            )
            
        except Exception as e:
            logger.error(f"Error in recommendation pipeline: {e}", exc_info=True)
            response['error'] = str(e)
        
        return response
    
    def _check_cache(self, request: RecommendationRequest) -> Optional[List[Dict]]:
        """Check cache for existing recommendations."""
        if not self.cache:
            return None
        
        return self.cache.get(
            user_id=request.user_id,
            item_id=request.item_id,
            context={
                'n': request.n_recommendations,
                'device_type': request.device_type
            }
        )
    
    def _generate_candidates(self, request: RecommendationRequest) -> List[Dict]:
        """Generate candidate items."""
        if self.candidate_generator is None:
            logger.warning("No candidate generator configured")
            return []
        
        try:
            if request.user_id is not None:
                return self.candidate_generator.generate(
                    user_id=request.user_id,
                    n=request.n_candidates,
                    context={
                        'device_type': request.device_type,
                        'category_filters': request.category_filters,
                        'price_range': request.price_range
                    }
                )
            elif request.item_id is not None:
                return self.candidate_generator.generate_similar(
                    item_id=request.item_id,
                    n=request.n_candidates
                )
        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")
            return []
        
        return []
    
    def _rank_candidates(
        self, 
        request: RecommendationRequest,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Rank candidates using the ranking model."""
        if self.ranker is None:
            logger.warning("No ranker configured, using default scores")
            return candidates
        
        try:
            return self.ranker.rank(
                candidates=candidates,
                user_id=request.user_id,
                context={
                    'device_type': request.device_type,
                    'session_id': request.session_id
                }
            )
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return candidates
    
    def _rerank_candidates(
        self,
        request: RecommendationRequest,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Apply re-ranking strategies."""
        if self.reranker is None:
            logger.debug("No reranker configured")
            return candidates
        
        try:
            rerank_params = {
                'diversity': request.apply_diversity,
                'freshness': request.apply_freshness
            }
            
            return self.reranker.rerank(
                candidates=candidates,
                user_id=request.user_id,
                params=rerank_params
            )
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return candidates
    
    def _apply_business_rules(
        self,
        request: RecommendationRequest,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Apply business rules to candidates."""
        if self.rule_engine is None:
            return candidates
        
        try:
            result = self.rule_engine.execute(
                candidates=candidates,
                user_id=request.user_id,
                context={
                    'device_type': request.device_type,
                    'session_id': request.session_id
                }
            )
            return result.get('candidates', candidates)
        except Exception as e:
            logger.error(f"Business rules failed: {e}")
            return candidates
    
    def _format_results(
        self,
        recommendations: List[Dict],
        request: RecommendationRequest
    ) -> List[Dict]:
        """Format recommendations for output."""
        formatted = []
        for i, rec in enumerate(recommendations):
            formatted_rec = {
                'rank': i + 1,
                'item_id': rec.get('item_id'),
                'score': rec.get('score', 0),
                'rerank_score': rec.get('rerank_score'),
                'reason': rec.get('reason', self._generate_reason(rec, request))
            }
            
            # Add optional fields
            if 'item_name' in rec:
                formatted_rec['item_name'] = rec['item_name']
            if 'price' in rec:
                formatted_rec['price'] = rec['price']
            if 'image_url' in rec:
                formatted_rec['image_url'] = rec['image_url']
            if 'applied_rules' in rec:
                formatted_rec['applied_rules'] = rec['applied_rules']
            
            formatted.append(formatted_rec)
        
        return formatted
    
    def _generate_reason(
        self, 
        recommendation: Dict, 
        request: RecommendationRequest
    ) -> str:
        """Generate a human-readable reason for the recommendation."""
        reasons = []
        
        if recommendation.get('source') == 'collaborative':
            reasons.append("Based on users like you")
        elif recommendation.get('source') == 'popularity':
            reasons.append("Trending now")
        elif recommendation.get('source') == 'similar':
            reasons.append("Similar to your interests")
        
        if recommendation.get('boost_reason'):
            reasons.append(recommendation['boost_reason'])
        
        return "; ".join(reasons) if reasons else "Recommended for you"
    
    def _cache_results(
        self, 
        request: RecommendationRequest, 
        results: List[Dict]
    ) -> None:
        """Cache recommendation results."""
        if not self.cache:
            return
        
        self.cache.set(
            value=results,
            user_id=request.user_id,
            item_id=request.item_id,
            context={
                'n': request.n_recommendations,
                'device_type': request.device_type
            }
        )
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def invalidate_user_cache(self, user_id: int) -> int:
        """Invalidate cached recommendations for a user."""
        if not self.cache:
            return 0
        return self.cache.invalidate(user_id=user_id)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'caching_enabled': self.enable_caching,
            'components': {
                'candidate_generator': self.candidate_generator is not None,
                'ranker': self.ranker is not None,
                'reranker': self.reranker is not None,
                'rule_engine': self.rule_engine is not None
            }
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        if self.session_handler:
            stats['sessions'] = self.session_handler.get_stats()
        
        return stats
