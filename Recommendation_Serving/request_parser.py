"""
Request Parser - Parses and validates recommendation requests.

Handles parsing of CLI and programmatic request inputs,
validating required fields, and normalizing request formats.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationRequest:
    """Structured representation of a recommendation request."""
    
    user_id: Optional[int] = None
    item_id: Optional[int] = None
    n_recommendations: int = 10
    n_candidates: int = 100
    
    # Context information
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    timestamp: Optional[str] = None
    location: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Filters
    category_filters: Optional[List[str]] = None
    price_range: Optional[Dict[str, float]] = None
    brand_filters: Optional[List[str]] = None
    
    # Business rules
    apply_diversity: bool = True
    apply_freshness: bool = True
    apply_business_boost: bool = True
    
    # Metadata
    request_id: Optional[str] = None
    client_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'n_recommendations': self.n_recommendations,
            'n_candidates': self.n_candidates,
            'session_id': self.session_id,
            'device_type': self.device_type,
            'timestamp': self.timestamp,
            'location': self.location,
            'category_filters': self.category_filters,
            'price_range': self.price_range,
            'brand_filters': self.brand_filters,
            'apply_diversity': self.apply_diversity,
            'apply_freshness': self.apply_freshness,
            'apply_business_boost': self.apply_business_boost,
            'request_id': self.request_id,
            'client_info': self.client_info
        }


class RequestParser:
    """
    Parses and validates recommendation requests.
    
    Responsibilities:
    - Parse CLI arguments into structured requests
    - Validate required fields
    - Normalize request formats
    - Apply default values
    - Extract context information
    """
    
    DEFAULT_N_RECOMMENDATIONS = 10
    DEFAULT_N_CANDIDATES = 100
    VALID_DEVICE_TYPES = ['desktop', 'mobile', 'tablet', 'unknown']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the request parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        logger.info("RequestParser initialized")
    
    def parse_user_request(
        self,
        user_id: int,
        n: int = DEFAULT_N_RECOMMENDATIONS,
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationRequest:
        """
        Parse a single-user recommendation request.
        
        Args:
            user_id: The user ID to get recommendations for
            n: Number of recommendations to return
            context: Optional context dictionary
            
        Returns:
            Structured RecommendationRequest
        """
        context = context or {}
        
        request = RecommendationRequest(
            user_id=user_id,
            n_recommendations=n,
            session_id=context.get('session_id'),
            device_type=self._validate_device_type(context.get('device_type')),
            timestamp=context.get('timestamp'),
            location=context.get('location', {}),
            category_filters=context.get('category_filters'),
            price_range=context.get('price_range'),
            brand_filters=context.get('brand_filters'),
            apply_diversity=context.get('apply_diversity', True),
            apply_freshness=context.get('apply_freshness', True),
            apply_business_boost=context.get('apply_business_boost', True),
            client_info=context.get('client_info', {})
        )
        
        logger.info(f"Parsed user request for user_id={user_id}, n={n}")
        return request
    
    def parse_item_request(
        self,
        item_id: int,
        n: int = DEFAULT_N_RECOMMENDATIONS,
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationRequest:
        """
        Parse a similar-items recommendation request.
        
        Args:
            item_id: The item ID to find similar items for
            n: Number of recommendations to return
            context: Optional context dictionary
            
        Returns:
            Structured RecommendationRequest
        """
        context = context or {}
        
        request = RecommendationRequest(
            item_id=item_id,
            n_recommendations=n,
            session_id=context.get('session_id'),
            device_type=self._validate_device_type(context.get('device_type')),
            timestamp=context.get('timestamp'),
            client_info=context.get('client_info', {})
        )
        
        logger.info(f"Parsed item request for item_id={item_id}, n={n}")
        return request
    
    def parse_batch_request(
        self,
        user_ids: List[int],
        n: int = DEFAULT_N_RECOMMENDATIONS,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationRequest]:
        """
        Parse a batch recommendation request.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            context: Optional context dictionary (applied to all users)
            
        Returns:
            List of RecommendationRequest objects
        """
        context = context or {}
        requests = []
        
        for user_id in user_ids:
            request = self.parse_user_request(user_id, n, context)
            requests.append(request)
        
        logger.info(f"Parsed batch request for {len(user_ids)} users")
        return requests
    
    def parse_cli_args(self, args: Dict[str, Any]) -> RecommendationRequest:
        """
        Parse CLI arguments into a recommendation request.
        
        Args:
            args: Dictionary of CLI arguments
            
        Returns:
            Structured RecommendationRequest
        """
        context = {
            'session_id': args.get('session_id'),
            'device_type': args.get('device_type'),
            'category_filters': args.get('categories'),
            'price_range': args.get('price_range'),
            'apply_diversity': args.get('no_diversity', True) is not True,
            'apply_freshness': args.get('no_freshness', True) is not True,
            'apply_business_boost': args.get('no_business_boost', True) is not True
        }
        
        if args.get('user_id'):
            return self.parse_user_request(
                user_id=args['user_id'],
                n=args.get('n', self.DEFAULT_N_RECOMMENDATIONS),
                context=context
            )
        elif args.get('item_id'):
            return self.parse_item_request(
                item_id=args['item_id'],
                n=args.get('n', self.DEFAULT_N_RECOMMENDATIONS),
                context=context
            )
        else:
            raise ValueError("Either user_id or item_id must be provided")
    
    def _validate_device_type(self, device_type: Optional[str]) -> str:
        """
        Validate and normalize device type.
        
        Args:
            device_type: Device type string
            
        Returns:
            Normalized device type
        """
        if not device_type:
            return 'unknown'
        
        device_type = device_type.lower()
        if device_type in self.VALID_DEVICE_TYPES:
            return device_type
        
        logger.warning(f"Unknown device type '{device_type}', defaulting to 'unknown'")
        return 'unknown'
    
    def validate_request(self, request: RecommendationRequest) -> bool:
        """
        Validate a recommendation request.
        
        Args:
            request: Request to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if request.user_id is None and request.item_id is None:
            raise ValueError("Either user_id or item_id must be provided")
        
        if request.n_recommendations <= 0:
            raise ValueError("n_recommendations must be positive")
        
        if request.n_candidates < request.n_recommendations:
            raise ValueError("n_candidates must be >= n_recommendations")
        
        if request.price_range:
            min_price = request.price_range.get('min', 0)
            max_price = request.price_range.get('max', float('inf'))
            if min_price > max_price:
                raise ValueError("price_range min cannot be greater than max")
        
        return True
