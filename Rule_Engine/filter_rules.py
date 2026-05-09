"""Filter Rules - Rules for filtering out unwanted items."""
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class FilterRules:
    """Collection of filter rules."""
    
    @staticmethod
    def in_stock(item: Dict, context: Dict) -> bool:
        """Filter out of stock items."""
        return item.get('in_stock', True)
    
    @staticmethod
    def age_appropriate(item: Dict, context: Dict) -> bool:
        """Filter based on user age."""
        user_age = context.get('user_age', 999)
        min_age = item.get('min_age', 0)
        return user_age >= min_age
    
    @staticmethod
    def region_available(item: Dict, context: Dict) -> bool:
        """Filter based on region availability."""
        user_region = context.get('region', '')
        available_regions = item.get('available_regions', [])
        return not available_regions or user_region in available_regions
    
    @staticmethod
    def price_range(item: Dict, context: Dict) -> bool:
        """Filter by price range preference."""
        max_price = context.get('max_price', float('inf'))
        return item.get('price', 0) <= max_price
