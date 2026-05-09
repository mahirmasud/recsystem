"""Boost Rules - Rules for boosting certain items."""
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

class BoostRules:
    """Collection of boost rules."""
    
    @staticmethod
    def high_margin(item: Dict, context: Dict) -> bool:
        """Boost high margin items."""
        return item.get('margin', 0) > 0.3
    
    @staticmethod
    def trending(item: Dict, context: Dict) -> bool:
        """Boost trending items."""
        return item.get('is_trending', False)
    
    @staticmethod
    def new_arrival(item: Dict, context: Dict) -> bool:
        """Boost new arrivals."""
        return item.get('is_new', False)
    
    @staticmethod
    def user_preference_match(item: Dict, context: Dict) -> bool:
        """Boost items matching user preferences."""
        pref_categories = context.get('preferred_categories', [])
        return item.get('category') in pref_categories
    
    @staticmethod
    def on_sale(item: Dict, context: Dict) -> bool:
        """Boost items on sale."""
        return item.get('on_sale', False)
