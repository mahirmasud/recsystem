"""Block Rules - Rules for blocking specific items."""
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

class BlockRules:
    """Collection of block rules."""
    
    @staticmethod
    def previously_purchased(item: Dict, context: Dict) -> bool:
        """Block already purchased items (for non-repeat products)."""
        purchased = context.get('purchased_items', [])
        return item['item_id'] in purchased and not item.get('is_repeatable', True)
    
    @staticmethod
    def low_rating(item: Dict, context: Dict) -> bool:
        """Block low-rated items."""
        min_rating = context.get('min_acceptable_rating', 3.0)
        return item.get('rating', 5.0) < min_rating
    
    @staticmethod
    def inappropriate_content(item: Dict, context: Dict) -> bool:
        """Block inappropriate content."""
        flagged = context.get('flagged_content', [])
        return item.get('content_id') in flagged
    
    @staticmethod
    def competitor_brand(item: Dict, context: Dict) -> bool:
        """Block competitor brands if specified."""
        blocked_brands = context.get('blocked_brands', [])
        return item.get('brand') in blocked_brands
