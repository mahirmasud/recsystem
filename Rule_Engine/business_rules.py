"""Business Rules - Custom business-specific rules."""
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class BusinessRules:
    """Collection of custom business rules."""
    
    @staticmethod
    def inventory_priority(item: Dict, context: Dict) -> bool:
        """Prioritize items with high inventory."""
        return item.get('inventory_level', 0) > context.get('min_inventory', 0)
    
    @staticmethod
    def seasonal_promotion(item: Dict, context: Dict) -> bool:
        """Apply seasonal promotion boosts."""
        season = context.get('season', '')
        item_seasons = item.get('promotional_seasons', [])
        return season in item_seasons
    
    @staticmethod
    def bundle_eligible(item: Dict, context: Dict) -> bool:
        """Check bundle eligibility."""
        return item.get('bundle_eligible', True)
    
    @staticmethod
    def loyalty_tier_bonus(item: Dict, context: Dict) -> float:
        """Calculate loyalty tier bonus multiplier."""
        tier = context.get('loyalty_tier', 'bronze')
        multipliers = {'bronze': 1.0, 'silver': 1.1, 'gold': 1.2, 'platinum': 1.3}
        return multipliers.get(tier, 1.0)
