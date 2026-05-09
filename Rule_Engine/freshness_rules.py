"""Freshness Rules - Rules for content freshness."""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

class FreshnessRules:
    """Collection of freshness rules."""
    
    @staticmethod
    def recent_content(item: Dict, context: Dict, days: int = 30) -> bool:
        """Check if content is recent."""
        created = item.get('created_date')
        if not created:
            return True
        return (datetime.now() - created).days <= days
    
    @staticmethod
    def boost_fresh(candidates: List[Dict], context: Dict, 
                    freshness_boost: float = 1.1) -> List[Dict]:
        """Boost fresh content."""
        for candidate in candidates:
            days_old = (datetime.now() - candidate.get('created_date', datetime.now())).days
            if days_old < 7:
                candidate['score'] *= freshness_boost * (1 + (7 - days_old) / 30)
        return candidates
    
    @staticmethod
    def decay_stale(candidates: List[Dict], context: Dict,
                    stale_days: int = 90, decay_rate: float = 0.9) -> List[Dict]:
        """Decay stale content scores."""
        for candidate in candidates:
            days_old = (datetime.now() - candidate.get('created_date', datetime.now())).days
            if days_old > stale_days:
                decay_factor = decay_rate ** ((days_old - stale_days) / 30)
                candidate['score'] *= decay_factor
        return candidates
