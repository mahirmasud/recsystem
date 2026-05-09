"""Cache Manager - Caching layer for recommendations."""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of recommendations."""
    def __init__(self, config: Dict[str, Any], ttl_seconds: int = 300):
        self.config = config
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache = {}
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expires']:
                self.hits += 1
                return entry['value']
            del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Cache a value."""
        self.cache[key] = {'value': value, 'expires': datetime.now() + self.ttl}
    
    def invalidate(self, user_id: int):
        """Invalidate cache for a user."""
        keys_to_remove = [k for k in self.cache if k.startswith(f'user_{user_id}')]
        for k in keys_to_remove:
            del self.cache[k]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for user {user_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0
        }
