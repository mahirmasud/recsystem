"""
Cache Handler - Caching layer for recommendations.

Provides in-memory caching with TTL support, cache statistics,
and cache invalidation for recommendation results.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import hashlib
from shared.logger import get_logger

logger = get_logger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() > self.expires_at
    
    def remaining_ttl(self) -> float:
        """Get remaining time-to-live in seconds."""
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)


class CacheHandler:
    """
    Manages caching of recommendation results.
    
    Features:
    - TTL-based expiration
    - Hit/miss tracking
    - LRU-style eviction
    - User-specific cache invalidation
    - Cache statistics
    """
    
    def __init__(
        self, 
        ttl_seconds: int = 3600,
        max_entries: int = 1000
    ):
        """
        Initialize the cache handler.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            max_entries: Maximum number of entries before eviction
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"CacheHandler initialized with TTL={ttl_seconds}s, max_entries={max_entries}")
    
    def _generate_key(self, user_id: Optional[int] = None, item_id: Optional[int] = None,
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key.
        
        Args:
            user_id: User ID
            item_id: Item ID
            context: Additional context
            
        Returns:
            Unique cache key string
        """
        key_parts = []
        if user_id is not None:
            key_parts.append(f"user:{user_id}")
        if item_id is not None:
            key_parts.append(f"item:{item_id}")
        if context:
            # Sort context keys for consistent hashing
            context_str = str(sorted(context.items()))
            key_parts.append(f"ctx:{hashlib.md5(context_str.encode()).hexdigest()[:8]}")
        
        return ":".join(key_parts) if key_parts else "default"
    
    def get(
        self, 
        user_id: Optional[int] = None, 
        item_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get cached recommendation results.
        
        Args:
            user_id: User ID
            item_id: Item ID
            context: Additional context
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._generate_key(user_id, item_id, context)
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                self._hits += 1
                logger.debug(f"Cache hit for key={key}")
                return entry.value
            else:
                # Remove expired entry
                del self._cache[key]
                logger.debug(f"Cache entry expired for key={key}")
        
        self._misses += 1
        logger.debug(f"Cache miss for key={key}")
        return None
    
    def set(
        self, 
        value: Any,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_ttl: Optional[int] = None
    ) -> str:
        """
        Store recommendation results in cache.
        
        Args:
            value: Value to cache
            user_id: User ID
            item_id: Item ID
            context: Additional context
            custom_ttl: Custom TTL in seconds (optional)
            
        Returns:
            Cache key
        """
        key = self._generate_key(user_id, item_id, context)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()
        
        ttl = custom_ttl or self.ttl_seconds
        self._cache[key] = CacheEntry(value, ttl)
        logger.debug(f"Cached value for key={key}, TTL={ttl}s")
        
        return key
    
    def invalidate(self, user_id: Optional[int] = None, item_id: Optional[int] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            user_id: User ID to invalidate (optional)
            item_id: Item ID to invalidate (optional)
            
        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        
        for key in self._cache:
            if user_id is not None and f"user:{user_id}" in key:
                keys_to_remove.append(key)
            elif item_id is not None and f"item:{item_id}" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
        return len(keys_to_remove)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry (LRU-style)."""
        if not self._cache:
            return
        
        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_key]
        self._evictions += 1
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_entries,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(hit_rate, 4),
            'evictions': self._evictions,
            'avg_ttl_remaining': self._avg_ttl_remaining()
        }
    
    def _avg_ttl_remaining(self) -> float:
        """Calculate average remaining TTL across entries."""
        if not self._cache:
            return 0
        
        total_remaining = sum(
            entry.remaining_ttl() for entry in self._cache.values()
        )
        return round(total_remaining / len(self._cache), 2)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
