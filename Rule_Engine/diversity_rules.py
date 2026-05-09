"""Diversity Rules - Rules for ensuring recommendation diversity."""
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)

class DiversityRules:
    """Collection of diversity rules."""
    
    @staticmethod
    def category_diversity(candidates: List[Dict], context: Dict, 
                           max_per_category: int = 5) -> List[Dict]:
        """Limit items per category."""
        result = []
        category_counts = {}
        
        for candidate in candidates:
            cat = candidate.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            if category_counts[cat] <= max_per_category:
                result.append(candidate)
        
        return result
    
    @staticmethod
    def brand_diversity(candidates: List[Dict], context: Dict,
                        max_per_brand: int = 3) -> List[Dict]:
        """Limit items per brand."""
        result = []
        brand_counts = {}
        
        for candidate in candidates:
            brand = candidate.get('brand', 'unknown')
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            if brand_counts[brand] <= max_per_brand:
                result.append(candidate)
        
        return result
    
    @staticmethod
    def price_diversity(candidates: List[Dict], context: Dict,
                        n_price_buckets: int = 3) -> List[Dict]:
        """Ensure price range diversity."""
        if not candidates:
            return []
        
        prices = [c.get('price', 0) for c in candidates]
        min_p, max_p = min(prices), max(prices)
        bucket_size = (max_p - min_p) / n_price_buckets + 1e-6
        
        result = []
        bucket_counts = {}
        
        for candidate in candidates:
            price = candidate.get('price', 0)
            bucket = int((price - min_p) / bucket_size)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            
            if bucket_counts[bucket] <= len(candidates) // n_price_buckets + 2:
                result.append(candidate)
        
        return result
