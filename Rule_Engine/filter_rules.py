"""
Filter Rules - Excludes items from recommendations based on conditions.

Responsible for:
- Filtering out-of-stock items
- Filtering previously purchased items
- Filtering blocked categories/brands
- Filtering low-rated items
- Applying attribute-based exclusion rules
"""

from typing import List, Dict, Any, Set
import pandas as pd

from shared.logger import get_logger
from .rule_parser import ParsedRule, RuleAction

logger = get_logger(__name__)


class FilterRules:
    """Applies filtering rules to remove items from recommendation lists."""
    
    def __init__(self):
        """Initialize FilterRules."""
        self.filtered_items: Set[str] = set()
        self.filter_reasons: Dict[str, str] = {}
        
    def apply(self, 
              recommendations: List[Dict[str, Any]], 
              rule: ParsedRule,
              user_context: Dict[str, Any],
              item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply filter rule to recommendations.
        
        Args:
            recommendations: List of recommendation items
            rule: Parsed filter rule
            user_context: User context dictionary
            item_catalog: Item metadata lookup
            
        Returns:
            Filtered list of recommendations
        """
        if not recommendations:
            return []
        
        original_count = len(recommendations)
        filtered = []
        
        conditions = rule.conditions
        parameters = rule.parameters
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            should_filter = False
            filter_reason = None
            
            # Check item attributes from catalog
            item_data = item_catalog.get(item_id, {})
            
            # Condition: Out of stock
            if conditions.get('out_of_stock', False):
                in_stock = item_data.get('in_stock')
                stock_qty = item_data.get('stock_quantity')
                
                # Filter if explicitly marked as out of stock OR if stock_quantity is present and <= 0
                if in_stock is False:
                    should_filter = True
                    filter_reason = "out_of_stock"
                elif stock_qty is not None and stock_qty <= 0:
                    should_filter = True
                    filter_reason = "out_of_stock"
            
            # Condition: Previously purchased - only check if explicitly enabled
            if conditions.get('exclude_purchased', False) and not should_filter:
                purchased_items = user_context.get('purchased_items', set())
                if isinstance(purchased_items, list):
                    purchased_items = set(purchased_items)
                if item_id in purchased_items:
                    should_filter = True
                    filter_reason = "previously_purchased"
            
            # Condition: Blocked categories - only check if explicitly provided
            if 'blocked_categories' in conditions and not should_filter:
                blocked_cats = set(conditions['blocked_categories'])
                if blocked_cats:  # Only filter if there are actually blocked categories
                    item_category = item_data.get('category_id') or item_data.get('category')
                    if item_category in blocked_cats:
                        should_filter = True
                        filter_reason = f"blocked_category:{item_category}"
            
            # Condition: Blocked brands - only check if explicitly provided
            if 'blocked_brands' in conditions and not should_filter:
                blocked_brands = set(conditions['blocked_brands'])
                if blocked_brands:  # Only filter if there are actually blocked brands
                    item_brand = item_data.get('brand_id') or item_data.get('brand')
                    if item_brand in blocked_brands:
                        should_filter = True
                        filter_reason = f"blocked_brand:{item_brand}"
            
            # Condition: Low rating threshold - only check if explicitly provided
            if 'min_rating' in conditions and not should_filter:
                min_rating = float(conditions['min_rating'])
                item_rating = item_data.get('rating', 0) or item_data.get('avg_rating', 0)
                if item_rating < min_rating:
                    should_filter = True
                    filter_reason = f"low_rating:{item_rating}"
            
            # Condition: Attribute-based filtering - only check if explicitly provided
            if 'item_attributes' in conditions and not should_filter:
                if self._matches_attribute_filters(item_data, conditions['item_attributes']):
                    should_filter = True
                    filter_reason = "attribute_filter"
            
            # Condition: Price range exclusion - only check if explicitly provided
            if 'price_range' in conditions and not should_filter:
                price_range = conditions['price_range']
                item_price = item_data.get('price', 0)
                min_price = price_range.get('min', float('-inf'))
                max_price = price_range.get('max', float('inf'))
                if item_price < min_price or item_price > max_price:
                    should_filter = True
                    filter_reason = f"price_out_of_range:{item_price}"
            
            if should_filter:
                self.filtered_items.add(item_id)
                self.filter_reasons[item_id] = filter_reason
                logger.debug(f"Filtered item {item_id}: {filter_reason}")
            else:
                filtered.append(rec)
        
        filtered_count = len(filtered)
        removed_count = original_count - filtered_count
        
        logger.info(f"Filter rule '{rule.id}' removed {removed_count}/{original_count} items")
        
        return filtered
    
    def _matches_attribute_filters(self, 
                                   item_data: Dict[str, Any], 
                                   attribute_filters: Dict[str, Any]) -> bool:
        """
        Check if item matches attribute filters (for exclusion).
        
        Args:
            item_data: Item attributes
            attribute_filters: Filter conditions
            
        Returns:
            True if item should be filtered out
        """
        for attr, condition in attribute_filters.items():
            item_value = item_data.get(attr)
            
            if item_value is None:
                continue
            
            # Handle different operators
            if isinstance(condition, dict):
                if 'eq' in condition and item_value != condition['eq']:
                    return False
                if 'neq' in condition and item_value == condition['neq']:
                    return True
                if 'in' in condition and item_value not in condition['in']:
                    return False
                if 'notin' in condition and item_value in condition['notin']:
                    return True
                if 'gte' in condition and item_value < condition['gte']:
                    return False
                if 'lte' in condition and item_value > condition['lte']:
                    return False
            elif isinstance(condition, list):
                if item_value not in condition:
                    return False
            else:
                if item_value != condition:
                    return False
        
        return True
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Get statistics about filtered items.
        
        Returns:
            Dictionary with filter statistics
        """
        reason_counts = {}
        for reason in self.filter_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            'total_filtered': len(self.filtered_items),
            'filtered_items': list(self.filtered_items),
            'reasons': reason_counts
        }
    
    def reset(self):
        """Reset filter state."""
        self.filtered_items.clear()
        self.filter_reasons.clear()
