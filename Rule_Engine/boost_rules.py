"""
Boost Rules - Adjusts recommendation scores based on business criteria.

Responsible for:
- Boosting high-margin products
- Boosting trending items
- Boosting new arrivals
- Boosting strategic categories
- Applying score multipliers and additive boosts
"""

from typing import List, Dict, Any
import math

from shared.logger import get_logger
from .rule_parser import ParsedRule, RuleAction

logger = get_logger(__name__)


class BoostRules:
    """Applies boosting rules to adjust recommendation scores."""
    
    def __init__(self):
        """Initialize BoostRules."""
        self.boost_history: List[Dict[str, Any]] = []
        
    def apply(self,
              recommendations: List[Dict[str, Any]],
              rule: ParsedRule,
              user_context: Dict[str, Any],
              item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply boost rule to recommendations.
        
        Args:
            recommendations: List of recommendation items with scores
            rule: Parsed boost rule
            user_context: User context dictionary
            item_catalog: Item metadata lookup
            
        Returns:
            List of recommendations with adjusted scores
        """
        if not recommendations:
            return []
        
        original_scores = {r['item_id']: r.get('score', 0) for r in recommendations}
        boosted_count = 0
        
        conditions = rule.conditions
        parameters = rule.parameters
        
        # Determine boost strategy
        boost_type = parameters.get('boost_type', 'multiplicative')
        boost_factor = float(parameters.get('boost_factor', 1.5))
        max_boost = float(parameters.get('max_boost', 10.0))
        min_score = float(parameters.get('min_score', 0.0))
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            current_score = rec.get('score', 0)
            
            # Check item attributes from catalog
            item_data = item_catalog.get(item_id, {})
            
            should_boost = False
            boost_reason = None
            actual_boost = 0
            
            # Condition: High margin boost
            if conditions.get('high_margin', False):
                margin = item_data.get('profit_margin', 0) or item_data.get('margin_percent', 0)
                threshold = conditions.get('margin_threshold', 0.3)
                if margin >= threshold:
                    should_boost = True
                    boost_reason = f"high_margin:{margin:.2f}"
                    actual_boost = boost_factor
            
            # Condition: Trending items boost
            if conditions.get('trending', False) and not should_boost:
                is_trending = item_data.get('is_trending', False)
                trend_score = item_data.get('trend_score', 0)
                trend_threshold = conditions.get('trend_threshold', 0.7)
                if is_trending or trend_score >= trend_threshold:
                    should_boost = True
                    boost_reason = f"trending:{trend_score:.2f}"
                    actual_boost = boost_factor
            
            # Condition: New arrivals boost
            if conditions.get('new_arrival', False) and not should_boost:
                days_since_launch = item_data.get('days_since_launch', 999)
                max_days = conditions.get('max_days_new', 30)
                if days_since_launch <= max_days:
                    should_boost = True
                    boost_reason = f"new_arrival:{days_since_launch}d"
                    actual_boost = boost_factor
            
            # Condition: Strategic category boost
            if conditions.get('strategic_categories') and not should_boost:
                strategic_cats = set(conditions['strategic_categories'])
                item_category = item_data.get('category_id') or item_data.get('category')
                if item_category in strategic_cats:
                    should_boost = True
                    boost_reason = f"strategic_category:{item_category}"
                    actual_boost = boost_factor
            
            # Condition: User affinity boost
            if conditions.get('user_affinity', False) and not should_boost:
                user_prefs = user_context.get('preferred_categories', [])
                user_brands = user_context.get('preferred_brands', [])
                item_category = item_data.get('category_id') or item_data.get('category')
                item_brand = item_data.get('brand_id') or item_data.get('brand')
                
                if item_category in user_prefs or item_brand in user_brands:
                    should_boost = True
                    boost_reason = "user_affinity"
                    actual_boost = boost_factor
            
            # Condition: Inventory clearance boost
            if conditions.get('clearance', False) and not should_boost:
                stock_level = item_data.get('stock_quantity', 0)
                clearance_threshold = conditions.get('clearance_threshold', 100)
                if stock_level > clearance_threshold:
                    should_boost = True
                    boost_reason = f"clearance:{stock_level}"
                    actual_boost = boost_factor
            
            # Apply boost if conditions met
            if should_boost:
                if boost_type == 'multiplicative':
                    new_score = current_score * actual_boost
                elif boost_type == 'additive':
                    new_score = current_score + actual_boost
                elif boost_type == 'logarithmic':
                    new_score = current_score + math.log(1 + actual_boost)
                else:
                    new_score = current_score * actual_boost
                
                # Cap the boost
                new_score = min(new_score, current_score + max_boost)
                new_score = max(new_score, min_score)
                
                rec['score'] = new_score
                rec['original_score'] = current_score
                rec['boost_applied'] = actual_boost
                rec['boost_reason'] = boost_reason
                
                boosted_count += 1
                
                self.boost_history.append({
                    'item_id': item_id,
                    'original_score': current_score,
                    'new_score': new_score,
                    'boost_factor': actual_boost,
                    'reason': boost_reason,
                    'rule_id': rule.id
                })
                
                logger.debug(f"Boosted item {item_id}: {current_score:.4f} -> {new_score:.4f} ({boost_reason})")
        
        logger.info(f"Boost rule '{rule.id}' applied to {boosted_count}/{len(recommendations)} items")
        
        # Re-sort by score descending
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return recommendations
    
    def get_boost_stats(self) -> Dict[str, Any]:
        """
        Get statistics about applied boosts.
        
        Returns:
            Dictionary with boost statistics
        """
        if not self.boost_history:
            return {'total_boosted': 0}
        
        total_boosted = len(self.boost_history)
        avg_boost = sum(b['boost_factor'] for b in self.boost_history) / total_boosted
        reasons = {}
        for b in self.boost_history:
            reason = b.get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            'total_boosted': total_boosted,
            'average_boost_factor': avg_boost,
            'boosts_by_reason': reasons,
            'boost_details': self.boost_history[-10:]  # Last 10 boosts
        }
    
    def reset(self):
        """Reset boost state."""
        self.boost_history.clear()
