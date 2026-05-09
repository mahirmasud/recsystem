"""
Context Rules - Applies rules based on contextual information.

Responsible for:
- Time-based rules (hour, day, season)
- Location-based rules
- Device/platform rules
- Session context rules
- Weather/external context rules
"""

from typing import List, Dict, Any
from datetime import datetime, time
from enum import Enum

from shared.logger import get_logger
from .rule_parser import ParsedRule, RuleAction

logger = get_logger(__name__)


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class ContextRules:
    """Applies context-aware rules to recommendations."""
    
    def __init__(self):
        """Initialize ContextRules."""
        self.context_evaluations: List[Dict[str, Any]] = []
        
    def apply(self,
              recommendations: List[Dict[str, Any]],
              rule: ParsedRule,
              user_context: Dict[str, Any],
              item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply context rule to recommendations.
        
        Args:
            recommendations: List of recommendation items
            rule: Parsed context rule
            user_context: User context dictionary
            item_catalog: Item metadata lookup
            
        Returns:
            List of recommendations with context adjustments
        """
        if not recommendations:
            return []
        
        conditions = rule.conditions
        parameters = rule.parameters
        
        # Get current context
        current_time = datetime.now()
        context_info = self._build_context_info(current_time, user_context)
        
        # Check if context matches rule conditions
        context_matches = self._check_context_match(conditions, context_info)
        
        if not context_matches:
            logger.debug(f"Context rule '{rule.id}' context does not match, skipping")
            return recommendations
        
        logger.info(f"Context rule '{rule.id}' matched, applying adjustments")
        
        # Store evaluation
        self.context_evaluations.append({
            'rule_id': rule.id,
            'context_matched': True,
            'timestamp': current_time.isoformat(),
            'context': context_info
        })
        
        # Apply context-based adjustments
        action = rule.action
        
        if action == RuleAction.BOOST_SCORE:
            return self._apply_context_boost(recommendations, parameters, item_catalog, context_info)
        elif action == RuleAction.EXCLUDE:
            return self._apply_context_exclude(recommendations, parameters, item_catalog, context_info)
        elif action == RuleAction.REORDER:
            return self._apply_context_reorder(recommendations, parameters, item_catalog, context_info)
        else:
            return recommendations
    
    def _build_context_info(self, 
                           current_time: datetime, 
                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive context information.
        
        Args:
            current_time: Current datetime
            user_context: User context
            
        Returns:
            Dictionary with all context information
        """
        return {
            'current_time': current_time,
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'day_name': current_time.strftime('%A'),
            'is_weekend': current_time.weekday() >= 5,
            'is_business_hours': 9 <= current_time.hour <= 17,
            'is_morning': 6 <= current_time.hour < 12,
            'is_afternoon': 12 <= current_time.hour < 18,
            'is_evening': 18 <= current_time.hour < 22,
            'is_night': current_time.hour < 6 or current_time.hour >= 22,
            'month': current_time.month,
            'season': self._get_season(current_time.month),
            'device': user_context.get('device', 'unknown'),
            'platform': user_context.get('platform', 'unknown'),
            'location': user_context.get('location', {}),
            'session_duration': user_context.get('session_duration', 0),
            'referrer': user_context.get('referrer', 'direct'),
        }
    
    def _get_season(self, month: int) -> str:
        """
        Get season from month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _check_context_match(self, 
                            conditions: Dict[str, Any], 
                            context_info: Dict[str, Any]) -> bool:
        """
        Check if current context matches rule conditions.
        
        Args:
            conditions: Rule conditions
            context_info: Current context information
            
        Returns:
            True if context matches
        """
        time_range = conditions.get('time_range')
        if time_range:
            if not self._check_time_range(time_range, context_info):
                return False
        
        days_of_week = conditions.get('days_of_week')
        if days_of_week:
            if isinstance(days_of_week, list):
                if context_info['day_of_week'] not in days_of_week:
                    return False
            elif isinstance(days_of_week, str):
                if days_of_week == 'weekend' and not context_info['is_weekend']:
                    return False
                if days_of_week == 'weekday' and context_info['is_weekend']:
                    return False
        
        hours = conditions.get('hours')
        if hours:
            if context_info['hour'] not in hours:
                return False
        
        seasons = conditions.get('seasons')
        if seasons:
            if isinstance(seasons, str):
                seasons = [seasons]
            if context_info['season'] not in seasons:
                return False
        
        devices = conditions.get('devices')
        if devices:
            if isinstance(devices, str):
                devices = [devices]
            if context_info['device'] not in devices:
                return False
        
        platforms = conditions.get('platforms')
        if platforms:
            if isinstance(platforms, str):
                platforms = [platforms]
            if context_info['platform'] not in platforms:
                return False
        
        locations = conditions.get('locations')
        if locations:
            user_location = context_info.get('location', {})
            if not self._check_location_match(user_location, locations):
                return False
        
        return True
    
    def _check_time_range(self, 
                         time_range: Dict[str, Any], 
                         context_info: Dict[str, Any]) -> bool:
        """
        Check if current time is within specified range.
        
        Args:
            time_range: Time range specification
            context_info: Context information
            
        Returns:
            True if time is within range
        """
        start = time_range.get('start')
        end = time_range.get('end')
        
        if start:
            if isinstance(start, str):
                try:
                    start = datetime.fromisoformat(start)
                except ValueError:
                    pass
        
        if end:
            if isinstance(end, str):
                try:
                    end = datetime.fromisoformat(end)
                except ValueError:
                    pass
        
        current = context_info['current_time']
        
        if start and current < start:
            return False
        if end and current > end:
            return False
        
        # Check days of week within time range
        range_days = time_range.get('days_of_week', list(range(7)))
        if current.weekday() not in range_days:
            return False
        
        # Check hours within time range
        range_hours = time_range.get('hours', list(range(24)))
        if current.hour not in range_hours:
            return False
        
        return True
    
    def _check_location_match(self, 
                             user_location: Dict[str, Any], 
                             target_locations: Dict[str, Any]) -> bool:
        """
        Check if user location matches target locations.
        
        Args:
            user_location: User's location info
            target_locations: Target location criteria
            
        Returns:
            True if location matches
        """
        if not user_location:
            return False
        
        # Check country
        if 'country' in target_locations:
            countries = target_locations['country']
            if isinstance(countries, str):
                countries = [countries]
            if user_location.get('country') not in countries:
                return False
        
        # Check region/state
        if 'region' in target_locations:
            regions = target_locations['region']
            if isinstance(regions, str):
                regions = [regions]
            if user_location.get('region') not in regions:
                return False
        
        # Check timezone
        if 'timezone' in target_locations:
            timezones = target_locations['timezone']
            if isinstance(timezones, str):
                timezones = [timezones]
            if user_location.get('timezone') not in timezones:
                return False
        
        return True
    
    def _apply_context_boost(self,
                            recommendations: List[Dict[str, Any]],
                            parameters: Dict[str, Any],
                            item_catalog: Dict[str, Dict[str, Any]],
                            context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based boost."""
        boost_factor = float(parameters.get('boost_factor', 1.5))
        context_attribute = parameters.get('context_attribute')
        
        # Determine which items to boost based on context
        for rec in recommendations:
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            
            should_boost = False
            
            # Boost based on time of day
            if context_info['is_morning'] and item_data.get('morning_product'):
                should_boost = True
            elif context_info['is_evening'] and item_data.get('evening_product'):
                should_boost = True
            elif context_info['is_weekend'] and item_data.get('weekend_product'):
                should_boost = True
            
            # Boost based on season
            item_season = item_data.get('season')
            if item_season and item_season == context_info['season']:
                should_boost = True
            
            # Boost based on device
            if context_info['device'] == 'mobile' and item_data.get('mobile_optimized'):
                should_boost = True
            
            if should_boost:
                current_score = rec.get('score', 0)
                rec['score'] = current_score * boost_factor
                rec['context_boost_applied'] = True
                rec['boost_reason'] = f"context:{context_info['day_name']}_{context_info['season']}"
        
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        return recommendations
    
    def _apply_context_exclude(self,
                              recommendations: List[Dict[str, Any]],
                              parameters: Dict[str, Any],
                              item_catalog: Dict[str, Dict[str, Any]],
                              context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based exclusion."""
        exclude_attribute = parameters.get('exclude_attribute')
        
        if not exclude_attribute:
            return recommendations
        
        filtered = []
        for rec in recommendations:
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            
            # Exclude items not suitable for current context
            if context_info['is_business_hours'] and item_data.get('after_hours_only'):
                continue
            
            if context_info['is_weekend'] and item_data.get('weekday_only'):
                continue
            
            filtered.append(rec)
        
        return filtered
    
    def _apply_context_reorder(self,
                              recommendations: List[Dict[str, Any]],
                              parameters: Dict[str, Any],
                              item_catalog: Dict[str, Dict[str, Any]],
                              context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based reordering."""
        # Reorder based on context preferences
        context_priority = parameters.get('context_priority', {})
        
        if not context_priority:
            return recommendations
        
        # Sort by context-specific priority
        def get_context_priority(rec):
            item_id = rec.get('item_id')
            item_data = item_catalog.get(item_id, {})
            
            priority = 0
            for context_key, boost in context_priority.items():
                if context_info.get(context_key) and item_data.get(context_key):
                    priority += boost
            
            return priority
        
        recommendations.sort(key=lambda x: (-x.get('score', 0), -get_context_priority(x)))
        return recommendations
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context rule evaluations."""
        total = len(self.context_evaluations)
        matched = sum(1 for e in self.context_evaluations if e.get('context_matched'))
        
        return {
            'total_evaluations': total,
            'contexts_matched': matched,
            'match_rate': matched / total if total > 0 else 0
        }
    
    def reset(self):
        """Reset context evaluation state."""
        self.context_evaluations.clear()
