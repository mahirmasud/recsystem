"""
Campaign Rules - Manages time-bound promotional campaigns.

Responsible for:
- Campaign activation/deactivation
- Priority-based campaign handling
- Campaign-specific boosting
- Limited-time offers
- A/B test campaign support
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from shared.logger import get_logger
from .rule_parser import ParsedRule, RuleAction, RuleType

logger = get_logger(__name__)


class Campaign:
    """Represents a single marketing campaign."""
    
    def __init__(self, 
                 campaign_id: str,
                 name: str,
                 start_time: datetime,
                 end_time: datetime,
                 priority: int = 0,
                 enabled: bool = True,
                 target_items: Optional[List[str]] = None,
                 target_categories: Optional[List[str]] = None,
                 boost_factor: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a campaign.
        
        Args:
            campaign_id: Unique campaign identifier
            name: Campaign name
            start_time: Campaign start time
            end_time: Campaign end time
            priority: Campaign priority (higher = more important)
            enabled: Whether campaign is active
            target_items: List of target item IDs
            target_categories: List of target category IDs
            boost_factor: Score boost multiplier
            metadata: Additional campaign metadata
        """
        self.campaign_id = campaign_id
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.enabled = enabled
        self.target_items = set(target_items) if target_items else set()
        self.target_categories = set(target_categories) if target_categories else set()
        self.boost_factor = boost_factor
        self.metadata = metadata or {}
        
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if campaign is currently active.
        
        Args:
            current_time: Time to check (defaults to now)
            
        Returns:
            True if campaign is active
        """
        if not self.enabled:
            return False
        
        now = current_time or datetime.now()
        return self.start_time <= now <= self.end_time
    
    def matches_item(self, item_id: str, item_data: Dict[str, Any]) -> bool:
        """
        Check if an item matches this campaign.
        
        Args:
            item_id: Item ID to check
            item_data: Item metadata
            
        Returns:
            True if item matches campaign criteria
        """
        # If no targets specified, match all
        if not self.target_items and not self.target_categories:
            return True
        
        # Check specific items
        if self.target_items and item_id in self.target_items:
            return True
        
        # Check categories
        if self.target_categories:
            item_category = item_data.get('category_id') or item_data.get('category')
            if item_category in self.target_categories:
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert campaign to dictionary."""
        return {
            'campaign_id': self.campaign_id,
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'priority': self.priority,
            'enabled': self.enabled,
            'target_items': list(self.target_items),
            'target_categories': list(self.target_categories),
            'boost_factor': self.boost_factor,
            'metadata': self.metadata
        }


class CampaignRules:
    """Manages and applies campaign rules to recommendations."""
    
    def __init__(self):
        """Initialize CampaignRules."""
        self.campaigns: Dict[str, Campaign] = {}
        self.campaign_history: List[Dict[str, Any]] = []
        
    def add_campaign(self, campaign: Campaign):
        """
        Add a campaign.
        
        Args:
            campaign: Campaign object to add
        """
        self.campaigns[campaign.campaign_id] = campaign
        logger.info(f"Added campaign: {campaign.name} ({campaign.campaign_id})")
    
    def remove_campaign(self, campaign_id: str):
        """
        Remove a campaign.
        
        Args:
            campaign_id: ID of campaign to remove
        """
        if campaign_id in self.campaigns:
            del self.campaigns[campaign_id]
            logger.info(f"Removed campaign: {campaign_id}")
    
    def load_from_rule(self, rule: ParsedRule):
        """
        Load campaign from parsed rule.
        
        Args:
            rule: Parsed rule with campaign configuration
        """
        params = rule.parameters
        
        # Parse time range
        start_time = params.get('start_time')
        end_time = params.get('end_time')
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        if not start_time:
            start_time = datetime.now()
        if not end_time:
            end_time = datetime.now() + timedelta(days=7)  # Default 7 days
        
        campaign = Campaign(
            campaign_id=rule.id,
            name=rule.name,
            start_time=start_time,
            end_time=end_time,
            priority=rule.priority,
            enabled=rule.enabled,
            target_items=params.get('target_items'),
            target_categories=params.get('target_categories'),
            boost_factor=params.get('boost_factor', 1.5),
            metadata=rule.metadata
        )
        
        self.add_campaign(campaign)
    
    def apply(self,
              recommendations: List[Dict[str, Any]],
              rule: ParsedRule,
              user_context: Dict[str, Any],
              item_catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply campaign rules to recommendations.
        
        Args:
            recommendations: List of recommendation items
            rule: Parsed rule
            user_context: User context
            item_catalog: Item catalog
            
        Returns:
            List of recommendations with campaign adjustments
        """
        if not recommendations:
            return []
        
        current_time = datetime.now()
        
        # Get active campaigns sorted by priority
        active_campaigns = [
            c for c in self.campaigns.values()
            if c.is_active(current_time)
        ]
        active_campaigns.sort(key=lambda c: c.priority, reverse=True)
        
        if not active_campaigns:
            logger.debug("No active campaigns")
            return recommendations
        
        logger.info(f"Applying {len(active_campaigns)} active campaigns")
        
        # Track which items have been boosted by which campaign
        item_campaigns: Dict[str, List[str]] = {}
        
        # Apply campaigns in priority order
        for campaign in active_campaigns:
            boosted_count = 0
            
            for rec in recommendations:
                item_id = rec.get('item_id')
                item_data = item_catalog.get(item_id, {})
                
                if not campaign.matches_item(item_id, item_data):
                    continue
                
                # Skip if already boosted by higher priority campaign
                if item_id in item_campaigns:
                    continue
                
                # Apply campaign boost
                current_score = rec.get('score', 0)
                new_score = current_score * campaign.boost_factor
                
                rec['score'] = new_score
                rec['campaign_applied'] = campaign.campaign_id
                rec['campaign_name'] = campaign.name
                rec['original_score'] = current_score
                rec['boost_reason'] = f"campaign:{campaign.name}"
                
                if item_id not in item_campaigns:
                    item_campaigns[item_id] = []
                item_campaigns[item_id].append(campaign.campaign_id)
                
                boosted_count += 1
                
                # Log campaign application
                self.campaign_history.append({
                    'campaign_id': campaign.campaign_id,
                    'item_id': item_id,
                    'timestamp': current_time.isoformat(),
                    'score_before': current_score,
                    'score_after': new_score
                })
            
            logger.debug(f"Campaign '{campaign.name}' boosted {boosted_count} items")
        
        # Re-sort by score
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return recommendations
    
    def get_active_campaigns(self, 
                            current_time: Optional[datetime] = None) -> List[Campaign]:
        """
        Get all currently active campaigns.
        
        Args:
            current_time: Time to check
            
        Returns:
            List of active campaigns
        """
        now = current_time or datetime.now()
        return [c for c in self.campaigns.values() if c.is_active(now)]
    
    def get_campaign_stats(self) -> Dict[str, Any]:
        """
        Get statistics about campaign applications.
        
        Returns:
            Dictionary with campaign statistics
        """
        total_campaigns = len(self.campaigns)
        active_campaigns = len(self.get_active_campaigns())
        
        # Count applications per campaign
        campaign_counts = {}
        for entry in self.campaign_history:
            cid = entry['campaign_id']
            campaign_counts[cid] = campaign_counts.get(cid, 0) + 1
        
        return {
            'total_campaigns': total_campaigns,
            'active_campaigns': active_campaigns,
            'total_applications': len(self.campaign_history),
            'applications_by_campaign': campaign_counts,
            'recent_applications': self.campaign_history[-20:]
        }
    
    def cleanup_expired(self):
        """Remove expired campaigns from memory."""
        current_time = datetime.now()
        expired = [
            cid for cid, campaign in self.campaigns.items()
            if campaign.end_time < current_time
        ]
        
        for cid in expired:
            self.remove_campaign(cid)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired campaigns")
        
        return len(expired)
    
    def reset(self):
        """Reset campaign state."""
        self.campaign_history.clear()
