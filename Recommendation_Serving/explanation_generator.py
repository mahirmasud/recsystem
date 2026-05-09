"""
Explanation Generator - Generates human-readable explanations for recommendations.

Provides interpretable reasons for why items were recommended,
supporting transparency and user trust.
"""

from typing import Dict, Any, Optional, List
import logging
from shared.logger import get_logger

logger = get_logger(__name__)


class ExplanationGenerator:
    """
    Generates human-readable explanations for recommendations.
    
    Supports multiple explanation strategies:
    - Collaborative filtering explanations
    - Content-based explanations
    - Popularity-based explanations
    - Business rule explanations
    - Session-based explanations
    """
    
    # Explanation templates
    TEMPLATES = {
        'collaborative': "Users with similar preferences also liked this",
        'collaborative_specific': "Customers who bought {item} also viewed this",
        'content_based': "Similar to items you've shown interest in",
        'content_category': "Matches your interest in {category}",
        'popularity': "Trending now with {metric}",
        'recency': "New arrival in {category}",
        'personalized': "Recommended based on your {signal}",
        'cold_start': "Popular item in your region",
        'business_boost': "Featured selection",
        'diversity': "Explore something different",
        'session_based': "Based on your recent activity"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the explanation generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.include_scores = self.config.get('include_scores', False)
        self.detailed_mode = self.config.get('detailed_mode', False)
        
        logger.info("ExplanationGenerator initialized")
    
    def generate_explanation(
        self,
        recommendation: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an explanation for a single recommendation.
        
        Args:
            recommendation: Recommendation dictionary
            user_context: Optional user context
            
        Returns:
            Human-readable explanation string
        """
        source = recommendation.get('source', 'unknown')
        explanation = self.TEMPLATES.get(source, "Recommended for you")
        
        # Apply template substitutions
        if source == 'collaborative_specific':
            ref_item = recommendation.get('reference_item', 'similar items')
            explanation = explanation.format(item=ref_item)
        
        elif source == 'content_category':
            category = recommendation.get('category', 'this category')
            explanation = explanation.format(category=category)
        
        elif source == 'popularity':
            metric = recommendation.get('popularity_metric', 'many users')
            explanation = explanation.format(metric=metric)
        
        elif source == 'recency':
            category = recommendation.get('category', 'our store')
            explanation = explanation.format(category=category)
        
        elif source == 'personalized':
            signal = recommendation.get('signal', 'preferences')
            explanation = explanation.format(signal=signal)
        
        # Add boost reason if applicable
        boost_reason = recommendation.get('boost_reason')
        if boost_reason:
            explanation = f"{explanation}; {boost_reason}"
        
        return explanation
    
    def generate_batch_explanations(
        self,
        recommendations: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[int, str]:
        """
        Generate explanations for multiple recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            user_context: Optional user context
            
        Returns:
            Dictionary mapping item_id to explanation
        """
        explanations = {}
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            if item_id is not None:
                explanations[item_id] = self.generate_explanation(rec, user_context)
        
        return explanations
    
    def generate_detailed_explanation(
        self,
        recommendation: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed structured explanation.
        
        Args:
            recommendation: Recommendation dictionary
            user_profile: Optional user profile data
            
        Returns:
            Structured explanation dictionary
        """
        explanation = {
            'primary_reason': self.generate_explanation(recommendation),
            'factors': [],
            'confidence': recommendation.get('score', 0),
            'metadata': {}
        }
        
        # Add contributing factors
        if recommendation.get('source') == 'collaborative':
            explanation['factors'].append({
                'type': 'collaborative_signal',
                'weight': recommendation.get('cf_weight', 0.5),
                'description': 'Based on similar user behavior'
            })
        
        if recommendation.get('source') == 'content_based':
            explanation['factors'].append({
                'type': 'content_similarity',
                'weight': recommendation.get('content_weight', 0.5),
                'description': 'Similar to previously interacted items'
            })
        
        if recommendation.get('popularity_score'):
            explanation['factors'].append({
                'type': 'popularity',
                'weight': recommendation.get('popularity_score'),
                'description': 'Popular among all users'
            })
        
        # Add business rule influences
        applied_rules = recommendation.get('applied_rules', [])
        for rule in applied_rules:
            explanation['factors'].append({
                'type': 'business_rule',
                'rule_id': rule.get('rule_id'),
                'description': rule.get('description', 'Business rule applied')
            })
        
        # Add user-specific signals
        if user_profile:
            if recommendation.get('category') in user_profile.get('favorite_categories', []):
                explanation['factors'].append({
                    'type': 'user_preference',
                    'description': 'Matches your favorite categories'
                })
        
        explanation['metadata'] = {
            'item_id': recommendation.get('item_id'),
            'source': recommendation.get('source'),
            'original_score': recommendation.get('score'),
            'rerank_score': recommendation.get('rerank_score')
        }
        
        return explanation
    
    def explain_ranking_change(
        self,
        original_rank: int,
        final_rank: int,
        applied_rules: List[Dict[str, Any]]
    ) -> str:
        """
        Explain why an item's rank changed during re-ranking.
        
        Args:
            original_rank: Original ranking position
            final_rank: Final ranking position
            applied_rules: List of applied business rules
            
        Returns:
            Explanation string
        """
        if original_rank == final_rank:
            return "Rank unchanged"
        
        direction = "promoted" if final_rank < original_rank else "demoted"
        change = abs(original_rank - final_rank)
        
        explanation = f"Item {direction} by {change} positions"
        
        if applied_rules:
            reasons = [rule.get('reason', '') for rule in applied_rules if rule.get('reason')]
            if reasons:
                explanation += f" due to: {', '.join(reasons)}"
        
        return explanation
    
    def generate_comparison_explanation(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any]
    ) -> str:
        """
        Generate explanation comparing two items.
        
        Args:
            item_a: First item recommendation
            item_b: Second item recommendation
            
        Returns:
            Comparison explanation
        """
        score_a = item_a.get('score', 0)
        score_b = item_b.get('score', 0)
        
        if score_a > score_b:
            higher_item = item_a.get('item_id')
            diff = score_a - score_b
        else:
            higher_item = item_b.get('item_id')
            diff = score_b - score_a
        
        return (
            f"Item {higher_item} ranked higher by {diff:.4f} points "
            f"based on personalized relevance"
        )
    
    def get_explanation_templates(self) -> Dict[str, str]:
        """Get available explanation templates."""
        return self.TEMPLATES.copy()
    
    def add_custom_template(
        self, 
        source: str, 
        template: str
    ) -> None:
        """
        Add a custom explanation template.
        
        Args:
            source: Source identifier
            template: Template string (may contain {placeholders})
        """
        self.TEMPLATES[source] = template
        logger.debug(f"Added custom template for source: {source}")
