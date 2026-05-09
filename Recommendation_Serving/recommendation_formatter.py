"""
Recommendation Formatter - Formats recommendation results for output.

Handles formatting recommendations into various output formats
including JSON, CSV, and Parquet structures.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from shared.logger import get_logger

logger = get_logger(__name__)


class RecommendationFormatter:
    """
    Formats recommendation results for various output formats.
    
    Supports:
    - JSON format
    - CSV format
    - Parquet structure
    - Human-readable format
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the formatter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.include_metadata = self.config.get('include_metadata', True)
        self.include_scores = self.config.get('include_scores', True)
        self.include_reasons = self.config.get('include_reasons', True)
        
        logger.info("RecommendationFormatter initialized")
    
    def format_json(
        self,
        recommendations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format recommendations as JSON-compatible dictionary.
        
        Args:
            recommendations: List of recommendation dictionaries
            metadata: Optional metadata dictionary
            
        Returns:
            JSON-compatible dictionary
        """
        result = {
            'generated_at': datetime.now().isoformat(),
            'recommendations': self._format_recommendations_list(recommendations),
            'count': len(recommendations)
        }
        
        if metadata and self.include_metadata:
            result['metadata'] = metadata
        
        return result
    
    def format_csv_rows(
        self,
        recommendations: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Format recommendations as CSV rows.
        
        Args:
            recommendations: List of recommendation dictionaries
            user_id: Optional user ID to include in each row
            
        Returns:
            List of dictionaries suitable for CSV writing
        """
        rows = []
        for rec in recommendations:
            row = {
                'user_id': user_id,
                'rank': rec.get('rank', 0),
                'item_id': rec.get('item_id'),
                'score': rec.get('score', 0),
                'reason': rec.get('reason', '')
            }
            
            # Add optional fields if present
            if 'item_name' in rec:
                row['item_name'] = rec['item_name']
            if 'price' in rec:
                row['price'] = rec['price']
            if 'category' in rec:
                row['category'] = rec['category']
            if 'rerank_score' in rec:
                row['rerank_score'] = rec['rerank_score']
            
            rows.append(row)
        
        return rows
    
    def format_parquet_record(
        self,
        recommendations: List[Dict[str, Any]],
        user_id: int,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Format recommendations as a parquet-compatible record.
        
        Args:
            recommendations: List of recommendation dictionaries
            user_id: User ID
            request_id: Request identifier
            
        Returns:
            Dictionary suitable for parquet conversion
        """
        return {
            'request_id': [request_id] * len(recommendations),
            'user_id': [user_id] * len(recommendations),
            'generated_at': [datetime.now().isoformat()] * len(recommendations),
            'rank': [rec.get('rank', 0) for rec in recommendations],
            'item_id': [rec.get('item_id') for rec in recommendations],
            'score': [rec.get('score', 0) for rec in recommendations],
            'rerank_score': [rec.get('rerank_score', 0) for rec in recommendations],
            'reason': [rec.get('reason', '') for rec in recommendations],
            'item_name': [rec.get('item_name', '') for rec in recommendations],
            'price': [rec.get('price', 0) for rec in recommendations],
            'category': [rec.get('category', '') for rec in recommendations]
        }
    
    def format_human_readable(
        self,
        recommendations: List[Dict[str, Any]],
        title: str = "Recommendations"
    ) -> str:
        """
        Format recommendations as human-readable text.
        
        Args:
            recommendations: List of recommendation dictionaries
            title: Title for the output
            
        Returns:
            Formatted string
        """
        lines = [f"\n{'='*60}", f"{title}", f"{'='*60}\n"]
        
        for rec in recommendations:
            rank = rec.get('rank', 0)
            item_id = rec.get('item_id', 'Unknown')
            item_name = rec.get('item_name', f'Item {item_id}')
            score = rec.get('score', 0)
            reason = rec.get('reason', '')
            price = rec.get('price')
            
            line = f"{rank}. {item_name} (ID: {item_id})"
            if price is not None:
                line += f" - ${price:.2f}"
            line += f"\n   Score: {score:.4f}"
            if reason:
                line += f"\n   Reason: {reason}"
            
            lines.append(line)
            lines.append("")
        
        lines.append(f"{'='*60}")
        lines.append(f"Total: {len(recommendations)} recommendations\n")
        
        return "\n".join(lines)
    
    def _format_recommendations_list(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format recommendations list with selected fields."""
        formatted = []
        
        for rec in recommendations:
            item = {
                'item_id': rec.get('item_id'),
                'rank': rec.get('rank', 0)
            }
            
            if self.include_scores:
                item['score'] = rec.get('score', 0)
                if 'rerank_score' in rec:
                    item['rerank_score'] = rec['rerank_score']
            
            if self.include_reasons and 'reason' in rec:
                item['reason'] = rec['reason']
            
            # Add display fields
            if 'item_name' in rec:
                item['item_name'] = rec['item_name']
            if 'price' in rec:
                item['price'] = rec['price']
            if 'image_url' in rec:
                item['image_url'] = rec['image_url']
            
            # Add business rule info
            if 'applied_rules' in rec:
                item['applied_rules'] = rec['applied_rules']
            
            formatted.append(item)
        
        return formatted
    
    def format_minimal(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Extract minimal item IDs from recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            List of item IDs
        """
        return [rec.get('item_id') for rec in recommendations]
    
    def format_with_explanations(
        self,
        recommendations: List[Dict[str, Any]],
        explanations: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """
        Format recommendations with custom explanations.
        
        Args:
            recommendations: List of recommendation dictionaries
            explanations: Dictionary mapping item_id to explanation
            
        Returns:
            Formatted recommendations with explanations
        """
        formatted = []
        
        for rec in recommendations:
            item_id = rec.get('item_id')
            item = rec.copy()
            
            # Override reason with custom explanation if available
            if item_id in explanations:
                item['explanation'] = explanations[item_id]
            
            formatted.append(item)
        
        return formatted
