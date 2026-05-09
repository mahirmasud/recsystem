"""
Score Adjuster - Normalizes and adjusts scores after rule application.

Responsible for:
- Score normalization (0-1 range)
- Score distribution adjustment
- Preventing score inflation
- Maintaining relative ranking integrity
- Applying global score modifiers
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from shared.logger import get_logger
from shared.exceptions import RuleEngineError

logger = get_logger(__name__)


class ScoreAdjuster:
    """Adjusts and normalizes recommendation scores after rule application."""
    
    def __init__(self, 
                 target_min: float = 0.0,
                 target_max: float = 1.0,
                 preserve_order: bool = True):
        """
        Initialize ScoreAdjuster.
        
        Args:
            target_min: Minimum target score value
            target_max: Maximum target score value
            preserve_order: Whether to preserve relative ordering
        """
        self.target_min = target_min
        self.target_max = target_max
        self.preserve_order = preserve_order
        self.adjustment_history: List[Dict[str, Any]] = []
        
    def normalize(self, 
                  recommendations: List[Dict[str, Any]],
                  method: str = 'minmax') -> List[Dict[str, Any]]:
        """
        Normalize scores to target range.
        
        Args:
            recommendations: List of recommendations with scores
            method: Normalization method ('minmax', 'zscore', 'sigmoid')
            
        Returns:
            List of recommendations with normalized scores
        """
        if not recommendations:
            return []
        
        # Extract scores
        scores = np.array([r.get('score', 0) for r in recommendations]).reshape(-1, 1)
        
        if method == 'minmax':
            normalized = self._minmax_normalize(scores)
        elif method == 'zscore':
            normalized = self._zscore_normalize(scores)
        elif method == 'sigmoid':
            normalized = self._sigmoid_normalize(scores)
        else:
            logger.warning(f"Unknown normalization method '{method}', using minmax")
            normalized = self._minmax_normalize(scores)
        
        # Apply normalized scores
        for i, rec in enumerate(recommendations):
            original_score = rec.get('score', 0)
            rec['normalized_score'] = float(normalized[i][0])
            rec['original_score_pre_normalization'] = original_score
        
        logger.info(f"Normalized {len(recommendations)} scores using {method} method")
        
        self.adjustment_history.append({
            'method': method,
            'count': len(recommendations),
            'score_range_before': (float(scores.min()), float(scores.max())),
            'score_range_after': (float(normalized.min()), float(normalized.max()))
        })
        
        return recommendations
    
    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalization to target range.
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores
        """
        if scores.max() == scores.min():
            # All scores are the same
            return np.full_like(scores, (self.target_min + self.target_max) / 2)
        
        scaler = MinMaxScaler(feature_range=(self.target_min, self.target_max))
        return scaler.fit_transform(scores)
    
    def _zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-score normalization followed by scaling to target range.
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores
        """
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return np.full_like(scores, (self.target_min + self.target_max) / 2)
        
        # Z-score
        z_scores = (scores - mean) / std
        
        # Scale to target range
        z_min = z_scores.min()
        z_max = z_scores.max()
        
        if z_max == z_min:
            return np.full_like(scores, (self.target_min + self.target_max) / 2)
        
        scaled = (z_scores - z_min) / (z_max - z_min)
        return scaled * (self.target_max - self.target_min) + self.target_min
    
    def _sigmoid_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Sigmoid normalization for handling outliers.
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores
        """
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return np.full_like(scores, (self.target_min + self.target_max) / 2)
        
        # Sigmoid transformation
        z = (scores - mean) / std
        sigmoid = 1 / (1 + np.exp(-z))
        
        # Scale to target range
        return sigmoid * (self.target_max - self.target_min) + self.target_min
    
    def apply_cap(self, 
                  recommendations: List[Dict[str, Any]],
                  cap_value: float) -> List[Dict[str, Any]]:
        """
        Apply upper cap to scores.
        
        Args:
            recommendations: List of recommendations
            cap_value: Maximum score value
            
        Returns:
            List with capped scores
        """
        capped_count = 0
        
        for rec in recommendations:
            current_score = rec.get('score', 0)
            if current_score > cap_value:
                rec['score'] = cap_value
                rec['score_capped'] = True
                capped_count += 1
        
        logger.info(f"Capped {capped_count} scores at {cap_value}")
        return recommendations
    
    def apply_floor(self,
                    recommendations: List[Dict[str, Any]],
                    floor_value: float) -> List[Dict[str, Any]]:
        """
        Apply lower floor to scores.
        
        Args:
            recommendations: List of recommendations
            floor_value: Minimum score value
            
        Returns:
            List with floored scores
        """
        floored_count = 0
        
        for rec in recommendations:
            current_score = rec.get('score', 0)
            if current_score < floor_value:
                rec['score'] = floor_value
                rec['score_floored'] = True
                floored_count += 1
        
        logger.info(f"Floored {floored_count} scores at {floor_value}")
        return recommendations
    
    def adjust_for_boost_inflation(self,
                                   recommendations: List[Dict[str, Any]],
                                   max_total_boost: float = 5.0) -> List[Dict[str, Any]]:
        """
        Adjust scores to prevent excessive boost inflation.
        
        Args:
            recommendations: List of recommendations
            max_total_boost: Maximum allowed total boost factor
            
        Returns:
            List with adjusted scores
        """
        for rec in recommendations:
            original_score = rec.get('original_score', rec.get('score', 0))
            current_score = rec.get('score', 0)
            
            if original_score > 0:
                effective_boost = current_score / original_score
                
                if effective_boost > max_total_boost:
                    # Cap the boost
                    rec['score'] = original_score * max_total_boost
                    rec['boost_capped'] = True
                    rec['effective_boost'] = max_total_boost
                else:
                    rec['effective_boost'] = effective_boost
        
        # Re-sort after adjustment
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Applied boost inflation control with max boost {max_total_boost}")
        return recommendations
    
    def blend_scores(self,
                     recommendations: List[Dict[str, Any]],
                     primary_weight: float = 0.7,
                     secondary_field: str = 'ml_score') -> List[Dict[str, Any]]:
        """
        Blend current scores with another score field.
        
        Args:
            recommendations: List of recommendations
            primary_weight: Weight for current score (0-1)
            secondary_field: Field name for secondary score
            
        Returns:
            List with blended scores
        """
        secondary_weight = 1.0 - primary_weight
        
        for rec in recommendations:
            primary_score = rec.get('score', 0)
            secondary_score = rec.get(secondary_field, primary_score)
            
            blended = (primary_score * primary_weight) + (secondary_score * secondary_weight)
            rec['score'] = blended
            rec['blended_score'] = True
            rec['score_components'] = {
                'primary': primary_score,
                'secondary': secondary_score,
                'primary_weight': primary_weight,
                'secondary_weight': secondary_weight
            }
        
        # Re-sort after blending
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Blended scores with weight {primary_weight} on primary")
        return recommendations
    
    def get_adjustment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about score adjustments.
        
        Returns:
            Dictionary with adjustment statistics
        """
        if not self.adjustment_history:
            return {'total_adjustments': 0}
        
        return {
            'total_adjustments': len(self.adjustment_history),
            'adjustment_history': self.adjustment_history[-10:],
            'methods_used': list(set(a['method'] for a in self.adjustment_history))
        }
    
    def reset(self):
        """Reset adjustment state."""
        self.adjustment_history.clear()
