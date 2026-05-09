"""
CTR Simulator - Click-Through Rate simulation and estimation

Simulates and calculates CTR metrics for recommendation evaluation.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CTRSimulator:
    """
    Simulate and calculate Click-Through Rate (CTR) metrics.
    
    Features:
    - CTR calculation from impressions and clicks
    - Position-bias aware CTR estimation
    - A/B test simulation
    - CTR prediction calibration
    """
    
    def __init__(self, position_bias: Optional[Dict[int, float]] = None):
        """
        Initialize CTR simulator.
        
        Args:
            position_bias: Optional dict of {position: bias_factor}
        """
        self.position_bias = position_bias or self._default_position_bias()
        
    def _default_position_bias(self) -> Dict[int, float]:
        """
        Create default position bias factors based on typical user behavior.
        
        Returns:
            Dictionary of position -> bias factor
        """
        # Typical examination probability by position
        return {
            1: 1.0,
            2: 0.85,
            3: 0.75,
            4: 0.65,
            5: 0.55,
            6: 0.45,
            7: 0.40,
            8: 0.35,
            9: 0.30,
            10: 0.25
        }
    
    def compute_ctr(
        self,
        impressions: int,
        clicks: int
    ) -> float:
        """
        Calculate raw CTR.
        
        Args:
            impressions: Number of times items were shown
            clicks: Number of clicks received
            
        Returns:
            CTR as a decimal (0 to 1)
        """
        if impressions == 0:
            return 0.0
        return clicks / impressions
    
    def compute_position_adjusted_ctr(
        self,
        recommendations: List[int],
        clicks: List[int],
        position_column: Optional[str] = None
    ) -> float:
        """
        Calculate CTR adjusted for position bias.
        
        Args:
            recommendations: List of recommended item IDs
            clicks: List of click indicators (0 or 1) for each position
            position_column: Optional column name if using DataFrame
            
        Returns:
            Position-adjusted CTR
        """
        if len(recommendations) == 0 or len(clicks) == 0:
            return 0.0
        
        total_weighted_clicks = 0.0
        total_weighted_impressions = 0.0
        
        for i, (item_id, clicked) in enumerate(zip(recommendations, clicks)):
            position = i + 1
            bias = self.position_bias.get(position, 0.2)
            
            total_weighted_clicks += clicked / bias if bias > 0 else 0
            total_weighted_impressions += 1.0 / bias if bias > 0 else 0
        
        if total_weighted_impressions == 0:
            return 0.0
        
        return total_weighted_clicks / total_weighted_impressions
    
    def simulate_clicks(
        self,
        scores: np.ndarray,
        true_relevance: np.ndarray,
        n_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        Simulate click behavior based on scores and true relevance.
        
        Args:
            scores: Predicted relevance scores
            true_relevance: True relevance values
            n_simulations: Number of simulation runs
            
        Returns:
            Dictionary with simulated CTR statistics
        """
        if len(scores) != len(true_relevance):
            raise ValueError("Scores and relevance must have same length")
        
        ctr_values = []
        
        for _ in range(n_simulations):
            # Simulate clicks based on relevance and position
            sorted_indices = np.argsort(-scores)
            clicks = []
            
            for i, idx in enumerate(sorted_indices[:10]):  # Top 10 positions
                position = i + 1
                position_factor = self.position_bias.get(position, 0.2)
                
                # Click probability = relevance * position_bias
                click_prob = true_relevance[idx] * position_factor
                click = 1 if np.random.random() < click_prob else 0
                clicks.append(click)
            
            ctr = sum(clicks) / len(clicks)
            ctr_values.append(ctr)
        
        return {
            'mean_simulated_ctr': np.mean(ctr_values),
            'std_simulated_ctr': np.std(ctr_values),
            'min_simulated_ctr': np.min(ctr_values),
            'max_simulated_ctr': np.max(ctr_values)
        }
    
    def evaluate_recommendations(
        self,
        recs_df: pd.DataFrame,
        click_column: str = 'clicked',
        score_column: str = 'ranking_score'
    ) -> Dict[str, float]:
        """
        Evaluate CTR from recommendation results.
        
        Args:
            recs_df: DataFrame with recommendations and click data
            click_column: Name of click indicator column
            score_column: Name of score column
            
        Returns:
            Dictionary with CTR metrics
        """
        if click_column not in recs_df.columns:
            logger.warning(f"Click column '{click_column}' not found")
            return {'ctr': 0.0}
        
        total_impressions = len(recs_df)
        total_clicks = recs_df[click_column].sum()
        
        raw_ctr = self.compute_ctr(total_impressions, total_clicks)
        
        # Calculate CTR by score decile
        recs_df = recs_df.copy()
        recs_df['score_decile'] = pd.qcut(recs_df[score_column].rank(method='first'), 10, labels=False, duplicates='drop')
        
        ctr_by_decile = recs_df.groupby('score_decile')[click_column].mean().to_dict()
        
        return {
            'ctr': raw_ctr,
            'total_impressions': total_impressions,
            'total_clicks': int(total_clicks),
            'ctr_by_score_decile': ctr_by_decile
        }
    
    def ab_test_analysis(
        self,
        control_clicks: int,
        control_impressions: int,
        treatment_clicks: int,
        treatment_impressions: int
    ) -> Dict[str, Any]:
        """
        Analyze A/B test results for CTR comparison.
        
        Args:
            control_clicks: Clicks in control group
            control_impressions: Impressions in control group
            treatment_clicks: Clicks in treatment group
            treatment_impressions: Impressions in treatment group
            
        Returns:
            Dictionary with A/B test analysis results
        """
        control_ctr = self.compute_ctr(control_impressions, control_clicks)
        treatment_ctr = self.compute_ctr(treatment_impressions, treatment_clicks)
        
        # Relative lift
        if control_ctr > 0:
            relative_lift = (treatment_ctr - control_ctr) / control_ctr
        else:
            relative_lift = float('inf') if treatment_ctr > 0 else 0.0
        
        # Simple z-test approximation
        p1, n1 = treatment_ctr, treatment_impressions
        p2, n2 = control_ctr, control_impressions
        
        if n1 > 0 and n2 > 0:
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                z_score = (p1 - p2) / se if se > 0 else 0
                # Approximate p-value (two-tailed)
                p_value = 2 * (1 - min(0.9999, 0.5 * (1 + np.erf(abs(z_score) / np.sqrt(2)))))
            else:
                z_score = 0
                p_value = 1.0
        else:
            z_score = 0
            p_value = 1.0
        
        return {
            'control_ctr': control_ctr,
            'treatment_ctr': treatment_ctr,
            'absolute_lift': treatment_ctr - control_ctr,
            'relative_lift': relative_lift,
            'z_score': z_score,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05
        }
