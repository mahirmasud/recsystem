"""A/B Testing - Experiment framework for recommendations."""
from typing import Dict, Any, List, Optional
import hashlib
import logging
logger = logging.getLogger(__name__)

class ABTesting:
    """A/B testing framework for recommendation strategies."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments = {}
        self.assignments = {}
        
    def create_experiment(self, name: str, variants: Dict[str, float], 
                          traffic_pct: float = 1.0):
        """Create an A/B test experiment."""
        self.experiments[name] = {
            'variants': variants,
            'traffic_pct': traffic_pct,
            'metrics': {'impressions': {}, 'clicks': {}, 'conversions': {}}
        }
        logger.info(f"Created experiment {name} with variants: {list(variants.keys())}")
    
    def get_variant(self, experiment_name: str, user_id: int) -> str:
        """Get variant assignment for a user."""
        key = f"{experiment_name}_{user_id}"
        if key in self.assignments:
            return self.assignments[key]
        
        # Hash-based assignment
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100
        
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return 'control'
        
        cumulative = 0
        for variant, pct in experiment['variants'].items():
            cumulative += pct * 100
            if hash_val < cumulative:
                self.assignments[key] = variant
                return variant
        
        self.assignments[key] = 'control'
        return 'control'
    
    def track_impression(self, experiment: str, variant: str, user_id: int):
        """Track recommendation impression."""
        if experiment in self.experiments:
            metrics = self.experiments[experiment]['metrics']
            metrics['impressions'][variant] = metrics['impressions'].get(variant, 0) + 1
    
    def track_click(self, experiment: str, variant: str, user_id: int):
        """Track recommendation click."""
        if experiment in self.experiments:
            metrics = self.experiments[experiment]['metrics']
            metrics['clicks'][variant] = metrics['clicks'].get(variant, 0) + 1
    
    def get_results(self, experiment: str) -> Dict[str, Any]:
        """Get experiment results."""
        exp = self.experiments.get(experiment)
        if not exp:
            return {}
        
        metrics = exp['metrics']
        results = {}
        for variant in exp['variants'].keys():
            impressions = metrics['impressions'].get(variant, 0)
            clicks = metrics['clicks'].get(variant, 0)
            results[variant] = {
                'impressions': impressions,
                'clicks': clicks,
                'ctr': clicks / impressions if impressions > 0 else 0
            }
        return results
