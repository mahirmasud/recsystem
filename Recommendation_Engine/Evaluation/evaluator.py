"""
Recommendation Evaluator - Unified evaluation framework

Combines all metrics into a comprehensive evaluation pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Set
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .precision import PrecisionAtK
from .recall import RecallAtK
from .map_metric import MAPMetric
from .ndcg import NDCGMetric
from .ctr import CTRSimulator
from .diversity_score import DiversityScorer
from .coverage import CoverageCalculator

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    
    Aggregates all evaluation metrics:
    - Ranking quality (Precision, Recall, MAP, NDCG)
    - Business metrics (CTR simulation)
    - Diversity metrics
    - Coverage metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration with metric parameters
        """
        self.config = config
        
        # Initialize metrics
        k_values = config.get('k_values', [5, 10, 20])
        self.k = config.get('default_k', 10)
        
        self.precision = {k: PrecisionAtK(k) for k in k_values}
        self.recall = {k: RecallAtK(k) for k in k_values}
        self.map_metric = {k: MAPMetric(k) for k in k_values}
        self.ndcg = {k: NDCGMetric(k) for k in k_values}
        
        self.ctr_simulator = CTRSimulator()
        self.diversity_scorer = DiversityScorer(
            category_column=config.get('category_column', 'category_id')
        )
        self.coverage_calculator = CoverageCalculator()
        
    def evaluate(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, List[int]],
        items_df: Optional[pd.DataFrame] = None,
        include_distribution: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            recommendations: Dict of {user_id: [item_ids]}
            ground_truth: Dict of {user_id: [relevant_item_ids]}
            items_df: Optional DataFrame with item metadata
            include_distribution: Whether to include distribution stats
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Align users
        common_users = set(recommendations.keys()) & set(ground_truth.keys())
        
        if len(common_users) == 0:
            logger.warning("No common users between recommendations and ground truth")
            return {'error': 'No common users'}
        
        # Convert to lists
        recs_list = [recommendations[u] for u in common_users]
        truth_list = [ground_truth[u] for u in common_users]
        
        results = {
            'num_users': len(common_users),
            'ranking_metrics': {},
            'diversity_metrics': {},
            'coverage_metrics': {}
        }
        
        # Ranking metrics by K
        for k in self.precision.keys():
            results['ranking_metrics'][f'precision@{k}'] = self.precision[k].compute_batch(recs_list, truth_list)['mean_precision']
            results['ranking_metrics'][f'recall@{k}'] = self.recall[k].compute_batch(recs_list, truth_list)['mean_recall']
            results['ranking_metrics'][f'map@{k}'] = self.map_metric[k].compute(recs_list, truth_list)
            results['ranking_metrics'][f'ndcg@{k}'] = self.ndcg[k].compute_batch(recs_list, truth_list)['mean_ndcg']
        
        # Diversity metrics
        if items_df is not None:
            # Convert recommendations to dict format
            recs_with_items = []
            for user_recs in recs_list:
                user_rec_dicts = []
                for item_id in user_recs[:self.k]:
                    item_row = items_df[items_df['item_id'] == item_id]
                    if len(item_row) > 0:
                        user_rec_dicts.append(item_row.iloc[0].to_dict())
                recs_with_items.append(user_rec_dicts)
            
            diversity_results = self.diversity_scorer.compute_batch(recs_with_items, self.k)
            results['diversity_metrics'] = {
                'category_diversity': diversity_results['mean_category_diversity'],
                'intra_list_distance': diversity_results['mean_ild']
            }
        
        # Coverage metrics
        all_items = set()
        for truth in truth_list:
            all_items.update(truth)
        
        coverage_results = self.coverage_calculator.compute_batch(recs_list, all_items, self.k)
        results['coverage_metrics'] = {
            'catalog_coverage': coverage_results['catalog_coverage'],
            'aggregate_diversity': coverage_results['aggregate_diversity']
        }
        
        if include_distribution:
            results['coverage_metrics']['distribution'] = coverage_results['distribution']
        
        return results
    
    def evaluate_from_scores(
        self,
        items: List[int],
        scores: np.ndarray,
        relevance: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Evaluate from item scores and relevance judgments.
        
        Args:
            items: List of item IDs
            scores: Predicted scores
            relevance: Dict of {item_id: relevance_score}
            
        Returns:
            Dictionary with metrics
        """
        sorted_indices = np.argsort(-scores)
        sorted_items = [items[i] for i in sorted_indices]
        
        results = {}
        
        for k, metric in self.precision.items():
            results[f'precision@{k}'] = metric.compute(sorted_items, list(relevance.keys()))
        
        for k, metric in self.recall.items():
            results[f'recall@{k}'] = metric.compute(sorted_items, list(relevance.keys()))
        
        for k, metric in self.ndcg.items():
            results[f'ndcg@{k}'] = metric.compute(sorted_items, relevance)
        
        return results
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[int, List[int]]],
        ground_truth: Dict[int, List[int]]
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            model_results: Dict of {model_name: {user_id: [item_ids]}}
            ground_truth: Ground truth recommendations
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, recommendations in model_results.items():
            results = self.evaluate(recommendations, ground_truth, include_distribution=False)
            
            row = {'model': model_name}
            row.update(results.get('ranking_metrics', {}))
            row.update(results.get('diversity_metrics', {}))
            row.update(results.get('coverage_metrics', {}))
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def save_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save evaluation report to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted summary string
        """
        lines = ["=" * 60, "RECOMMENDATION EVALUATION SUMMARY", "=" * 60, ""]
        
        lines.append(f"Users evaluated: {results.get('num_users', 'N/A')}")
        lines.append("")
        
        lines.append("RANKING METRICS:")
        lines.append("-" * 40)
        ranking = results.get('ranking_metrics', {})
        for metric, value in ranking.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
        lines.append("")
        
        lines.append("DIVERSITY METRICS:")
        lines.append("-" * 40)
        diversity = results.get('diversity_metrics', {})
        for metric, value in diversity.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
        lines.append("")
        
        lines.append("COVERAGE METRICS:")
        lines.append("-" * 40)
        coverage = results.get('coverage_metrics', {})
        for metric, value in coverage.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
