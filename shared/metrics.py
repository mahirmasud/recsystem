"""
Metrics calculation utilities for the recommendation system.
Common metric computations for evaluation and reporting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from shared.logger import get_logger


logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculator for common metrics used in recommendation systems.
    
    Supports:
    - Ranking metrics (Precision, Recall, NDCG, MAP)
    - Business metrics (Revenue, Conversion Rate)
    - Diversity metrics
    - Coverage metrics
    """
    
    @staticmethod
    def precision_at_k(
        relevant: List[int],
        recommended: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant: List of relevant item IDs
            recommended: List of recommended item IDs
            k: Number of recommendations to consider
        
        Returns:
            Precision score (0-1)
        """
        recommended_k = recommended[:k]
        hits = len(set(relevant) & set(recommended_k))
        return hits / len(recommended_k) if recommended_k else 0.0
    
    @staticmethod
    def recall_at_k(
        relevant: List[int],
        recommended: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant: List of relevant item IDs
            recommended: List of recommended item IDs
            k: Number of recommendations to consider
        
        Returns:
            Recall score (0-1)
        """
        recommended_k = recommended[:k]
        hits = len(set(relevant) & set(recommended_k))
        return hits / len(relevant) if relevant else 0.0
    
    @staticmethod
    def dcg_at_k(
        relevance: List[float],
        k: int = 10
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.
        
        Args:
            relevance: List of relevance scores
            k: Number of items to consider
        
        Returns:
            DCG score
        """
        relevance = relevance[:k]
        gains = np.array(relevance)
        discounts = np.log2(np.arange(len(gains)) + 2)
        return np.sum(gains / discounts)
    
    @staticmethod
    def ndcg_at_k(
        relevance: List[float],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        Args:
            relevance: List of relevance scores
            k: Number of items to consider
        
        Returns:
            NDCG score (0-1)
        """
        dcg = MetricsCalculator.dcg_at_k(relevance, k)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = MetricsCalculator.dcg_at_k(ideal_relevance, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(
        relevant_items: List[List[int]],
        recommended_items: List[List[int]],
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision at K.
        
        Args:
            relevant_items: List of relevant item lists (one per user)
            recommended_items: List of recommended item lists (one per user)
            k: Number of recommendations to consider
        
        Returns:
            MAP score (0-1)
        """
        ap_scores = []
        
        for relevant, recommended in zip(relevant_items, recommended_items):
            ap = MetricsCalculator.average_precision(relevant, recommended, k)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def average_precision(
        relevant: List[int],
        recommended: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Average Precision.
        
        Args:
            relevant: List of relevant item IDs
            recommended: List of recommended item IDs
            k: Number of recommendations to consider
        
        Returns:
            Average Precision score
        """
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant_set)
    
    @staticmethod
    def hit_rate(
        relevant_items: List[List[int]],
        recommended_items: List[List[int]],
        k: int = 10
    ) -> float:
        """
        Calculate Hit Rate (proportion of users with at least one hit).
        
        Args:
            relevant_items: List of relevant item lists
            recommended_items: List of recommended item lists
            k: Number of recommendations to consider
        
        Returns:
            Hit Rate score (0-1)
        """
        hits = 0
        
        for relevant, recommended in zip(relevant_items, recommended_items):
            recommended_k = recommended[:k]
            if set(relevant) & set(recommended_k):
                hits += 1
        
        return hits / len(relevant_items) if relevant_items else 0.0
    
    @staticmethod
    def diversity_score(
        recommendations: List[int],
        categories: Dict[int, int]
    ) -> float:
        """
        Calculate diversity score based on category distribution.
        
        Args:
            recommendations: List of recommended item IDs
            categories: Mapping of item_id to category_id
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not recommendations:
            return 0.0
        
        rec_categories = [categories.get(item_id, -1) for item_id in recommendations]
        unique_categories = len(set(rec_categories))
        max_categories = len(recommendations)
        
        return unique_categories / max_categories if max_categories > 0 else 0.0
    
    @staticmethod
    def coverage_score(
        all_recommendations: List[List[int]],
        total_items: int
    ) -> float:
        """
        Calculate catalog coverage (proportion of items ever recommended).
        
        Args:
            all_recommendations: All recommendations across users
            total_items: Total number of items in catalog
        
        Returns:
            Coverage score (0-1)
        """
        all_recommended_items = set()
        for recs in all_recommendations:
            all_recommended_items.update(recs)
        
        return len(all_recommended_items) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def calculate_lift(
        treatment_metric: float,
        control_metric: float
    ) -> float:
        """
        Calculate lift percentage.
        
        Args:
            treatment_metric: Metric value for treatment group
            control_metric: Metric value for control group
        
        Returns:
            Lift percentage
        """
        if control_metric == 0:
            return 0.0
        return ((treatment_metric - control_metric) / control_metric) * 100
    
    @staticmethod
    def calculate_roas(
        revenue: float,
        ad_spend: float
    ) -> float:
        """
        Calculate Return on Ad Spend.
        
        Args:
            revenue: Revenue generated
            ad_spend: Advertising spend
        
        Returns:
            ROAS ratio
        """
        return revenue / ad_spend if ad_spend > 0 else 0.0
    
    @staticmethod
    def calculate_conversion_rate(
        conversions: int,
        impressions: int
    ) -> float:
        """
        Calculate conversion rate.
        
        Args:
            conversions: Number of conversions
            impressions: Number of impressions
        
        Returns:
            Conversion rate (0-1)
        """
        return conversions / impressions if impressions > 0 else 0.0
    
    @staticmethod
    def calculate_ctr(
        clicks: int,
        impressions: int
    ) -> float:
        """
        Calculate Click-Through Rate.
        
        Args:
            clicks: Number of clicks
            impressions: Number of impressions
        
        Returns:
            CTR (0-1)
        """
        return clicks / impressions if impressions > 0 else 0.0
    
    @staticmethod
    def calculate_aov(
        total_revenue: float,
        total_orders: int
    ) -> float:
        """
        Calculate Average Order Value.
        
        Args:
            total_revenue: Total revenue
            total_orders: Total number of orders
        
        Returns:
            Average Order Value
        """
        return total_revenue / total_orders if total_orders > 0 else 0.0
    
    @staticmethod
    def calculate_clv(
        avg_order_value: float,
        purchase_frequency: float,
        customer_lifespan: float
    ) -> float:
        """
        Calculate Customer Lifetime Value.
        
        Args:
            avg_order_value: Average order value
            purchase_frequency: Purchases per period
            customer_lifespan: Customer lifespan in periods
        
        Returns:
            Customer Lifetime Value
        """
        return avg_order_value * purchase_frequency * customer_lifespan
