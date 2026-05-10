"""
Primitive Registry - Custom Featuretools primitives for multi-domain feature engineering.

Registers custom aggregation and transformation primitives that work across domains.
"""

import featuretools as ft
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RecencyScorePrimitive(AggregationPrimitive):
    """
    Calculates recency score based on time since last event.
    
    Lower recency (more recent) = higher score
    """
    name = "recency_score"
    input_types = [ColumnSchema(semantic_tags={'datetime'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def __init__(self, reference_date=None):
        self.reference_date = reference_date or pd.Timestamp.now()
        super().__init__()
    
    def get_function(self):
        def recency_score(timestamps):
            if len(timestamps) == 0:
                return 0.0
            max_date = timestamps.max()
            days_since = (self.reference_date - max_date).days
            max_possible_days = 365  # Normalize to 1 year
            score = max(0, 1 - (days_since / max_possible_days))
            return score
        return recency_score


class FrequencyScorePrimitive(AggregationPrimitive):
    """
    Calculates frequency score based on event count.
    """
    name = "frequency_score"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def frequency_score(values):
            if len(values) == 0:
                return 0.0
            count = len(values)
            # Log scale to handle wide range
            return np.log1p(count)
        return frequency_score


class TrendVelocityPrimitive(AggregationPrimitive):
    """
    Calculates trend velocity (rate of change over time).
    """
    name = "trend_velocity"
    input_types = [ColumnSchema(semantic_tags={'numeric'}), ColumnSchema(semantic_tags={'datetime'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def trend_velocity(values, timestamps):
            if len(values) < 2 or len(timestamps) < 2:
                return 0.0
            
            # Sort by timestamp
            sorted_idx = np.argsort(timestamps)
            sorted_values = values[sorted_idx]
            sorted_times = timestamps[sorted_idx]
            
            # Calculate time span in days
            time_span = (sorted_times[-1] - sorted_times[0]).total_seconds() / 86400
            if time_span == 0:
                return 0.0
            
            # Calculate value change rate
            value_change = sorted_values[-1] - sorted_values[0]
            velocity = value_change / time_span
            
            return velocity
        return trend_velocity


class EntityAffinityPrimitive(AggregationPrimitive):
    """
    Calculates affinity score for categorical entities.
    """
    name = "entity_affinity"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def entity_affinity(categories):
            if len(categories) == 0:
                return 0.0
            
            # Calculate entropy-based affinity
            unique, counts = np.unique(categories, return_counts=True)
            proportions = counts / len(categories)
            
            # Shannon entropy
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(len(unique)) if len(unique) > 1 else 1
            
            # Normalize: lower entropy = higher affinity
            if max_entropy == 0:
                return 1.0
            affinity = 1 - (entropy / max_entropy)
            
            return affinity
        return entity_affinity


class InteractionDensityPrimitive(AggregationPrimitive):
    """
    Calculates interaction density (events per time unit).
    """
    name = "interaction_density"
    input_types = [ColumnSchema(semantic_tags={'datetime'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def interaction_density(timestamps):
            if len(timestamps) < 2:
                return float(len(timestamps))
            
            time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # hours
            if time_span == 0:
                return float(len(timestamps))
            
            density = len(timestamps) / time_span
            return density
        return interaction_density


class PercentileRankPrimitive(TransformPrimitive):
    """
    Calculates percentile rank of a value within its group.
    """
    name = "percentile_rank"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def percentile_rank(values):
            if len(values) == 0:
                return pd.Series([], dtype=float)
            
            ranks = pd.Series(values).rank(pct=True)
            return ranks
        return percentile_rank


class RollingMeanPrimitive(AggregationPrimitive):
    """
    Calculates rolling mean over a window.
    """
    name = "rolling_mean"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        super().__init__()
    
    def get_function(self):
        def rolling_mean(values):
            if len(values) < self.window_size:
                return np.mean(values) if len(values) > 0 else 0.0
            
            return np.mean(values[-self.window_size:])
        return rolling_mean


class EntropyPrimitive(AggregationPrimitive):
    """
    Calculates Shannon entropy of categorical distribution.
    """
    name = "entropy"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def entropy(categories):
            if len(categories) == 0:
                return 0.0
            
            unique, counts = np.unique(categories, return_counts=True)
            proportions = counts / len(categories)
            
            # Shannon entropy
            entropy_value = -np.sum(proportions * np.log2(proportions + 1e-10))
            
            return entropy_value
        return entropy_value


class ModePrimitive(AggregationPrimitive):
    """
    Returns the mode (most frequent value) of a categorical column.
    """
    name = "mode"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(logical_type=Categorical)
    stack_on_self = False
    
    def get_function(self):
        def mode_func(categories):
            if len(categories) == 0:
                return None
            
            unique, counts = np.unique(categories, return_counts=True)
            return unique[np.argmax(counts)]
        return mode_func


class SkewPrimitive(AggregationPrimitive):
    """
    Calculates skewness of numeric distribution.
    """
    name = "skew"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def skew_func(values):
            if len(values) < 3:
                return 0.0
            
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=0)
            
            if std == 0:
                return 0.0
            
            skew = np.mean(((values - mean) / std) ** 3)
            return skew
        return skew_func


class KurtosisPrimitive(AggregationPrimitive):
    """
    Calculates kurtosis of numeric distribution.
    """
    name = "kurtosis"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    stack_on_self = False
    
    def get_function(self):
        def kurtosis_func(values):
            if len(values) < 4:
                return 0.0
            
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=0)
            
            if std == 0:
                return 0.0
            
            kurtosis = np.mean(((values - mean) / std) ** 4) - 3
            return kurtosis
        return kurtosis_func


class PrimitiveRegistry:
    """
    Registry for managing custom and built-in primitives.
    """
    
    _instance = None
    _primitives = {}
    _domain_mappings = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._register_default_primitives()
            self._setup_domain_mappings()
    
    def _register_default_primitives(self):
        """Register all custom primitives."""
        custom_primitives = [
            RecencyScorePrimitive,
            FrequencyScorePrimitive,
            TrendVelocityPrimitive,
            EntityAffinityPrimitive,
            InteractionDensityPrimitive,
            PercentileRankPrimitive,
            RollingMeanPrimitive,
            EntropyPrimitive,
            ModePrimitive,
            SkewPrimitive,
            KurtosisPrimitive,
        ]
        
        for prim in custom_primitives:
            self._primitives[prim.name] = prim
            logger.debug(f"Registered primitive: {prim.name}")
    
    def _setup_domain_mappings(self):
        """Setup domain-specific primitive mappings."""
        self._domain_mappings = {
            'ecommerce': {
                'aggregation': ['sum', 'mean', 'count', 'std', 'max', 'min', 
                               'num_unique', 'mode', 'entropy'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['recency_score', 'frequency_score', 'entity_affinity']
            },
            'fintech': {
                'aggregation': ['sum', 'mean', 'std', 'skew', 'kurtosis', 
                               'max', 'min', 'count'],
                'transformation': ['percentile_rank'],
                'custom': ['recency_score', 'frequency_score', 'trend_velocity']
            },
            'healthcare': {
                'aggregation': ['mean', 'std', 'count', 'max', 'min', 
                               'mode', 'entropy'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['recency_score', 'interaction_density']
            },
            'edtech': {
                'aggregation': ['count', 'mean', 'sum', 'mode', 'entropy'],
                'transformation': ['percentile_rank'],
                'custom': ['frequency_score', 'interaction_density', 'entity_affinity']
            },
            'saas': {
                'aggregation': ['count', 'sum', 'mean', 'std', 'num_unique'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['recency_score', 'frequency_score', 'interaction_density']
            },
            'crm': {
                'aggregation': ['count', 'sum', 'mean', 'max', 'min', 'mode'],
                'transformation': ['percentile_rank'],
                'custom': ['recency_score', 'entity_affinity']
            },
            'erp': {
                'aggregation': ['sum', 'mean', 'count', 'std', 'max', 'min'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['trend_velocity', 'frequency_score']
            },
            'analytics': {
                'aggregation': ['count', 'sum', 'mean', 'std', 'entropy', 
                               'num_unique', 'mode'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['interaction_density', 'entity_affinity']
            },
            'recommendation': {
                'aggregation': ['count', 'mean', 'sum', 'entropy', 'mode'],
                'transformation': ['percentile_rank'],
                'custom': ['recency_score', 'frequency_score', 'entity_affinity']
            },
            'behavioral': {
                'aggregation': ['count', 'mean', 'std', 'entropy', 'num_unique'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['interaction_density', 'trend_velocity', 'recency_score']
            },
            'generic': {
                'aggregation': ['sum', 'mean', 'count', 'std', 'max', 'min', 
                               'num_unique', 'mode', 'entropy', 'skew', 'kurtosis'],
                'transformation': ['percentile_rank', 'rolling_mean'],
                'custom': ['recency_score', 'frequency_score', 'trend_velocity', 
                          'entity_affinity', 'interaction_density']
            }
        }
    
    def get_primitive(self, name: str):
        """Get a primitive by name."""
        return self._primitives.get(name)
    
    def get_primitives_for_domain(self, domain: str = 'generic') -> Dict[str, List]:
        """Get recommended primitives for a specific domain."""
        return self._domain_mappings.get(domain, self._domain_mappings['generic'])
    
    def get_all_custom_primitives(self) -> List:
        """Get all registered custom primitives."""
        return list(self._primitives.values())
    
    def register_primitive(self, primitive_class):
        """Register a new custom primitive."""
        self._primitives[primitive_class.name] = primitive_class
        logger.info(f"Registered new primitive: {primitive_class.name}")
    
    def detect_domain(self, schema_info: Dict[str, Any]) -> str:
        """Auto-detect domain from schema information."""
        column_names = set(schema_info.get('columns', []))
        
        # Domain detection heuristics
        domain_indicators = {
            'ecommerce': {'transaction', 'product', 'cart', 'checkout', 'revenue', 'sales'},
            'fintech': {'payment', 'transfer', 'account', 'balance', 'interest', 'loan'},
            'healthcare': {'patient', 'diagnosis', 'treatment', 'medication', 'symptom'},
            'edtech': {'student', 'course', 'lesson', 'quiz', 'grade', 'enrollment'},
            'saas': {'subscription', 'user_session', 'feature_usage', 'plan'},
            'crm': {'lead', 'opportunity', 'contact', 'deal', 'pipeline'},
            'erp': {'inventory', 'order', 'shipment', 'warehouse', 'procurement'},
            'analytics': {'event', 'pageview', 'click', 'session', 'conversion'},
            'behavioral': {'action', 'behavior', 'activity', 'engagement'},
        }
        
        scores = {}
        for domain, keywords in domain_indicators.items():
            score = len(keywords & column_names)
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return 'generic'
        
        return max(scores, key=scores.get)


# Initialize registry
primitive_registry = PrimitiveRegistry()
