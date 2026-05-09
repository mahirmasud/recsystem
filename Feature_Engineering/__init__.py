"""
Feature Engineering Module - Module 9

Generates ML-ready features for recommendation systems including:
- Aggregation features
- Temporal features  
- Behavioral features
- Recency/Frequency/Monetary (RFM) features
- Category affinity
- Customer lifetime value
- Feature versioning and selection
"""

from .entity_builder import EntityBuilder
from .feature_generator import FeatureGenerator
from .aggregation_features import AggregationFeatureBuilder
from .temporal_features import TemporalFeatureBuilder
from .behavioral_features import BehavioralFeatureBuilder
from .recency_features import RecencyFeatureBuilder
from .frequency_features import FrequencyFeatureBuilder
from .monetary_features import MonetaryFeatureBuilder
from .category_affinity import CategoryAffinityBuilder
from .customer_value import CustomerValueBuilder
from .feature_store import FeatureStore
from .feature_selector import FeatureSelector
from .feature_versioning import FeatureVersioning

__all__ = [
    'EntityBuilder',
    'FeatureGenerator',
    'AggregationFeatureBuilder',
    'TemporalFeatureBuilder',
    'BehavioralFeatureBuilder',
    'RecencyFeatureBuilder',
    'FrequencyFeatureBuilder',
    'MonetaryFeatureBuilder',
    'CategoryAffinityBuilder',
    'CustomerValueBuilder',
    'FeatureStore',
    'FeatureSelector',
    'FeatureVersioning',
]
