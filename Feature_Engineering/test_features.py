"""
Feature Engineering Module Tests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest


def test_aggregation_features():
    """Test aggregation feature building."""
    from .aggregation_features import AggregationFeatureBuilder
    
    config = {}
    builder = AggregationFeatureBuilder(config)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'user_id': np.random.choice([1, 2, 3], 100),
        'item_id': np.random.choice([101, 102, 103, 104], 100),
        'category_id': np.random.choice([10, 11, 12], 100),
        'transaction_id': range(100),
        'quantity': np.random.randint(1, 5, 100),
        'net_sales': np.random.uniform(10, 500, 100),
        'unit_price': np.random.uniform(10, 100, 100),
        'discount_amount': np.random.uniform(0, 20, 100),
        'transaction_date': np.random.choice(dates, 100)
    })
    
    user_agg = builder.build_user_aggregations(df)
    item_agg = builder.build_item_aggregations(df)
    
    assert 'user_id' in user_agg.columns
    assert 'total_transactions' in user_agg.columns
    assert 'item_id' in item_agg.columns
    assert 'total_units_sold' in item_agg.columns
    
    print("✓ Aggregation features test passed")


def test_temporal_features():
    """Test temporal feature building."""
    from .temporal_features import TemporalFeatureBuilder
    
    config = {}
    builder = TemporalFeatureBuilder(config)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'user_id': np.random.choice([1, 2, 3], 100),
        'item_id': np.random.choice([101, 102], 100),
        'transaction_id': range(100),
        'net_sales': np.random.uniform(10, 500, 100),
        'transaction_date': dates
    })
    
    reference_date = datetime(2024, 1, 10)
    user_temporal = builder.build_user_temporal_features(df, reference_date)
    
    assert 'recency_days' in user_temporal.columns or 'days_since_last_purchase' in user_temporal.columns
    assert 'customer_tenure_days' in user_temporal.columns
    
    print("✓ Temporal features test passed")


def test_rfm_features():
    """Test RFM feature building."""
    from .recency_features import RecencyFeatureBuilder
    from .frequency_features import FrequencyFeatureBuilder
    from .monetary_features import MonetaryFeatureBuilder
    
    config = {}
    ref_date = datetime(2024, 1, 10)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'user_id': np.random.choice([1, 2, 3], 100),
        'item_id': np.random.choice([101, 102], 100),
        'category_id': np.random.choice([10, 11], 100),
        'transaction_id': range(100),
        'quantity': np.random.randint(1, 5, 100),
        'net_sales': np.random.uniform(10, 500, 100),
        'transaction_date': np.random.choice(dates, 100)
    })
    
    recency_builder = RecencyFeatureBuilder(config)
    frequency_builder = FrequencyFeatureBuilder(config)
    monetary_builder = MonetaryFeatureBuilder(config)
    
    recency = recency_builder.build_recency_features(df, ref_date)
    frequency = frequency_builder.build_frequency_features(df)
    monetary = monetary_builder.build_monetary_features(df)
    
    assert 'recency_days' in recency.columns
    assert 'total_transactions' in frequency.columns
    assert 'total_revenue' in monetary.columns
    
    print("✓ RFM features test passed")


def test_feature_store():
    """Test feature store functionality."""
    from .feature_store import FeatureStore
    
    config = {}
    store = FeatureStore(config, store_path="output/features/test")
    
    df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'feature_1': [0.1, 0.2, 0.3],
        'feature_2': [10, 20, 30]
    })
    
    store.store_features('users', df)
    
    retrieved = store.get_features('users')
    assert retrieved is not None
    assert len(retrieved) == 3
    
    metadata = store.get_feature_metadata()
    assert 'users' in metadata
    
    print("✓ Feature store test passed")


def test_feature_versioning():
    """Test feature versioning."""
    from .feature_versioning import FeatureVersioning
    
    versioning = FeatureVersioning(output_dir="output/features/test_versions")
    
    df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'feature_1': [0.1, 0.2, 0.3]
    })
    
    version_id = versioning.create_version({'users': df}, description="Test version")
    
    assert version_id is not None
    
    versions = versioning.list_versions()
    assert len(versions) >= 1
    
    loaded = versioning.get_version(version_id)
    assert loaded is not None
    assert 'users' in loaded
    
    print("✓ Feature versioning test passed")


if __name__ == "__main__":
    test_aggregation_features()
    test_temporal_features()
    test_rfm_features()
    test_feature_store()
    test_feature_versioning()
    print("\n✅ All Feature Engineering tests passed!")
