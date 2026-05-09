"""
System-wide constants for the recommendation engine.
Centralized configuration for paths, defaults, and system parameters.
"""

import os
from pathlib import Path


class Constants:
    """
    Centralized constants for the recommendation system.
    
    Provides consistent access to:
    - Directory paths
    - File extensions
    - Default parameters
    - System limits
    - Entity types
    """
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # Output subdirectories
    STANDARDIZED_DIR = OUTPUT_DIR / 'standardized'
    VALIDATED_DIR = OUTPUT_DIR / 'validated'
    FEATURES_DIR = OUTPUT_DIR / 'features'
    MODELS_DIR = OUTPUT_DIR / 'models'
    RECOMMENDATIONS_DIR = OUTPUT_DIR / 'recommendations'
    REPORTS_DIR = OUTPUT_DIR / 'reports'
    LOGS_DIR = OUTPUT_DIR / 'logs'
    
    # Configuration
    CONFIG_FILE = OUTPUT_DIR / 'rec_config.json'
    
    # File extensions
    PARQUET_EXT = '.parquet'
    CSV_EXT = '.csv'
    JSON_EXT = '.json'
    YAML_EXT = '.yaml'
    JOBLIB_EXT = '.joblib'
    
    # Entity types
    ENTITY_TYPES = ['users', 'items', 'transactions', 'interactions', 'categories']
    
    # Canonical table names
    USERS_TABLE = 'users'
    ITEMS_TABLE = 'items'
    TRANSACTIONS_TABLE = 'transactions'
    INTERACTIONS_TABLE = 'interactions'
    CATEGORIES_TABLE = 'categories'
    
    # Feature types
    AGGREGATION_FEATURES = 'aggregation'
    TEMPORAL_FEATURES = 'temporal'
    BEHAVIORAL_FEATURES = 'behavioral'
    RECENCY_FEATURES = 'recency'
    FREQUENCY_FEATURES = 'frequency'
    MONETARY_FEATURES = 'monetary'
    CATEGORY_AFFINITY = 'category_affinity'
    CUSTOMER_VALUE = 'customer_value'
    
    # Model types
    THREE_TOWER_MODEL = 'three_tower'
    DLRM_MODEL = 'dlrm'
    XGBOOST_MODEL = 'xgboost'
    LIGHTGBM_MODEL = 'lightgbm'
    MATRIX_FACTORIZATION = 'matrix_factorization'
    
    # Recommendation stages
    STAGE_CANDIDATE_GENERATION = 'candidate_generation'
    STAGE_RANKING = 'ranking'
    STAGE_RERANKING = 'reranking'
    
    # Default parameters
    DEFAULT_TOP_K = 10
    DEFAULT_CANDIDATE_COUNT = 100
    DEFAULT_EMBEDDING_DIM = 64
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 10
    
    # Validation severity levels
    SEVERITY_CRITICAL = 'critical'
    SEVERITY_WARNING = 'warning'
    SEVERITY_INFO = 'info'
    
    # Interaction types
    INTERACTION_VIEW = 'view'
    INTERACTION_CLICK = 'click'
    INTERACTION_CART = 'add_to_cart'
    INTERACTION_PURCHASE = 'purchase'
    INTERACTION_WISHLIST = 'wishlist'
    
    # Device types
    DEVICE_DESKTOP = 'desktop'
    DEVICE_MOBILE = 'mobile'
    DEVICE_TABLET = 'tablet'
    
    # Date formats
    DATE_FORMAT = '%Y-%m-%d'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    ISO_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
    
    # Null handling
    NULL_FILL_NUMERIC = 0
    NULL_FILL_CATEGORICAL = 'UNKNOWN'
    
    # Cache settings
    CACHE_EXPIRY_SECONDS = 3600  # 1 hour
    CACHE_MAX_ENTRIES = 1000
    
    # Logging
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_LEVEL = 'INFO'
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create all output directories if they don't exist."""
        directories = [
            cls.STANDARDIZED_DIR,
            cls.VALIDATED_DIR,
            cls.FEATURES_DIR,
            cls.MODELS_DIR,
            cls.RECOMMENDATIONS_DIR,
            cls.REPORTS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_standardized_path(cls, entity_type: str) -> Path:
        """Get path for standardized entity parquet file."""
        return cls.STANDARDIZED_DIR / f"{entity_type}{cls.PARQUET_EXT}"
    
    @classmethod
    def get_validated_path(cls, entity_type: str) -> Path:
        """Get path for validated entity parquet file."""
        return cls.VALIDATED_DIR / f"{entity_type}{cls.PARQUET_EXT}"
    
    @classmethod
    def get_features_path(cls, feature_type: str) -> Path:
        """Get path for features parquet file."""
        return cls.FEATURES_DIR / f"{feature_type}_features{cls.PARQUET_EXT}"
    
    @classmethod
    def get_model_path(cls, model_name: str, version: str = 'v1') -> Path:
        """Get path for model file."""
        model_dir = cls.MODELS_DIR / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"model{cls.JOBLIB_EXT}"
    
    @classmethod
    def get_recommendations_path(cls, user_id: int = None, batch: bool = False) -> Path:
        """Get path for recommendations output."""
        if batch:
            return cls.RECOMMENDATIONS_DIR / f"batch_recommendations{cls.CSV_EXT}"
        return cls.RECOMMENDATIONS_DIR / f"user_{user_id}{cls.JSON_EXT}"
    
    @classmethod
    def get_report_path(cls, report_type: str) -> Path:
        """Get path for report file."""
        return cls.REPORTS_DIR / f"{report_type}_report{cls.JSON_EXT}"
    
    @classmethod
    def get_log_path(cls, module_name: str) -> Path:
        """Get path for log file."""
        return cls.LOGS_DIR / f"{module_name}.log"
