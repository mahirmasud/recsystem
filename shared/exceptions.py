"""
Custom exceptions for the recommendation system.
Provides typed exceptions for different error scenarios.
"""


class RecommendationSystemError(Exception):
    """Base exception for all recommendation system errors."""
    pass


class ConfigurationError(RecommendationSystemError):
    """
    Raised when there is a configuration problem.
    
    Examples:
    - Missing configuration file
    - Invalid configuration values
    - Missing required settings
    """
    pass


class DataValidationError(RecommendationSystemError):
    """
    Raised when data validation fails.
    
    Examples:
    - Schema mismatch
    - Invalid data types
    - Missing required columns
    - Constraint violations
    """
    pass


class FeatureEngineeringError(RecommendationSystemError):
    """
    Raised when feature engineering fails.
    
    Examples:
    - Feature computation errors
    - Missing source data
    - Invalid feature configurations
    """
    pass


class ModelTrainingError(RecommendationSystemError):
    """
    Raised when model training fails.
    
    Examples:
    - Training data issues
    - Hyperparameter errors
    - Convergence failures
    """
    pass


class ModelInferenceError(RecommendationSystemError):
    """
    Raised when model inference fails.
    
    Examples:
    - Missing model files
    - Input shape mismatches
    - Prediction errors
    """
    pass


class RecommendationError(RecommendationSystemError):
    """
    Raised when recommendation generation fails.
    
    Examples:
    - Candidate retrieval failures
    - Ranking errors
    - Re-ranking failures
    """
    pass


class RuleExecutionError(RecommendationSystemError):
    """
    Raised when rule execution fails.
    
    Examples:
    - Invalid rule syntax
    - Rule evaluation errors
    - Missing rule dependencies
    """
    pass


class DataLoadError(RecommendationSystemError):
    """
    Raised when data loading fails.
    
    Examples:
    - File not found
    - Parse errors
    - Connection failures
    """
    pass


class DataSaveError(RecommendationSystemError):
    """
    Raised when saving data fails.
    
    Examples:
    - Permission errors
    - Disk full
    - Serialization errors
    """
    pass


class CacheError(RecommendationSystemError):
    """
    Raised when cache operations fail.
    
    Examples:
    - Cache miss (when critical)
    - Cache corruption
    - Cache write failures
    """
    pass


class EmbeddingError(RecommendationSystemError):
    """
    Raised when embedding operations fail.
    
    Examples:
    - Embedding generation failures
    - Dimension mismatches
    - Index building errors
    """
    pass


class EvaluationError(RecommendationSystemError):
    """
    Raised when evaluation metrics calculation fails.
    
    Examples:
    - Missing ground truth
    - Metric computation errors
    - Invalid input formats
    """
    pass
