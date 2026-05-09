"""
Recommendation Serving - Module 12

CLI-based recommendation serving infrastructure.

This module provides:
- Single-user recommendations
- Batch recommendations
- Recommendation explanations
- Recommendation exports
- Caching layer
- Trace logging

Export formats: JSON, CSV, parquet
"""

from .request_parser import RequestParser
from .recommendation_service import RecommendationService
from .session_handler import SessionHandler
from .cache_handler import CacheHandler
from .recommendation_formatter import RecommendationFormatter
from .explanation_generator import ExplanationGenerator
from .trace_logger import TraceLogger
from .realtime_recommendation import RealtimeRecommendation
from .batch_recommendation import BatchRecommendation
from .export_manager import ExportManager
from .run import run_serving

__all__ = [
    'RequestParser',
    'RecommendationService',
    'SessionHandler',
    'CacheHandler',
    'RecommendationFormatter',
    'ExplanationGenerator',
    'TraceLogger',
    'RealtimeRecommendation',
    'BatchRecommendation',
    'ExportManager',
    'run_serving'
]
