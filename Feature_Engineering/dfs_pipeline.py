"""
DFS Pipeline - Deep Feature Synthesis pipeline using Featuretools.

Implements automated feature generation with:
- Dynamic primitive selection
- Multi-domain support
- Cutoff time awareness
- Scalable DFS execution
"""

import featuretools as ft
from featuretools.primitives import (
    Sum, Mean, Count, Std, Max, Min, Mode, NumUnique,
    PercentTrue, NMostCommon, TimeSincePrevious, TimeSince
)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime

from .primitive_registry import PrimitiveRegistry, primitive_registry
from .cutoff_manager import CutoffManager

logger = logging.getLogger(__name__)


class DFSPipeline:
    """Deep Feature Synthesis pipeline for automated feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.primitive_registry = PrimitiveRegistry()
        self.cutoff_manager = CutoffManager()
        self.feature_matrices = {}
        self.feature_definitions = {}
        
        logger.info("DFSPipeline initialized")
    
    def run_dfs(self,
                entityset: ft.EntitySet,
                target_dataframe_name: str,
                target_column: Optional[str] = None,
                max_depth: int = 2,
                agg_primitives: Optional[List[str]] = None,
                trans_primitives: Optional[List[str]] = None,
                cutoff_time: Optional[pd.DataFrame] = None,
                cutoff_time_in_index: bool = False,
                training_window: Optional[str] = None,
                n_jobs: int = 1,
                verbose: bool = False) -> Tuple[pd.DataFrame, List]:
        """
        Run Deep Feature Synthesis to generate features.
        
        Args:
            entityset: Featuretools EntitySet
            target_dataframe_name: Name of target dataframe
            target_column: Optional target column for supervised learning
            max_depth: Maximum depth of feature generation
            agg_primitives: List of aggregation primitive names
            trans_primitives: List of transformation primitive names
            cutoff_time: Optional cutoff time dataframe
            cutoff_time_in_index: Whether to include cutoff time in index
            training_window: Optional training window for temporal features
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            
        Returns:
            Tuple of (feature_matrix, feature_definitions)
        """
        logger.info(f"Running DFS on '{target_dataframe_name}' with max_depth={max_depth}")
        
        # Get domain-specific primitives if not specified
        if not agg_primitives or not trans_primitives:
            domain = self.config.get('domain', 'generic')
            primitives_config = self.primitive_registry.get_primitives_for_domain(domain)
            
            if not agg_primitives:
                agg_primitives = primitives_config.get('aggregation', [])
            if not trans_primitives:
                trans_primitives = primitives_config.get('transformation', [])
        
        # Add custom primitives
        custom_primitives = primitives_config.get('custom', [])
        all_agg_primitives = list(agg_primitives) + custom_primitives
        
        # Convert string names to actual primitive classes
        resolved_agg = self._resolve_primitives(all_agg_primitives, is_aggregation=True)
        resolved_trans = self._resolve_primitives(trans_primitives, is_aggregation=False)
        
        # Filter out None values (unresolved primitives)
        resolved_agg = [p for p in resolved_agg if p is not None]
        resolved_trans = [p for p in resolved_trans if p is not None]
        
        logger.info(f"Using {len(resolved_agg)} aggregation primitives and {len(resolved_trans)} transformation primitives")
        
        # Run DFS
        feature_matrix, feature_defs = ft.dfs(
            entityset=entityset,
            target_dataframe_name=target_dataframe_name,
            target_column=target_column,
            max_depth=max_depth,
            agg_primitives=resolved_agg,
            trans_primitives=resolved_trans,
            cutoff_time=cutoff_time,
            cutoff_time_in_index=cutoff_time_in_index,
            training_window=training_window,
            n_jobs=n_jobs,
            verbose=verbose,
            return_feature_defs=True
        )
        
        # Store results
        self.feature_matrices[target_dataframe_name] = feature_matrix
        self.feature_definitions[target_dataframe_name] = feature_defs
        
        logger.info(f"Generated {len(feature_matrix.columns)} features for {len(feature_matrix)} records")
        
        return feature_matrix, feature_defs
    
    def _resolve_primitives(self, primitive_names: List[str], 
                           is_aggregation: bool = True) -> List:
        """Resolve primitive names to actual primitive classes."""
        resolved = []
        
        # Built-in Featuretools primitives
        builtin_primitives = {
            'sum': Sum,
            'mean': Mean,
            'count': Count,
            'std': Std,
            'max': Max,
            'min': Min,
            'mode': Mode,
            'num_unique': NumUnique,
            'percent_true': PercentTrue,
            'n_most_common': NMostCommon,
            'time_since_previous': TimeSincePrevious,
            'time_since': TimeSince,
        }
        
        for name in primitive_names:
            name_lower = name.lower()
            
            # Check built-in primitives
            if name_lower in builtin_primitives:
                resolved.append(builtin_primitives[name_lower])
                continue
            
            # Check custom primitives
            custom_prim = self.primitive_registry.get_primitive(name_lower)
            if custom_prim:
                resolved.append(custom_prim)
                continue
            
            logger.warning(f"Unknown primitive: {name}")
            resolved.append(None)
        
        return resolved
    
    def run_dfs_with_cutoffs(self,
                            entityset: ft.EntitySet,
                            target_dataframe_name: str,
                            instances: Optional[List] = None,
                            cutoff_times: Optional[Union[pd.DataFrame, List]] = None,
                            max_depth: int = 2,
                            **dfs_kwargs) -> pd.DataFrame:
        """
        Run DFS with cutoff times to prevent data leakage.
        
        Args:
            entityset: Featuretools EntitySet
            target_dataframe_name: Target dataframe name
            instances: List of instance IDs to generate features for
            cutoff_times: Cutoff times (DataFrame or list)
            max_depth: Maximum DFS depth
            **dfs_kwargs: Additional arguments for ft.dfs
            
        Returns:
            Feature matrix
        """
        logger.info("Running DFS with cutoff times")
        
        # Build cutoff time dataframe if needed
        if cutoff_times is not None:
            cutoff_df = self.cutoff_manager.build_cutoff_dataframe(
                instances=instances,
                cutoff_times=cutoff_times,
                target_dataframe_name=target_dataframe_name
            )
        else:
            cutoff_df = None
        
        return self.run_dfs(
            entityset=entityset,
            target_dataframe_name=target_dataframe_name,
            cutoff_time=cutoff_df,
            max_depth=max_depth,
            **dfs_kwargs
        )[0]
    
    def calculate_feature_importance(self,
                                    feature_matrix: pd.DataFrame,
                                    target: pd.Series,
                                    problem_type: str = 'classification',
                                    n_estimators: int = 100) -> pd.DataFrame:
        """
        Calculate feature importance using tree-based models.
        
        Args:
            feature_matrix: Feature matrix from DFS
            target: Target variable
            problem_type: 'classification' or 'regression'
            n_estimators: Number of trees in Random Forest
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        logger.info(f"Calculating feature importance ({problem_type})")
        
        # Handle categorical features
        X = feature_matrix.select_dtypes(include=[np.number]).fillna(0)
        
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        
        model.fit(X, target)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_feature_descriptions(self, feature_defs: List) -> Dict[str, str]:
        """
        Get human-readable descriptions for features.
        
        Args:
            feature_defs: List of feature definitions from DFS
            
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {}
        
        for feat in feature_defs:
            name = feat.get_name()
            description = feat.get_description()
            descriptions[name] = description
        
        return descriptions
    
    def filter_features_by_type(self, 
                                feature_matrix: pd.DataFrame,
                                feature_defs: List,
                                include_types: Optional[List[str]] = None,
                                exclude_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter features based on their type/operation.
        
        Args:
            feature_matrix: Feature matrix from DFS
            feature_defs: Feature definitions
            include_types: Types to include (e.g., ['sum', 'mean'])
            exclude_types: Types to exclude
            
        Returns:
            Filtered feature matrix
        """
        if not include_types and not exclude_types:
            return feature_matrix
        
        selected_cols = []
        
        for feat in feature_defs:
            name = feat.get_name()
            feat_str = str(feat)
            
            should_include = True
            
            if exclude_types:
                for excl in exclude_types:
                    if excl.lower() in feat_str.lower():
                        should_include = False
                        break
            
            if include_types and should_include:
                should_include = any(inc.lower() in feat_str.lower() for inc in include_types)
            
            if should_include:
                selected_cols.append(name)
        
        # Always keep non-feature columns (like ID)
        non_feature_cols = [c for c in feature_matrix.columns if c not in [f.get_name() for f in feature_defs]]
        selected_cols = non_feature_cols + selected_cols
        
        return feature_matrix[selected_cols]
    
    def export_feature_definitions(self, 
                                   feature_defs: List,
                                   output_path: str) -> None:
        """Export feature definitions to JSON."""
        import json
        
        defs_export = []
        for feat in feature_defs:
            defs_export.append({
                'name': feat.get_name(),
                'description': feat.get_description(),
                'dtype': str(feat.column_schema.semantic_tags),
                'formula': str(feat)
            })
        
        with open(output_path, 'w') as f:
            json.dump(defs_export, f, indent=2)
        
        logger.info(f"Exported {len(defs_export)} feature definitions to {output_path}")
