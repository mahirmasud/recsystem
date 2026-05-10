"""
Semantic transformer for applying semantic-aware data transformations.
Normalizes column meanings and aligns business semantics across datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


@dataclass
class SemanticRule:
    """Definition of a semantic transformation rule."""
    source_pattern: str
    target_canonical: str
    transformation_type: str  # rename, normalize, map_values, compute
    transformation_func: Optional[Callable] = None
    value_mapping: Optional[Dict[str, str]] = None
    description: str = ""


class SemanticTransformer:
    """
    Applies semantic transformations to standardize data meanings.
    
    Features:
    - Column semantic normalization
    - Value mapping and standardization
    - Business semantic alignment
    - Interaction type normalization
    - Event structure standardization
    """
    
    # Common semantic patterns for column renaming
    COLUMN_SEMANTIC_PATTERNS = {
        'user_id': ['customer_id', 'user_identifier', 'client_id', 'account_id', 'member_id'],
        'item_id': ['product_id', 'sku', 'item_identifier', 'product_identifier', 'article_id'],
        'transaction_id': ['order_id', 'purchase_id', 'sale_id', 'payment_id'],
        'timestamp': ['event_time', 'created_at', 'occurred_at', 'visit_time', 'datetime'],
        'interaction_type': ['event_type', 'action_type', 'behavior_type', 'activity_type'],
        'session_id': ['session_token', 'visit_id', 'session_identifier'],
        'device_type': ['device_category', 'platform', 'device_class'],
        'quantity': ['qty', 'amount', 'count', 'num_items'],
        'price': ['unit_price', 'sale_price', 'cost_per_unit', 'list_price'],
        'total_amount': ['line_total', 'total_price', 'order_total', 'grand_total'],
        'discount': ['discount_amount', 'price_reduction', 'markdown'],
        'category_id': ['cat_id', 'category_identifier', 'department_id'],
        'category_name': ['category_label', 'category_title', 'department_name'],
    }
    
    # Common interaction type normalizations
    INTERACTION_TYPE_MAPPINGS = {
        'click': 'click',
        'view': 'view',
        'impression': 'view',
        'page_view': 'view',
        'product_view': 'view',
        'add_to_cart': 'add_to_cart',
        'cart_add': 'add_to_cart',
        'purchase': 'purchase',
        'buy': 'purchase',
        'order': 'purchase',
        'checkout': 'purchase',
        'search': 'search',
        'like': 'engagement',
        'favorite': 'engagement',
        'share': 'engagement',
        'review': 'engagement',
        'rating': 'engagement',
    }
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize semantic transformer.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
        self.semantic_rules: List[SemanticRule] = []
        self._build_default_rules()
    
    def _build_default_rules(self) -> None:
        """Build default semantic transformation rules."""
        # Build column renaming rules from patterns
        for canonical, patterns in self.COLUMN_SEMANTIC_PATTERNS.items():
            for pattern in patterns:
                rule = SemanticRule(
                    source_pattern=pattern,
                    target_canonical=canonical,
                    transformation_type='rename',
                    description=f"Rename {pattern} to {canonical}"
                )
                self.semantic_rules.append(rule)
    
    def apply_semantic_transformation(
        self,
        df: pd.DataFrame,
        entity_type: str,
        custom_rules: Optional[List[SemanticRule]] = None
    ) -> pd.DataFrame:
        """
        Apply semantic transformations to a DataFrame.
        
        Args:
            df: Input DataFrame
            entity_type: Entity type name
            custom_rules: Additional custom semantic rules
        
        Returns:
            Transformed DataFrame with normalized semantics
        """
        logger.info(f"Applying semantic transformations for {entity_type}")
        
        df_result = df.copy()
        all_rules = self.semantic_rules + (custom_rules or [])
        
        # Apply column renaming based on semantic patterns
        df_result = self._apply_column_renaming(df_result, all_rules)
        
        # Apply value mappings for known columns
        df_result = self._apply_value_mappings(df_result, entity_type)
        
        # Normalize interaction types if present
        if 'interaction_type' in df_result.columns:
            df_result = self._normalize_interaction_types(df_result)
        
        logger.info(f"Completed semantic transformation for {entity_type}")
        return df_result
    
    def _apply_column_renaming(
        self,
        df: pd.DataFrame,
        rules: List[SemanticRule]
    ) -> pd.DataFrame:
        """Apply column renaming based on semantic rules."""
        rename_map = {}
        
        for rule in rules:
            if rule.transformation_type != 'rename':
                continue
            
            if rule.source_pattern in df.columns and rule.target_canonical not in df.columns:
                rename_map[rule.source_pattern] = rule.target_canonical
                logger.debug(f"Semantic rename: {rule.source_pattern} -> {rule.target_canonical}")
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def _apply_value_mappings(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> pd.DataFrame:
        """Apply value mappings for standardized categorical values."""
        df_result = df.copy()
        
        # Apply interaction type normalization
        if 'interaction_type' in df_result.columns:
            df_result['interaction_type'] = df_result['interaction_type'].str.lower().str.strip()
        
        # Apply device type normalization
        if 'device_type' in df_result.columns:
            device_mapping = {
                'mobile': 'mobile',
                'smartphone': 'mobile',
                'phone': 'mobile',
                'desktop': 'desktop',
                'computer': 'desktop',
                'pc': 'desktop',
                'tablet': 'tablet',
                'ipad': 'tablet',
                'unknown': 'unknown',
                '': 'unknown'
            }
            df_result['device_type'] = df_result['device_type'].str.lower().str.strip()
            df_result['device_type'] = df_result['device_type'].map(
                lambda x: device_mapping.get(x, x)
            )
        
        return df_result
    
    def _normalize_interaction_types(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Normalize interaction type values to canonical forms."""
        df_result = df.copy()
        
        original_col = df_result['interaction_type'].copy()
        
        # Apply mapping
        df_result['interaction_type'] = df_result['interaction_type'].str.lower().str.strip()
        df_result['interaction_type'] = df_result['interaction_type'].map(
            lambda x: self.INTERACTION_TYPE_MAPPINGS.get(x, x)
        )
        
        changed = (original_col.str.lower() != df_result['interaction_type']).sum()
        if changed > 0:
            logger.debug(f"Normalized {changed} interaction type values")
        
        return df_result
    
    def infer_entity_role(
        self,
        df: pd.DataFrame,
        column_name: str
    ) -> Optional[str]:
        """
        Infer the semantic role of a column.
        
        Args:
            df: DataFrame containing the column
            column_name: Name of column to analyze
        
        Returns:
            Inferred role or None
        """
        col_lower = column_name.lower()
        
        # Check for ID patterns
        if col_lower.endswith('_id') or 'identifier' in col_lower:
            if 'user' in col_lower or 'customer' in col_lower or 'client' in col_lower:
                return 'user_id'
            elif 'item' in col_lower or 'product' in col_lower or 'sku' in col_lower:
                return 'item_id'
            elif 'transaction' in col_lower or 'order' in col_lower:
                return 'transaction_id'
            elif 'interaction' in col_lower or 'event' in col_lower:
                return 'interaction_id'
            elif 'category' in col_lower:
                return 'category_id'
            elif 'session' in col_lower:
                return 'session_id'
        
        # Check for timestamp patterns
        if any(pattern in col_lower for pattern in ['time', 'date', 'timestamp', 'at']):
            return 'timestamp'
        
        # Check for numeric measures
        if any(pattern in col_lower for pattern in ['amount', 'price', 'total', 'sum']):
            return 'monetary_value'
        elif any(pattern in col_lower for pattern in ['qty', 'quantity', 'count']):
            return 'quantity'
        
        return None
    
    def detect_semantic_type(
        self,
        series: pd.Series,
        column_name: str
    ) -> str:
        """
        Detect the semantic type of a column based on values.
        
        Args:
            series: Column data
            column_name: Column name
        
        Returns:
            Detected semantic type
        """
        # Check for email pattern
        if series.dtype == 'object' or series.dtype == 'string':
            sample = series.dropna().head(10)
            if any('@' in str(val) for val in sample):
                return 'email'
            
            # Check for URL pattern
            if any(str(val).startswith('http') for val in sample):
                return 'url'
        
        # Check for boolean
        if series.dtype == 'bool' or set(series.dropna().unique()) <= {True, False, 0, 1}:
            return 'boolean'
        
        # Check for categorical
        if series.dtype == 'object' or series.dtype == 'string':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 and series.nunique() < 50:
                return 'categorical'
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 20:
                return 'discrete_numeric'
            return 'continuous_numeric'
        
        return 'unknown'
    
    def build_semantic_profile(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Build a semantic profile of a DataFrame.
        
        Args:
            df: Input DataFrame
            entity_type: Entity type name
        
        Returns:
            Semantic profile dictionary
        """
        profile = {
            'entity_type': entity_type,
            'columns': {},
            'inferred_roles': {},
            'semantic_types': {}
        }
        
        for col in df.columns:
            # Infer role from column name
            inferred_role = self.infer_entity_role(df, col)
            if inferred_role:
                profile['inferred_roles'][col] = inferred_role
            
            # Detect semantic type
            semantic_type = self.detect_semantic_type(df[col], col)
            profile['semantic_types'][col] = semantic_type
            
            # Column statistics
            profile['columns'][col] = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'semantic_type': semantic_type,
                'inferred_role': inferred_role
            }
        
        return profile
    
    def add_custom_rule(self, rule: SemanticRule) -> None:
        """Add a custom semantic transformation rule."""
        self.semantic_rules.append(rule)
        logger.debug(f"Added custom semantic rule: {rule.source_pattern} -> {rule.target_canonical}")
    
    def clear_rules(self) -> None:
        """Clear all semantic rules."""
        self.semantic_rules = []
        self._build_default_rules()
