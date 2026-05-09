"""
Metric builder for creating derived business metrics.
Implements formulas from rec_config.json business_meanings section.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


class MetricBuilder:
    """
    Builds derived metrics from standardized data.
    
    Supports:
    - net_sales calculation
    - profit_margin calculation
    - customer_lifetime_value
    - discount_rate
    - Custom formula evaluation
    """
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize metric builder.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
    
    def build_all_metrics(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> pd.DataFrame:
        """
        Build all applicable metrics for an entity type.
        
        Args:
            df: Standardized DataFrame
            entity_type: Entity type name
        
        Returns:
            DataFrame with added metric columns
        """
        logger.info(f"Building metrics for {entity_type}")
        
        df_result = df.copy()
        business_meanings = self.config_reader.get_business_meanings()
        
        for metric_name, definition in business_meanings.items():
            try:
                df_result = self._build_metric(df_result, metric_name, definition)
            except Exception as e:
                logger.warning(f"Failed to build metric {metric_name}: {e}")
        
        return df_result
    
    def _build_metric(
        self,
        df: pd.DataFrame,
        metric_name: str,
        definition: Dict[str, str]
    ) -> pd.DataFrame:
        """Build a single metric column."""
        formula = definition.get('formula', '')
        
        if not formula:
            return df
        
        df_result = df.copy()
        
        try:
            # Evaluate formula using pandas eval
            df_result[metric_name] = df_result.eval(formula)
            logger.debug(f"Built metric: {metric_name}")
        except Exception as e:
            logger.warning(f"Failed to evaluate formula for {metric_name}: {e}")
        
        return df_result
    
    def build_net_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate net sales (total_amount - discount).
        
        Args:
            df: DataFrame with total_amount and discount columns
        
        Returns:
            DataFrame with net_sales column
        """
        df_result = df.copy()
        
        required_cols = ['total_amount', 'discount']
        if all(col in df_result.columns for col in required_cols):
            df_result['net_sales'] = df_result['total_amount'] - df_result['discount']
            logger.debug("Built net_sales metric")
        
        return df_result
    
    def build_profit_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate profit margin ((price - cost) / price).
        
        Args:
            df: DataFrame with price and cost columns
        
        Returns:
            DataFrame with profit_margin column
        """
        df_result = df.copy()
        
        if 'price' in df_result.columns and 'cost' in df_result.columns:
            df_result['profit_margin'] = (df_result['price'] - df_result['cost']) / df_result['price'].replace(0, np.nan)
            df_result['profit_margin'] = df_result['profit_margin'].fillna(0)
            logger.debug("Built profit_margin metric")
        
        return df_result
    
    def build_discount_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate discount rate (discount / total_amount).
        
        Args:
            df: DataFrame with discount and total_amount columns
        
        Returns:
            DataFrame with discount_rate column
        """
        df_result = df.copy()
        
        if 'discount' in df_result.columns and 'total_amount' in df_result.columns:
            df_result['discount_rate'] = df_result['discount'] / df_result['total_amount'].replace(0, np.nan)
            df_result['discount_rate'] = df_result['discount_rate'].fillna(0)
            logger.debug("Built discount_rate metric")
        
        return df_result
    
    def build_customer_lifetime_value(
        self,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate customer lifetime value (sum of all transactions per user).
        
        Args:
            transactions_df: Transactions DataFrame
        
        Returns:
            DataFrame with CLV per user_id
        """
        if 'user_id' not in transactions_df.columns or 'total_amount' not in transactions_df.columns:
            return transactions_df
        
        clv = transactions_df.groupby('user_id')['total_amount'].sum().reset_index()
        clv.rename(columns={'total_amount': 'customer_lifetime_value'}, inplace=True)
        
        logger.debug("Built customer_lifetime_value metric")
        return clv
    
    def build_aggregate_metrics(
        self,
        df: pd.DataFrame,
        group_by: str,
        metrics: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Build aggregate metrics grouped by a column.
        
        Args:
            df: Input DataFrame
            group_by: Column to group by
            metrics: List of metric definitions with aggregation
        
        Returns:
            DataFrame with aggregated metrics
        """
        agg_dict = {}
        for metric_def in metrics:
            col = metric_def.get('column')
            agg = metric_def.get('aggregation', 'sum')
            if col and col in df.columns:
                agg_dict[col] = agg
        
        if not agg_dict:
            return df
        
        result = df.groupby(group_by).agg(agg_dict).reset_index()
        logger.debug(f"Built aggregate metrics grouped by {group_by}")
        return result
