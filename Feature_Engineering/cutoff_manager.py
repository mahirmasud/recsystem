"""
Cutoff Manager - Manages cutoff times for leakage-free feature generation.

Ensures temporal safety in feature engineering by:
- Building cutoff time dataframes
- Managing training/inference cutoffs
- Preventing data leakage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CutoffManager:
    """Manages cutoff times for temporally-safe feature generation."""
    
    def __init__(self):
        self.cutoff_history = []
        logger.info("CutoffManager initialized")
    
    def build_cutoff_dataframe(self,
                               instances: Optional[List] = None,
                               cutoff_times: Optional[Union[pd.DataFrame, List, pd.Series]] = None,
                               target_dataframe_name: str = "data",
                               instance_column: str = "id") -> pd.DataFrame:
        """
        Build a cutoff time dataframe for Featuretools DFS.
        
        Args:
            instances: List of instance IDs
            cutoff_times: Cutoff times (DataFrame, list, or Series)
            target_dataframe_name: Name of target dataframe
            instance_column: Name of instance ID column
            
        Returns:
            Cutoff time DataFrame with columns: [instance_column, 'time']
        """
        logger.info(f"Building cutoff dataframe for {target_dataframe_name}")
        
        if isinstance(cutoff_times, pd.DataFrame):
            # Already a proper cutoff dataframe
            return cutoff_times
        
        # Build from instances and times
        if instances is None:
            raise ValueError("Must provide instances when cutoff_times is not a DataFrame")
        
        if cutoff_times is None:
            # Use current time for all instances
            cutoff_df = pd.DataFrame({
                instance_column: instances,
                'time': [pd.Timestamp.now()] * len(instances)
            })
        elif isinstance(cutoff_times, (list, pd.Series)):
            # Pair instances with times
            if len(cutoff_times) != len(instances):
                raise ValueError(f"Length mismatch: {len(instances)} instances vs {len(cutoff_times)} times")
            
            cutoff_df = pd.DataFrame({
                instance_column: instances,
                'time': list(cutoff_times)
            })
        else:
            # Single cutoff time for all instances
            cutoff_df = pd.DataFrame({
                instance_column: instances,
                'time': [cutoff_times] * len(instances)
            })
        
        # Ensure time column is datetime
        cutoff_df['time'] = pd.to_datetime(cutoff_df['time'])
        
        # Sort by time
        cutoff_df = cutoff_df.sort_values('time')
        
        self.cutoff_history.append({
            'timestamp': datetime.now(),
            'target': target_dataframe_name,
            'instance_count': len(cutoff_df)
        })
        
        return cutoff_df
    
    def create_time_split_cutoffs(self,
                                  df: pd.DataFrame,
                                  instance_col: str,
                                  time_col: str,
                                  split_date: datetime,
                                  include_only_before_split: bool = True) -> pd.DataFrame:
        """
        Create cutoff times for train/test split.
        
        Args:
            df: Source dataframe
            instance_col: Instance ID column
            time_col: Timestamp column
            split_date: Date to split on
            include_only_before_split: If True, only include instances with data before split
            
        Returns:
            Cutoff dataframe for training period
        """
        logger.info(f"Creating time-split cutoffs at {split_date}")
        
        if include_only_before_split:
            # Get last event before split for each instance
            filtered = df[df[time_col] < split_date].copy()
        else:
            filtered = df.copy()
        
        # Get the maximum time per instance (their "current" state at cutoff)
        cutoffs = filtered.groupby(instance_col)[time_col].max().reset_index()
        cutoffs.columns = [instance_col, 'time']
        
        # For instances with no events before split, use split_date
        if include_only_before_split:
            all_instances = df[instance_col].unique()
            existing = set(cutoffs[instance_col])
            missing = set(all_instances) - existing
            
            if missing:
                logger.info(f"Adding {len(missing)} instances with split_date as cutoff")
                missing_df = pd.DataFrame({
                    instance_col: list(missing),
                    'time': [split_date] * len(missing)
                })
                cutoffs = pd.concat([cutoffs, missing_df], ignore_index=True)
        
        cutoffs = cutoffs.sort_values('time')
        
        return cutoffs
    
    def create_rolling_window_cutoffs(self,
                                      df: pd.DataFrame,
                                      instance_col: str,
                                      time_col: str,
                                      window_size: str = '30D',
                                      step_size: str = '7D') -> List[pd.DataFrame]:
        """
        Create multiple cutoff dataframes for rolling window validation.
        
        Args:
            df: Source dataframe
            instance_col: Instance ID column
            time_col: Timestamp column
            window_size: Size of each window (pandas offset string)
            step_size: Step between windows (pandas offset string)
            
        Returns:
            List of cutoff dataframes, one per window
        """
        logger.info(f"Creating rolling window cutoffs (window={window_size}, step={step_size})")
        
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        
        cutoffs_list = []
        current_start = min_time
        
        while current_start < max_time:
            window_end = current_start + pd.Timedelta(window_size)
            
            if window_end > max_time:
                break
            
            # Get instances active in this window
            window_data = df[(df[time_col] >= current_start) & (df[time_col] < window_end)]
            
            if len(window_data) > 0:
                # Cutoff is at end of window for each instance
                cutoffs = window_data.groupby(instance_col)[time_col].max().reset_index()
                cutoffs.columns = [instance_col, 'time']
                cutoffs_list.append(cutoffs)
            
            current_start += pd.Timedelta(step_size)
        
        logger.info(f"Created {len(cutoffs_list)} rolling windows")
        return cutoffs_list
    
    def validate_cutoff_safety(self,
                               entityset: ft.EntitySet,
                               cutoff_df: pd.DataFrame,
                               target_dataframe_name: str,
                               time_column: str) -> Dict[str, Any]:
        """
        Validate that cutoff times are safe (no future data leakage).
        
        Args:
            entityset: Featuretools EntitySet
            cutoff_df: Cutoff time dataframe
            target_dataframe_name: Target dataframe name
            time_column: Time index column in target
            
        Returns:
            Validation report dictionary
        """
        import featuretools as ft
        
        validation = {
            'is_safe': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for cutoff times before any data
        target_df = entityset[target_dataframe_name]
        if time_column in target_df.columns:
            min_time = target_df[time_column].min()
            early_cutoffs = cutoff_df[cutoff_df['time'] < min_time]
            
            if len(early_cutoffs) > 0:
                validation['warnings'].append(
                    f"{len(early_cutoffs)} cutoffs are before earliest data ({min_time})"
                )
        
        # Check for duplicate instance-time combinations
        dup_check = cutoff_df.groupby(cutoff_df.columns.tolist()).size()
        duplicates = dup_check[dup_check > 1]
        
        if len(duplicates) > 0:
            validation['issues'].append(
                f"Found {len(duplicates)} duplicate instance-time combinations"
            )
            validation['is_safe'] = False
        
        # Check for null cutoff times
        null_times = cutoff_df['time'].isnull().sum()
        if null_times > 0:
            validation['issues'].append(f"{null_times} null cutoff times found")
            validation['is_safe'] = False
        
        return validation
    
    def get_training_cutoff_mask(self,
                                 cutoff_df: pd.DataFrame,
                                 max_cutoff_time: datetime) -> pd.Series:
        """
        Get boolean mask for training period cutoffs.
        
        Args:
            cutoff_df: Cutoff time dataframe
            max_cutoff_time: Maximum cutoff time for training
            
        Returns:
            Boolean Series indicating training rows
        """
        return cutoff_df['time'] <= max_cutoff_time
    
    def export_cutoff_history(self, output_path: str) -> None:
        """Export cutoff history to JSON."""
        import json
        
        history_export = []
        for record in self.cutoff_history:
            export_record = record.copy()
            export_record['timestamp'] = record['timestamp'].isoformat()
            history_export.append(export_record)
        
        with open(output_path, 'w') as f:
            json.dump(history_export, f, indent=2)
        
        logger.info(f"Exported cutoff history to {output_path}")
