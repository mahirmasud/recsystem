"""
File loading utilities for the recommendation system.
Supports loading data from CSV, Parquet, and JSON formats.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
from shared.logger import get_logger


logger = get_logger(__name__)


class FileLoader:
    """
    Universal file loader for various data formats.
    
    Supports:
    - CSV files
    - Parquet files
    - JSON files
    - Automatic format detection
    """
    
    @staticmethod
    def load_csv(
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pd.read_csv
        
        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)
        logger.info(f"Loading CSV from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    
    @staticmethod
    def load_parquet(
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            **kwargs: Additional arguments passed to pd.read_parquet
        
        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)
        logger.info(f"Loading Parquet from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {filepath}")
        
        df = pd.read_parquet(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    
    @staticmethod
    def load_json(
        filepath: Union[str, Path],
        **kwargs
    ) -> Union[Dict[str, Any], list]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            **kwargs: Additional arguments passed to json.load
        
        Returns:
            Parsed JSON data (dict or list)
        """
        filepath = Path(filepath)
        logger.info(f"Loading JSON from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f, **kwargs)
        
        logger.info(f"Loaded JSON from {filepath}")
        return data
    
    @staticmethod
    def load_auto(
        filepath: Union[str, Path],
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, Any], list]:
        """
        Automatically detect file format and load accordingly.
        
        Args:
            filepath: Path to file
            **kwargs: Additional arguments passed to load function
        
        Returns:
            Loaded data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            return FileLoader.load_csv(filepath, **kwargs)
        elif suffix == '.parquet':
            return FileLoader.load_parquet(filepath, **kwargs)
        elif suffix == '.json':
            return FileLoader.load_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def save_csv(
        df: pd.DataFrame,
        filepath: Union[str, Path],
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Output path
            index: Whether to include index
            **kwargs: Additional arguments passed to df.to_csv
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving CSV to {filepath} ({len(df)} rows)")
        df.to_csv(filepath, index=index, **kwargs)
        logger.info(f"Saved CSV to {filepath}")
    
    @staticmethod
    def save_parquet(
        df: pd.DataFrame,
        filepath: Union[str, Path],
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Save DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            filepath: Output path
            index: Whether to include index
            **kwargs: Additional arguments passed to df.to_parquet
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving Parquet to {filepath} ({len(df)} rows)")
        df.to_parquet(filepath, index=index, **kwargs)
        logger.info(f"Saved Parquet to {filepath}")
    
    @staticmethod
    def save_json(
        data: Union[Dict[str, Any], list],
        filepath: Union[str, Path],
        indent: int = 2,
        **kwargs
    ) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save (dict or list)
            filepath: Output path
            indent: JSON indentation level
            **kwargs: Additional arguments passed to json.dump
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving JSON to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, **kwargs)
        logger.info(f"Saved JSON to {filepath}")
