"""
Dataset writer for standardized data.
Writes standardized DataFrames to parquet files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from shared.constants import Constants
from shared.file_loader import FileLoader


logger = get_logger(__name__)


class DatasetWriter:
    """
    Writes standardized datasets to output files.
    
    Features:
    - Parquet output format
    - Automatic directory creation
    - Metadata tracking
    - Batch writing support
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize dataset writer.
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Constants.STANDARDIZED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_history: List[Dict[str, Any]] = []
    
    def write(
        self,
        df: pd.DataFrame,
        entity_type: str,
        filename: Optional[str] = None,
        compression: str = 'snappy',
        include_index: bool = False
    ) -> Path:
        """
        Write DataFrame to parquet file.
        
        Args:
            df: DataFrame to write
            entity_type: Entity type name
            filename: Custom filename (default: {entity_type}.parquet)
            compression: Parquet compression codec
            include_index: Whether to include index
        
        Returns:
            Path to written file
        """
        if filename is None:
            filename = f"{entity_type}.parquet"
        
        filepath = self.output_dir / filename
        
        logger.info(f"Writing {entity_type} to {filepath} ({len(df)} rows)")
        
        try:
            FileLoader.save_parquet(df, filepath, index=include_index, compression=compression)
            
            # Record write operation
            self.write_history.append({
                'entity_type': entity_type,
                'filepath': str(filepath),
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'compression': compression
            })
            
            logger.info(f"Successfully wrote {entity_type} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to write {entity_type}: {e}")
            raise
    
    def write_multiple(
        self,
        dataframes: Dict[str, pd.DataFrame],
        compression: str = 'snappy'
    ) -> Dict[str, Path]:
        """
        Write multiple DataFrames.
        
        Args:
            dataframes: Dictionary mapping entity type to DataFrame
            compression: Parquet compression codec
        
        Returns:
            Dictionary mapping entity type to output path
        """
        results = {}
        
        for entity_type, df in dataframes.items():
            try:
                filepath = self.write(df, entity_type, compression=compression)
                results[entity_type] = filepath
            except Exception as e:
                logger.error(f"Failed to write {entity_type}: {e}")
                raise
        
        logger.info(f"Successfully wrote {len(results)} datasets")
        return results
    
    def write_validated(
        self,
        df: pd.DataFrame,
        entity_type: str,
        validated_dir: bool = True
    ) -> Path:
        """
        Write validated dataset to appropriate directory.
        
        Args:
            df: Validated DataFrame
            entity_type: Entity type name
            validated_dir: If True, write to validated/ directory
        
        Returns:
            Path to written file
        """
        if validated_dir:
            output_dir = Constants.VALIDATED_DIR
        else:
            output_dir = self.output_dir
        
        writer = DatasetWriter(output_dir)
        return writer.write(df, entity_type)
    
    def get_write_summary(self) -> Dict[str, Any]:
        """
        Get summary of all write operations.
        
        Returns:
            Summary dictionary
        """
        total_rows = sum(record['row_count'] for record in self.write_history)
        
        return {
            'total_datasets': len(self.write_history),
            'total_rows': total_rows,
            'datasets': [
                {
                    'entity_type': record['entity_type'],
                    'filepath': record['filepath'],
                    'row_count': record['row_count'],
                    'column_count': record['column_count']
                }
                for record in self.write_history
            ]
        }
    
    def load_dataset(self, entity_type: str) -> pd.DataFrame:
        """
        Load a previously written dataset.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Loaded DataFrame
        """
        filepath = self.output_dir / f"{entity_type}.parquet"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        return FileLoader.load_parquet(filepath)
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets in output directory.
        
        Returns:
            List of entity type names
        """
        datasets = []
        for filepath in self.output_dir.glob("*.parquet"):
            datasets.append(filepath.stem)
        return datasets
    
    def clear_output(self, entity_type: Optional[str] = None) -> None:
        """
        Clear output files.
        
        Args:
            entity_type: Specific entity to clear (None for all)
        """
        if entity_type:
            filepath = self.output_dir / f"{entity_type}.parquet"
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Cleared {entity_type} dataset")
        else:
            for filepath in self.output_dir.glob("*.parquet"):
                filepath.unlink()
            logger.info("Cleared all datasets")
        
        self.write_history = []
