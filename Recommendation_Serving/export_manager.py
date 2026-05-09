"""
Export Manager - Manages export of recommendations to various formats.

Supports JSON, CSV, and Parquet export formats with
configurable output paths and naming conventions.
"""

from typing import Dict, Any, Optional, List
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import logging
from shared.logger import get_logger
from shared.constants import Constants

logger = get_logger(__name__)


class ExportManager:
    """
    Manages export of recommendation results to files.
    
    Supported formats:
    - JSON
    - CSV
    - Parquet
    
    Features:
    - Automatic directory creation
    - Timestamped filenames
    - Batch export support
    - User-specific export
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the export manager.
        
        Args:
            output_dir: Base directory for exports
        """
        self.output_dir = Path(output_dir) if output_dir else Constants.RECOMMENDATIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExportManager initialized with output_dir={self.output_dir}")
    
    def export_user_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        format: str = 'json',
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export recommendations for a single user.
        
        Args:
            user_id: User ID
            recommendations: List of recommendations
            format: Output format (json, csv, parquet)
            metadata: Optional metadata
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            return self._export_user_json(user_id, recommendations, metadata, timestamp)
        elif format == 'csv':
            return self._export_user_csv(user_id, recommendations, timestamp)
        elif format == 'parquet':
            return self._export_user_parquet(user_id, recommendations, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_user_json(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """Export user recommendations as JSON."""
        filename = f"user_{user_id}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        data = {
            'user_id': user_id,
            'generated_at': datetime.now().isoformat(),
            'recommendations': recommendations,
            'count': len(recommendations)
        }
        
        if metadata:
            data['metadata'] = metadata
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported JSON for user {user_id} to {filepath}")
        return str(filepath)
    
    def _export_user_csv(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """Export user recommendations as CSV."""
        filename = f"user_{user_id}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        if not recommendations:
            # Create empty file
            filepath.touch()
            return str(filepath)
        
        # Determine columns from first recommendation
        base_columns = ['rank', 'item_id', 'score', 'reason']
        optional_columns = ['item_name', 'price', 'category', 'rerank_score']
        
        all_columns = base_columns + [c for c in optional_columns if c in recommendations[0]]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
            writer.writeheader()
            
            for rec in recommendations:
                row = {col: rec.get(col, '') for col in all_columns}
                writer.writerow(row)
        
        logger.info(f"Exported CSV for user {user_id} to {filepath}")
        return str(filepath)
    
    def _export_user_parquet(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """Export user recommendations as Parquet."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available, falling back to JSON")
            return self._export_user_json(user_id, recommendations, None, timestamp)
        
        filename = f"user_{user_id}_{timestamp}.parquet"
        filepath = self.output_dir / filename
        
        if not recommendations:
            # Create empty parquet
            df = pd.DataFrame()
            df.to_parquet(filepath)
            return str(filepath)
        
        df = pd.DataFrame(recommendations)
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Exported Parquet for user {user_id} to {filepath}")
        return str(filepath)
    
    def export_batch(
        self,
        batch_result: Dict[str, Any],
        format: str = 'json',
        filename: Optional[str] = None
    ) -> str:
        """
        Export batch recommendation results.
        
        Args:
            batch_result: Batch result dictionary
            format: Output format
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if filename is None:
            filename = f"batch_recommendations_{timestamp}"
        
        if format == 'json':
            return self._export_batch_json(batch_result, filename)
        elif format == 'csv':
            return self._export_batch_csv(batch_result, filename)
        elif format == 'parquet':
            return self._export_batch_parquet(batch_result, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_batch_json(
        self,
        batch_result: Dict[str, Any],
        filename: str
    ) -> str:
        """Export batch results as JSON."""
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(batch_result, f, indent=2, default=str)
        
        logger.info(f"Exported batch JSON to {filepath}")
        return str(filepath)
    
    def _export_batch_csv(
        self,
        batch_result: Dict[str, Any],
        filename: str
    ) -> str:
        """Export batch results as CSV (flattened)."""
        filepath = self.output_dir / f"{filename}.csv"
        
        results = batch_result.get('results', [])
        
        if not results:
            filepath.touch()
            return str(filepath)
        
        # Flatten structure
        rows = []
        for user_result in results:
            user_id = user_result.get('user_id')
            recommendations = user_result.get('recommendations', [])
            
            for rec in recommendations:
                row = {'user_id': user_id, **rec}
                rows.append(row)
        
        if not rows:
            filepath.touch()
            return str(filepath)
        
        # Determine columns
        base_columns = ['user_id', 'rank', 'item_id', 'score', 'reason']
        optional_columns = ['item_name', 'price', 'category', 'rerank_score']
        
        all_columns = base_columns + [c for c in optional_columns if c in rows[0]]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported batch CSV to {filepath}")
        return str(filepath)
    
    def _export_batch_parquet(
        self,
        batch_result: Dict[str, Any],
        filename: str
    ) -> str:
        """Export batch results as Parquet."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available, falling back to JSON")
            return self._export_batch_json(batch_result, filename)
        
        filepath = self.output_dir / f"{filename}.parquet"
        
        results = batch_result.get('results', [])
        
        # Flatten structure
        rows = []
        for user_result in results:
            user_id = user_result.get('user_id')
            recommendations = user_result.get('recommendations', [])
            
            for rec in recommendations:
                row = {'user_id': user_id, **rec}
                rows.append(row)
        
        if not rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows)
        
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Exported batch Parquet to {filepath}")
        return str(filepath)
    
    def cleanup_old_exports(
        self, 
        max_age_days: int = 7,
        pattern: Optional[str] = None
    ) -> int:
        """
        Clean up old export files.
        
        Args:
            max_age_days: Maximum age of files to keep
            pattern: Optional filename pattern to match
            
        Returns:
            Number of files deleted
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0
        
        for filepath in self.output_dir.iterdir():
            if not filepath.is_file():
                continue
            
            if pattern and pattern not in filepath.name:
                continue
            
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")
        
        logger.info(f"Cleaned up {deleted} old export files")
        return deleted
    
    def list_exports(
        self, 
        pattern: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List export files.
        
        Args:
            pattern: Optional filename pattern
            limit: Maximum number of files to return
            
        Returns:
            List of file info dictionaries
        """
        files = []
        
        for filepath in sorted(
            self.output_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]:
            if pattern and pattern not in filepath.name:
                continue
            
            files.append({
                'filename': filepath.name,
                'path': str(filepath),
                'size_bytes': filepath.stat().st_size,
                'modified': datetime.fromtimestamp(
                    filepath.stat().st_mtime
                ).isoformat()
            })
        
        return files
