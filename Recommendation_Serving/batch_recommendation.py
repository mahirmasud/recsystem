"""
Batch Recommendation - Batch recommendation generation.

Handles bulk recommendation requests for multiple users,
with support for file-based input/output.
"""

from typing import Dict, Any, Optional, List
import time
import os
import json
import csv
import logging
from pathlib import Path
from shared.logger import get_logger
from shared.constants import Constants
from .recommendation_service import RecommendationService
from .recommendation_formatter import RecommendationFormatter
from .export_manager import ExportManager

logger = get_logger(__name__)


class BatchRecommendation:
    """
    Handles batch recommendation requests.
    
    Features:
    - Process multiple users efficiently
    - File-based input (CSV, JSON)
    - Multiple output formats
    - Progress tracking
    - Error handling per user
    """
    
    def __init__(
        self,
        service: RecommendationService,
        output_dir: Optional[str] = None
    ):
        """
        Initialize batch recommendation handler.
        
        Args:
            service: RecommendationService instance
            output_dir: Directory for output files
        """
        self.service = service
        self.formatter = RecommendationFormatter()
        self.export_manager = ExportManager(output_dir or Constants.RECOMMENDATIONS_DIR)
        
        logger.info("BatchRecommendation initialized")
    
    def generate_for_users(
        self,
        user_ids: List[int],
        n: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            context: Optional context dictionary
            
        Returns:
            Batch result with all recommendations
        """
        start_time = time.time()
        results = []
        errors = []
        
        logger.info(f"Starting batch generation for {len(user_ids)} users")
        
        for i, user_id in enumerate(user_ids):
            try:
                result = self.service.get_recommendations(
                    user_id=user_id,
                    n=n,
                    context=context or {}
                )
                
                if result.get('error'):
                    errors.append({
                        'user_id': user_id,
                        'error': result['error']
                    })
                else:
                    results.append({
                        'user_id': user_id,
                        'recommendations': result.get('recommendations', []),
                        'metadata': result.get('metadata', {})
                    })
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(user_ids)} users")
                    
            except Exception as e:
                logger.error(f"Error for user {user_id}: {e}")
                errors.append({
                    'user_id': user_id,
                    'error': str(e)
                })
        
        total_time = (time.time() - start_time) * 1000
        
        batch_result = {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'total_users': len(user_ids),
            'successful': len(results),
            'failed': len(errors),
            'processing_time_ms': round(total_time, 2),
            'avg_time_per_user_ms': round(total_time / len(user_ids), 2) if user_ids else 0,
            'results': results,
            'errors': errors
        }
        
        logger.info(
            f"Batch complete: {len(results)} successful, "
            f"{len(errors)} failed in {total_time:.2f}ms"
        )
        
        return batch_result
    
    def generate_from_file(
        self,
        input_file: str,
        n: int = 10,
        output_format: str = 'json',
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate recommendations from a file containing user IDs.
        
        Args:
            input_file: Path to input file (CSV or JSON)
            n: Number of recommendations per user
            output_format: Output format (json, csv, parquet)
            context: Optional context dictionary
            
        Returns:
            Path to output file
        """
        user_ids = self._load_user_ids(input_file)
        
        if not user_ids:
            raise ValueError(f"No user IDs found in {input_file}")
        
        logger.info(f"Loaded {len(user_ids)} user IDs from {input_file}")
        
        # Generate recommendations
        batch_result = self.generate_for_users(user_ids, n, context)
        
        # Export results
        output_path = self.export_manager.export_batch(
            batch_result=batch_result,
            format=output_format
        )
        
        logger.info(f"Batch recommendations exported to {output_path}")
        return output_path
    
    def _load_user_ids(self, input_file: str) -> List[int]:
        """
        Load user IDs from a file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of user IDs
        """
        user_ids = []
        file_path = Path(input_file)
        
        if not file_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return []
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                user_ids = self._load_from_csv(file_path)
            elif suffix == '.json':
                user_ids = self._load_from_json(file_path)
            else:
                logger.error(f"Unsupported file format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading user IDs: {e}")
        
        return user_ids
    
    def _load_from_csv(self, file_path: Path) -> List[int]:
        """Load user IDs from CSV file."""
        user_ids = []
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Try common column names
            id_columns = ['user_id', 'userId', 'user', 'id', 'customer_id']
            id_column = None
            
            if reader.fieldnames:
                for col in id_columns:
                    if col in reader.fieldnames:
                        id_column = col
                        break
                
                if not id_column:
                    id_column = reader.fieldnames[0]
                    logger.warning(f"Using first column '{id_column}' as user ID")
            
            for row in reader:
                try:
                    user_id = int(row.get(id_column, 0))
                    if user_id > 0:
                        user_ids.append(user_id)
                except (ValueError, TypeError):
                    continue
        
        return user_ids
    
    def _load_from_json(self, file_path: Path) -> List[int]:
        """Load user IDs from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of user IDs
            return [int(x) for x in data if isinstance(x, (int, float))]
        elif isinstance(data, dict):
            # Dictionary with user_ids key
            user_ids = data.get('user_ids', data.get('users', []))
            return [int(x) for x in user_ids if isinstance(x, (int, float))]
        
        return []
    
    def export_results(
        self,
        batch_result: Dict[str, Any],
        output_format: str = 'json',
        filename: Optional[str] = None
    ) -> str:
        """
        Export batch results to file.
        
        Args:
            batch_result: Batch result dictionary
            output_format: Output format
            filename: Optional custom filename
            
        Returns:
            Path to output file
        """
        return self.export_manager.export_batch(batch_result, output_format, filename)
    
    def get_batch_stats(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about a batch result.
        
        Args:
            batch_result: Batch result dictionary
            
        Returns:
            Statistics dictionary
        """
        results = batch_result.get('results', [])
        
        total_recommendations = sum(
            len(r.get('recommendations', [])) for r in results
        )
        
        return {
            'total_users': batch_result.get('total_users', 0),
            'successful': batch_result.get('successful', 0),
            'failed': batch_result.get('failed', 0),
            'total_recommendations': total_recommendations,
            'avg_recommendations_per_user': (
                total_recommendations / batch_result.get('successful', 1)
            ),
            'processing_time_ms': batch_result.get('processing_time_ms', 0),
            'success_rate': (
                batch_result.get('successful', 0) / 
                batch_result.get('total_users', 1)
            )
        }
