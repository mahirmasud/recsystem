"""
Main entry point for the Standardized Data Layer pipeline.
Orchestrates the full standardization process.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from shared.logger import get_logger
from shared.constants import Constants
from shared.file_loader import FileLoader
from Standardized_Data_Layer.config_reader import ConfigReader
from Standardized_Data_Layer.dataframe_standardizer import DataFrameStandardizer
from Standardized_Data_Layer.schema_validator import SchemaValidator
from Standardized_Data_Layer.dataset_writer import DatasetWriter
from Standardized_Data_Layer.lineage_manager import LineageManager
from Standardized_Data_Layer.sync_manager import SyncManager


logger = get_logger(__name__)


class StandardizationPipeline:
    """
    Main pipeline orchestrator for data standardization.
    
    Coordinates:
    - Loading source data
    - Applying mappings
    - Validating schemas
    - Writing outputs
    - Tracking lineage
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize standardization pipeline.
        
        Args:
            config_path: Path to rec_config.json
        """
        self.config_path = config_path
        self.config_reader = ConfigReader(config_path)
        self.standardizer = DataFrameStandardizer(config_path)
        self.validator = SchemaValidator()
        self.writer = DatasetWriter()
        self.lineage_manager = LineageManager()
        self.sync_manager = SyncManager()
    
    def run(
        self,
        source_data: Dict[str, pd.DataFrame],
        entity_types: Optional[List[str]] = None,
        validate: bool = True,
        write_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full standardization pipeline.
        
        Args:
            source_data: Dictionary mapping entity type to source DataFrame
            entity_types: List of entity types to process (default: all)
            validate: Whether to validate after standardization
            write_output: Whether to write parquet files
        
        Returns:
            Pipeline results dictionary
        """
        start_time = time.time()
        logger.info("Starting standardization pipeline")
        
        entity_types = entity_types or list(source_data.keys())
        results = {
            'success': True,
            'entities_processed': [],
            'errors': [],
            'warnings': []
        }
        
        standardized_data = {}
        
        for entity_type in entity_types:
            if entity_type not in source_data:
                logger.warning(f"No source data for {entity_type}, skipping")
                results['warnings'].append(f"No source data for {entity_type}")
                continue
            
            try:
                # Standardize
                df_standardized = self.standardizer.standardize(
                    source_data[entity_type],
                    entity_type
                )
                standardized_data[entity_type] = df_standardized
                
                # Validate
                if validate:
                    is_valid, report = self.validator.validate(df_standardized, entity_type)
                    if not is_valid:
                        results['errors'].extend(report.get('errors', []))
                        results['warnings'].extend(report.get('warnings', []))
                
                # Write output
                if write_output:
                    self.writer.write(df_standardized, entity_type)
                
                # Record lineage
                self.lineage_manager.build_full_lineage(entity_type, self.config_reader)
                
                # Mark as processed
                self.sync_manager.mark_entity_processed(
                    entity_type,
                    'standardized',
                    row_count=len(df_standardized)
                )
                
                results['entities_processed'].append(entity_type)
                logger.info(f"Completed standardization for {entity_type}")
                
            except Exception as e:
                logger.error(f"Failed to standardize {entity_type}: {e}")
                results['errors'].append({
                    'entity_type': entity_type,
                    'error': str(e)
                })
                results['success'] = False
        
        # Save lineage
        self.lineage_manager.save_lineage()
        
        # Record pipeline run
        duration = time.time() - start_time
        self.sync_manager.record_pipeline_run(
            'standardization',
            results['entities_processed'],
            duration,
            results['success']
        )
        
        results['duration_seconds'] = duration
        results['summary'] = self._build_summary(standardized_data, results)
        
        logger.info(f"Standardization pipeline completed in {duration:.2f}s")
        return results
    
    def run_from_files(
        self,
        file_paths: Dict[str, str],
        entity_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run pipeline loading from files.
        
        Args:
            file_paths: Dictionary mapping entity type to file path
            entity_types: List of entity types to process
            **kwargs: Additional arguments passed to run()
        
        Returns:
            Pipeline results
        """
        source_data = {}
        
        for entity_type, filepath in file_paths.items():
            try:
                df = FileLoader.load_auto(filepath)
                source_data[entity_type] = df
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                raise
        
        return self.run(source_data, entity_types, **kwargs)
    
    def _build_summary(
        self,
        data: Dict[str, pd.DataFrame],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build pipeline summary."""
        return {
            'total_entities': len(data),
            'successful_entities': len(results['entities_processed']),
            'total_rows': sum(len(df) for df in data.values()),
            'error_count': len([e for e in results['errors'] if 'error' in e]),
            'warning_count': len(results['warnings'])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self.sync_manager.get_sync_report()


def main():
    """CLI entry point for standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data standardization pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to rec_config.json')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate after standardization')
    parser.add_argument('--no-write', action='store_true', help='Skip writing output files')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Constants.ensure_directories()
    
    # Example usage - in production, source data would come from actual files
    pipeline = StandardizationPipeline(args.config)
    
    print("Standardization pipeline ready.")
    print(f"Config: {pipeline.config_path}")
    print(f"Output directory: {Constants.STANDARDIZED_DIR}")


if __name__ == '__main__':
    main()
