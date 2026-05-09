"""Main entry point for the Validation Layer pipeline."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from shared.logger import get_logger
from shared.constants import Constants
from shared.file_loader import FileLoader
from Validation_Layer.validator import DataValidator
from Validation_Layer.report_generator import ValidationReportGenerator
from Validation_Layer.invalid_handler import InvalidRecordHandler


logger = get_logger(__name__)


class ValidationPipeline:
    """Main pipeline orchestrator for data validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validation pipeline."""
        self.config_path = config_path
        self.validator = DataValidator(config_path)
        self.report_generator = ValidationReportGenerator()
        self.invalid_handler = InvalidRecordHandler()
    
    def run(
        self,
        dataframes: Dict[str, pd.DataFrame],
        generate_report: bool = True,
        save_invalid: bool = True
    ) -> Dict[str, Any]:
        """Run the full validation pipeline."""
        start_time = time.time()
        logger.info("Starting validation pipeline")
        
        # Run validation
        results = self.validator.validate(dataframes)
        
        # Generate report
        if generate_report:
            report_path = self.report_generator.generate_report(results)
            results['report_path'] = report_path
        
        # Save invalid records
        if save_invalid:
            for entity_type, df in dataframes.items():
                entity_result = results['entity_results'].get(entity_type, {})
                all_issues = (
                    entity_result.get('null_issues', []) +
                    entity_result.get('type_issues', []) +
                    entity_result.get('range_issues', []) +
                    entity_result.get('uniqueness_issues', [])
                )
                
                if all_issues:
                    invalid_df = self.invalid_handler.extract_invalid_records(
                        df, all_issues, entity_type
                    )
                    if not invalid_df.empty:
                        self.invalid_handler.save_invalid_records(
                            invalid_df, entity_type
                        )
        
        duration = time.time() - start_time
        results['duration_seconds'] = duration
        
        logger.info(f"Validation pipeline completed in {duration:.2f}s")
        return results
    
    def run_from_files(
        self,
        file_paths: Dict[str, str],
        **kwargs
    ) -> Dict[str, Any]:
        """Run pipeline loading from files."""
        dataframes = {}
        
        for entity_type, filepath in file_paths.items():
            try:
                df = FileLoader.load_parquet(filepath)
                dataframes[entity_type] = df
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                raise
        
        return self.run(dataframes, **kwargs)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return self.validator.get_summary()


def main():
    """CLI entry point for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data validation pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to rec_config.json')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    Constants.ensure_directories()
    pipeline = ValidationPipeline(args.config)
    
    print("Validation pipeline ready.")
    print(f"Output directory: {Constants.REPORTS_DIR}")


if __name__ == '__main__':
    main()
