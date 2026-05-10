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


def create_sample_data(config_reader: ConfigReader) -> Dict[str, pd.DataFrame]:
    """
    Create sample data based on configuration mappings.
    
    Args:
        config_reader: ConfigReader instance with loaded configuration
        
    Returns:
        Dictionary mapping entity types to sample DataFrames
    """
    sample_data = {}
    
    trusted_mappings = config_reader.get_trusted_mappings()
    
    if not trusted_mappings:
        logger.warning("No trusted mappings found in config, cannot generate sample data")
        return sample_data
    
    for entity_type, column_mappings in trusted_mappings.items():
        try:
            # Extract source columns and create sample data
            sample_rows = []
            num_samples = 5  # Generate 5 sample rows per entity
            
            for i in range(num_samples):
                row = {}
                for canonical_col, mapping_info in column_mappings.items():
                    source_col = mapping_info.get('source_column', canonical_col)
                    dtype = mapping_info.get('dtype', 'string')
                    
                    # Generate sample values based on dtype and column name
                    if 'id' in canonical_col.lower() or 'id' in source_col.lower():
                        row[source_col] = i + 1
                    elif dtype in ['int64', 'int32', 'int']:
                        if 'age' in canonical_col.lower() or 'age' in source_col.lower():
                            row[source_col] = 25 + i * 5
                        elif 'quantity' in canonical_col.lower() or 'qty' in source_col.lower():
                            row[source_col] = (i % 5) + 1
                        elif 'price' in canonical_col.lower() or 'cost' in canonical_col.lower():
                            row[source_col] = int(10.0 + i * 5.5)
                        else:
                            row[source_col] = i * 100
                    elif dtype in ['float64', 'float32', 'float']:
                        if 'price' in canonical_col.lower() or 'cost' in canonical_col.lower():
                            row[source_col] = 10.0 + i * 5.5
                        elif 'discount' in canonical_col.lower():
                            row[source_col] = i * 0.1
                        elif 'total' in canonical_col.lower() or 'amount' in canonical_col.lower():
                            row[source_col] = 50.0 + i * 25.0
                        else:
                            row[source_col] = float(i * 10.5)
                    elif dtype in ['datetime64[ns]', 'datetime']:
                        # Generate timestamps
                        base_ts = pd.Timestamp('2024-01-01')
                        row[source_col] = base_ts + pd.Timedelta(days=i)
                    elif 'email' in canonical_col.lower() or 'email' in source_col.lower():
                        row[source_col] = f'user{i+1}@example.com'
                    elif 'country' in canonical_col.lower() or 'country' in source_col.lower():
                        countries = ['US', 'UK', 'CA', 'DE', 'FR']
                        row[source_col] = countries[i % len(countries)]
                    elif 'gender' in canonical_col.lower() or 'gender' in source_col.lower():
                        genders = ['M', 'F', 'O']
                        row[source_col] = genders[i % len(genders)]
                    elif 'name' in canonical_col.lower() or 'label' in source_col.lower():
                        row[source_col] = f'{entity_type.capitalize()}_{i+1}'
                    elif 'type' in canonical_col.lower() or 'category' in canonical_col.lower():
                        types = ['TypeA', 'TypeB', 'TypeC']
                        row[source_col] = types[i % len(types)]
                    elif 'device' in canonical_col.lower():
                        devices = ['mobile', 'desktop', 'tablet']
                        row[source_col] = devices[i % len(devices)]
                    elif 'session' in canonical_col.lower():
                        row[source_col] = f'session_{i+1}_abc'
                    else:
                        # Default string value
                        row[source_col] = f'value_{i+1}'
                
                sample_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(sample_rows)
            sample_data[entity_type] = df
            logger.info(f"Generated sample data for {entity_type} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to generate sample data for {entity_type}: {e}")
    
    return sample_data


def main():
    """CLI entry point for standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data standardization pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to rec_config.json')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate after standardization')
    parser.add_argument('--no-write', action='store_true', help='Skip writing output files')
    parser.add_argument('--sample', action='store_true', help='Generate and process sample data')
    parser.add_argument('--files', type=str, nargs='+', help='Source data files (format: entity_type=file_path)')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Constants.ensure_directories()
    
    # Initialize pipeline
    pipeline = StandardizationPipeline(args.config)
    
    # Check if files provided
    if args.files:
        # Parse file arguments
        file_paths = {}
        for file_arg in args.files:
            if '=' in file_arg:
                entity_type, filepath = file_arg.split('=', 1)
                file_paths[entity_type] = filepath
            else:
                logger.error(f"Invalid file argument format: {file_arg}. Use entity_type=file_path")
                return
        
        print(f"Running standardization from files: {file_paths}")
        results = pipeline.run_from_files(
            file_paths,
            validate=args.validate,
            write_output=not args.no_write
        )
        
    elif args.sample:
        # Generate and process sample data
        print("Generating sample data from configuration...")
        sample_data = create_sample_data(pipeline.config_reader)
        
        if not sample_data:
            print("ERROR: Could not generate sample data. Check your rec_config.json")
            return
        
        print(f"Generated sample data for {len(sample_data)} entities:")
        for entity_type, df in sample_data.items():
            print(f"  - {entity_type}: {len(df)} rows, {len(df.columns)} columns")
        
        print("\nRunning standardization pipeline...")
        results = pipeline.run(
            sample_data,
            validate=args.validate,
            write_output=not args.no_write
        )
        
    else:
        # Interactive mode - show instructions
        print("Standardization Pipeline Initialized")
        print(f"Config: {pipeline.config_path}")
        print(f"Output directory: {Constants.STANDARDIZED_DIR}")
        print("\nTo run standardization, use one of the following options:\n")
        print("1. Process sample data (generated from config):")
        print("   python cli.py standardize --config output/rec_config.json --sample\n")
        print("2. Process actual data files:")
        print("   python cli.py standardize --config output/rec_config.json --files users=data/users.csv items=data/items.csv\n")
        print("3. Use Python API:")
        print("   from Standardized_Data_Layer.run import StandardizationPipeline")
        print("   pipeline = StandardizationPipeline('output/rec_config.json')")
        print("   pipeline.run_from_files({'users': 'data/users.csv', 'items': 'data/items.csv'})\n")
        return
    
    # Print results
    print("\n" + "="*60)
    print("STANDARDIZATION RESULTS")
    print("="*60)
    print(f"Success: {results.get('success', False)}")
    print(f"Entities processed: {len(results.get('entities_processed', []))}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    
    if results.get('summary'):
        summary = results['summary']
        print(f"Total rows: {summary.get('total_rows', 0)}")
        print(f"Errors: {summary.get('error_count', 0)}")
        print(f"Warnings: {summary.get('warning_count', 0)}")
    
    if results.get('errors'):
        print("\nErrors encountered:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    if results.get('warnings'):
        print("\nWarnings:")
        for warning in results['warnings'][:5]:  # Show first 5 warnings
            print(f"  - {warning}")
    
    print("\nOutput files written to:", Constants.STANDARDIZED_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
