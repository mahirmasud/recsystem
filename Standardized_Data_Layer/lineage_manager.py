"""
Lineage manager for tracking data transformations.
Records source-to-target mappings and transformation history.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


class LineageManager:
    """
    Manages data lineage metadata for standardization processes.
    
    Tracks:
    - Source columns and tables
    - Transformation formulas
    - Timestamp of transformations
    - Version information
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize lineage manager.
        
        Args:
            output_dir: Directory for lineage files
        """
        self.output_dir = Path(output_dir) if output_dir else Constants.REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_records: Dict[str, Any] = {}
    
    def record_transformation(
        self,
        entity_type: str,
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Record a data transformation.
        
        Args:
            entity_type: Entity type being transformed
            source_info: Information about source data
            target_info: Information about target data
            transformations: List of transformation steps
        
        Returns:
            Lineage record dictionary
        """
        timestamp = datetime.now().isoformat()
        
        record = {
            'entity_type': entity_type,
            'timestamp': timestamp,
            'source': source_info,
            'target': target_info,
            'transformations': transformations,
            'version': '1.0'
        }
        
        self.lineage_records[entity_type] = record
        logger.info(f"Recorded lineage for {entity_type}")
        
        return record
    
    def record_column_mapping(
        self,
        entity_type: str,
        column_name: str,
        source_column: str,
        source_table: str,
        transformation: str = 'direct_mapping',
        formula: Optional[str] = None
    ) -> None:
        """
        Record a single column mapping.
        
        Args:
            entity_type: Entity type
            column_name: Target canonical column name
            source_column: Source column name
            source_table: Source table name
            transformation: Type of transformation
            formula: Formula if computed column
        """
        if entity_type not in self.lineage_records:
            self.lineage_records[entity_type] = {
                'entity_type': entity_type,
                'timestamp': datetime.now().isoformat(),
                'columns': {},
                'transformations': []
            }
        
        self.lineage_records[entity_type]['columns'][column_name] = {
            'source_column': source_column,
            'source_table': source_table,
            'transformation': transformation,
            'formula': formula
        }
        
        logger.debug(f"Recorded column mapping: {source_column} -> {column_name}")
    
    def record_derived_metric(
        self,
        entity_type: str,
        metric_name: str,
        formula: str,
        source_columns: List[str],
        description: str = ''
    ) -> None:
        """
        Record a derived metric calculation.
        
        Args:
            entity_type: Entity type
            metric_name: Name of derived metric
            formula: Calculation formula
            source_columns: Columns used in calculation
            description: Metric description
        """
        if entity_type not in self.lineage_records:
            self.lineage_records[entity_type] = {
                'entity_type': entity_type,
                'timestamp': datetime.now().isoformat(),
                'columns': {},
                'derived_metrics': {}
            }
        
        if 'derived_metrics' not in self.lineage_records[entity_type]:
            self.lineage_records[entity_type]['derived_metrics'] = {}
        
        self.lineage_records[entity_type]['derived_metrics'][metric_name] = {
            'formula': formula,
            'source_columns': source_columns,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        logger.debug(f"Recorded derived metric: {metric_name}")
    
    def get_lineage(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Get lineage record for an entity type.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Lineage record or None
        """
        return self.lineage_records.get(entity_type)
    
    def get_all_lineage(self) -> Dict[str, Any]:
        """
        Get all lineage records.
        
        Returns:
            Dictionary of all lineage records
        """
        return self.lineage_records
    
    def save_lineage(self, filename: str = 'data_lineage.json') -> Path:
        """
        Save lineage records to file.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.lineage_records, f, indent=2, default=str)
        
        logger.info(f"Saved lineage to {filepath}")
        return filepath
    
    def load_lineage(self, filepath: str) -> Dict[str, Any]:
        """
        Load lineage records from file.
        
        Args:
            filepath: Path to lineage file
        
        Returns:
            Loaded lineage records
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            self.lineage_records = json.load(f)
        
        logger.info(f"Loaded lineage from {filepath}")
        return self.lineage_records
    
    def build_full_lineage(
        self,
        entity_type: str,
        config_reader
    ) -> Dict[str, Any]:
        """
        Build complete lineage from configuration.
        
        Args:
            entity_type: Entity type
            config_reader: ConfigReader instance
        
        Returns:
            Complete lineage record
        """
        mappings = config_reader.get_entity_mapping(entity_type)
        
        for canonical, details in mappings.items():
            self.record_column_mapping(
                entity_type=entity_type,
                column_name=canonical,
                source_column=details['source_column'],
                source_table=details.get('source_table', ''),
                transformation='direct_mapping'
            )
        
        # Add business meanings as derived metrics
        business_meanings = config_reader.get_business_meanings()
        for metric_name, definition in business_meanings.items():
            self.record_derived_metric(
                entity_type=entity_type,
                metric_name=metric_name,
                formula=definition.get('formula', ''),
                source_columns=[],
                description=definition.get('description', '')
            )
        
        return self.get_lineage(entity_type)
