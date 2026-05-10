"""
Metadata tracker for tracking schema and transformation metadata.
Provides comprehensive metadata management for standardization processes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    dtype: str
    source_column: str
    source_table: Optional[str] = None
    semantic_type: Optional[str] = None
    null_count: int = 0
    unique_count: int = 0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None


@dataclass
class EntityMetadata:
    """Metadata for an entity type."""
    entity_type: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    primary_key: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_dataset: Optional[str] = None
    quality_score: float = 1.0


@dataclass
class ProcessingMetadata:
    """Metadata for a processing run."""
    run_id: str
    pipeline_name: str
    started_at: str
    completed_at: Optional[str] = None
    entities_processed: List[str] = field(default_factory=list)
    total_rows_processed: int = 0
    status: str = 'running'  # running, completed, failed
    error_message: Optional[str] = None
    configuration_hash: Optional[str] = None


class MetadataTracker:
    """
    Comprehensive metadata tracking for standardization.
    
    Features:
    - Column-level metadata
    - Entity-level statistics
    - Processing run tracking
    - Quality metrics
    - Schema evolution history
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize metadata tracker.
        
        Args:
            metadata_path: Path to store metadata files
        """
        self.metadata_path = Path(metadata_path) if metadata_path else Constants.OUTPUT_DIR / 'metadata'
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        self.entity_metadata: Dict[str, EntityMetadata] = {}
        self.processing_runs: List[ProcessingMetadata] = []
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load existing metadata from disk."""
        entity_file = self.metadata_path / 'entity_metadata.json'
        runs_file = self.metadata_path / 'processing_runs.json'
        
        if entity_file.exists():
            try:
                with open(entity_file, 'r') as f:
                    data = json.load(f)
                
                for entity_type, entity_data in data.items():
                    columns = {}
                    for col_name, col_data in entity_data.get('columns', {}).items():
                        columns[col_name] = ColumnMetadata(**col_data)
                    
                    self.entity_metadata[entity_type] = EntityMetadata(
                        entity_type=entity_type,
                        row_count=entity_data.get('row_count', 0),
                        column_count=entity_data.get('column_count', 0),
                        columns=columns,
                        primary_key=entity_data.get('primary_key'),
                        created_at=entity_data.get('created_at', ''),
                        updated_at=entity_data.get('updated_at', ''),
                        source_dataset=entity_data.get('source_dataset'),
                        quality_score=entity_data.get('quality_score', 1.0)
                    )
                
                logger.info(f"Loaded metadata for {len(self.entity_metadata)} entities")
            except Exception as e:
                logger.warning(f"Failed to load entity metadata: {e}")
        
        if runs_file.exists():
            try:
                with open(runs_file, 'r') as f:
                    data = json.load(f)
                
                self.processing_runs = [ProcessingMetadata(**r) for r in data]
                logger.info(f"Loaded {len(self.processing_runs)} processing runs")
            except Exception as e:
                logger.warning(f"Failed to load processing runs: {e}")
    
    def track_entity_metadata(
        self,
        entity_type: str,
        df,
        source_dataset: Optional[str] = None,
        column_mappings: Optional[Dict[str, str]] = None
    ) -> EntityMetadata:
        """
        Track metadata for an entity.
        
        Args:
            entity_type: Entity type name
            df: DataFrame to analyze
            source_dataset: Source dataset name
            column_mappings: Mapping of canonical to source columns
        
        Returns:
            Created entity metadata
        """
        import pandas as pd
        
        logger.info(f"Tracking metadata for {entity_type}")
        
        columns = {}
        
        for col in df.columns:
            series = df[col]
            
            col_meta = ColumnMetadata(
                name=col,
                dtype=str(series.dtype),
                source_column=column_mappings.get(col, col) if column_mappings else col,
                null_count=int(series.isnull().sum()),
                unique_count=int(series.nunique())
            )
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(series):
                col_meta.min_value = float(series.min()) if not series.empty else None
                col_meta.max_value = float(series.max()) if not series.empty else None
                col_meta.mean_value = float(series.mean()) if not series.empty else None
            
            columns[col] = col_meta
        
        # Calculate quality score
        total_cells = len(df) * len(df.columns)
        null_cells = sum(cm.null_count for cm in columns.values())
        quality_score = 1.0 - (null_cells / total_cells) if total_cells > 0 else 1.0
        
        metadata = EntityMetadata(
            entity_type=entity_type,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            primary_key=self._detect_primary_key(df, entity_type),
            source_dataset=source_dataset,
            quality_score=quality_score,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.entity_metadata[entity_type] = metadata
        self._save_entity_metadata()
        
        return metadata
    
    def _detect_primary_key(self, df, entity_type: str) -> Optional[str]:
        """Detect potential primary key column."""
        # Common PK patterns
        pk_candidates = [f'{entity_type[:-1]}_id' if entity_type.endswith('s') else f'{entity_type}_id',
                        'id', f'{entity_type}_identifier']
        
        for candidate in pk_candidates:
            if candidate in df.columns:
                if df[candidate].is_unique:
                    return candidate
        
        # Check for any unique ID column
        for col in df.columns:
            if 'id' in col.lower() and df[col].is_unique:
                return col
        
        return None
    
    def start_processing_run(
        self,
        pipeline_name: str,
        configuration_hash: Optional[str] = None
    ) -> ProcessingMetadata:
        """
        Start tracking a processing run.
        
        Args:
            pipeline_name: Name of the pipeline
            configuration_hash: Hash of configuration used
        
        Returns:
            Processing metadata record
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_runs) + 1}"
        
        run = ProcessingMetadata(
            run_id=run_id,
            pipeline_name=pipeline_name,
            started_at=datetime.now().isoformat(),
            configuration_hash=configuration_hash
        )
        
        self.processing_runs.append(run)
        self._save_processing_runs()
        
        logger.info(f"Started processing run: {run_id}")
        return run
    
    def complete_processing_run(
        self,
        run_id: str,
        entities_processed: List[str],
        total_rows: int,
        status: str = 'completed',
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark a processing run as complete.
        
        Args:
            run_id: Run ID to complete
            entities_processed: List of processed entities
            total_rows: Total rows processed
            status: Final status
            error_message: Error message if failed
        """
        for run in self.processing_runs:
            if run.run_id == run_id:
                run.completed_at = datetime.now().isoformat()
                run.entities_processed = entities_processed
                run.total_rows_processed = total_rows
                run.status = status
                run.error_message = error_message
                break
        
        self._save_processing_runs()
        logger.info(f"Completed processing run: {run_id} with status {status}")
    
    def get_entity_statistics(self, entity_type: str) -> Dict[str, Any]:
        """
        Get statistics for an entity.
        
        Args:
            entity_type: Entity type
        
        Returns:
            Statistics dictionary
        """
        if entity_type not in self.entity_metadata:
            return {'error': f'No metadata found for {entity_type}'}
        
        meta = self.entity_metadata[entity_type]
        
        return {
            'entity_type': entity_type,
            'row_count': meta.row_count,
            'column_count': meta.column_count,
            'quality_score': meta.quality_score,
            'primary_key': meta.primary_key,
            'null_summary': {
                col: cm.null_count for col, cm in meta.columns.items()
            },
            'dtype_summary': {
                col: cm.dtype for col, cm in meta.columns.items()
            }
        }
    
    def get_all_entity_types(self) -> List[str]:
        """Get all tracked entity types."""
        return list(self.entity_metadata.keys())
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate overall quality report."""
        if not self.entity_metadata:
            return {'error': 'No entity metadata available'}
        
        avg_quality = sum(m.quality_score for m in self.entity_metadata.values()) / len(self.entity_metadata)
        total_rows = sum(m.row_count for m in self.entity_metadata.values())
        total_columns = sum(m.column_count for m in self.entity_metadata.values())
        
        return {
            'total_entities': len(self.entity_metadata),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'average_quality_score': avg_quality,
            'entities': {
                entity: {
                    'quality_score': meta.quality_score,
                    'row_count': meta.row_count,
                    'null_percentage': round(
                        (1 - meta.quality_score) * 100, 2
                    )
                }
                for entity, meta in self.entity_metadata.items()
            }
        }
    
    def _save_entity_metadata(self) -> None:
        """Save entity metadata to disk."""
        entity_file = self.metadata_path / 'entity_metadata.json'
        
        data = {}
        for entity_type, meta in self.entity_metadata.items():
            data[entity_type] = {
                'entity_type': meta.entity_type,
                'row_count': meta.row_count,
                'column_count': meta.column_count,
                'columns': {col: cm.__dict__ for col, cm in meta.columns.items()},
                'primary_key': meta.primary_key,
                'created_at': meta.created_at,
                'updated_at': meta.updated_at,
                'source_dataset': meta.source_dataset,
                'quality_score': meta.quality_score
            }
        
        with open(entity_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug("Saved entity metadata")
    
    def _save_processing_runs(self) -> None:
        """Save processing runs to disk."""
        runs_file = self.metadata_path / 'processing_runs.json'
        
        data = [run.__dict__ for run in self.processing_runs]
        
        with open(runs_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {len(self.processing_runs)} processing runs")
    
    def export_metadata(self, output_path: Optional[str] = None) -> Path:
        """
        Export all metadata to file.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path) if output_path else self.metadata_path / 'schema_metadata.json'
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'entity_count': len(self.entity_metadata),
            'entities': {
                entity: self.get_entity_statistics(entity)
                for entity in self.entity_metadata
            },
            'quality_report': self.get_quality_report(),
            'recent_runs': [
                run.__dict__ for run in self.processing_runs[-10:]
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metadata to {output_path}")
        return output_path
    
    def clear(self) -> None:
        """Clear all metadata."""
        self.entity_metadata = {}
        self.processing_runs = []
        self._save_entity_metadata()
        self._save_processing_runs()
