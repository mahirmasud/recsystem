"""
Sync manager for coordinating data synchronization across layers.
Ensures consistency between standardized, validated, and feature datasets.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


class SyncManager:
    """
    Manages synchronization state across data processing layers.
    
    Tracks:
    - Which entities have been processed
    - Processing timestamps
    - Version information
    - Dependencies between datasets
    """
    
    SYNC_FILE = 'sync_state.json'
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize sync manager.
        
        Args:
            output_dir: Directory for sync state file
        """
        self.output_dir = Path(output_dir) if output_dir else Constants.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sync_file = self.output_dir / self.SYNC_FILE
        self.state: Dict[str, Any] = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load sync state from file."""
        if self.sync_file.exists():
            try:
                with open(self.sync_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
        
        return {
            'last_updated': None,
            'entities': {},
            'pipeline_runs': []
        }
    
    def _save_state(self) -> None:
        """Save sync state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        with open(self.sync_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.debug(f"Saved sync state to {self.sync_file}")
    
    def mark_entity_processed(
        self,
        entity_type: str,
        stage: str,
        row_count: int = 0,
        checksum: Optional[str] = None
    ) -> None:
        """
        Mark an entity as processed for a stage.
        
        Args:
            entity_type: Entity type name
            stage: Processing stage (standardized, validated, features)
            row_count: Number of rows processed
            checksum: Optional data checksum
        """
        if entity_type not in self.state['entities']:
            self.state['entities'][entity_type] = {
                'stages': {}
            }
        
        self.state['entities'][entity_type]['stages'][stage] = {
            'processed_at': datetime.now().isoformat(),
            'row_count': row_count,
            'checksum': checksum,
            'status': 'completed'
        }
        
        self._save_state()
        logger.info(f"Marked {entity_type} as processed for stage {stage}")
    
    def is_entity_processed(
        self,
        entity_type: str,
        stage: str
    ) -> bool:
        """
        Check if entity has been processed for a stage.
        
        Args:
            entity_type: Entity type name
            stage: Processing stage
        
        Returns:
            True if processed
        """
        if entity_type not in self.state['entities']:
            return False
        
        stages = self.state['entities'][entity_type].get('stages', {})
        return stage in stages and stages[stage].get('status') == 'completed'
    
    def get_processing_status(self, entity_type: str) -> Dict[str, Any]:
        """
        Get processing status for an entity.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Status dictionary
        """
        if entity_type not in self.state['entities']:
            return {'status': 'not_started', 'stages': {}}
        
        entity_state = self.state['entities'][entity_type]
        stages = entity_state.get('stages', {})
        
        completed_stages = [s for s, info in stages.items() if info.get('status') == 'completed']
        
        return {
            'status': 'in_progress' if completed_stages else 'not_started',
            'completed_stages': completed_stages,
            'stages': stages
        }
    
    def record_pipeline_run(
        self,
        pipeline_name: str,
        entities_processed: List[str],
        duration_seconds: float,
        success: bool = True
    ) -> None:
        """
        Record a pipeline execution.
        
        Args:
            pipeline_name: Name of pipeline
            entities_processed: List of entity types processed
            duration_seconds: Execution duration
            success: Whether pipeline succeeded
        """
        run_record = {
            'pipeline': pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'entities': entities_processed,
            'duration_seconds': duration_seconds,
            'success': success
        }
        
        self.state['pipeline_runs'].append(run_record)
        
        # Keep only last 100 runs
        if len(self.state['pipeline_runs']) > 100:
            self.state['pipeline_runs'] = self.state['pipeline_runs'][-100:]
        
        self._save_state()
        logger.info(f"Recorded pipeline run: {pipeline_name}")
    
    def check_dependencies_ready(
        self,
        required_entities: List[str],
        required_stage: str
    ) -> tuple[bool, List[str]]:
        """
        Check if required entities are ready at specified stage.
        
        Args:
            required_entities: List of entity types needed
            required_stage: Required processing stage
        
        Returns:
            Tuple of (all_ready, missing_entities)
        """
        missing = []
        
        for entity in required_entities:
            if not self.is_entity_processed(entity, required_stage):
                missing.append(entity)
        
        return len(missing) == 0, missing
    
    def get_all_synced_entities(self, stage: str) -> List[str]:
        """
        Get all entities synced at a stage.
        
        Args:
            stage: Processing stage
        
        Returns:
            List of entity types
        """
        synced = []
        
        for entity, info in self.state['entities'].items():
            stages = info.get('stages', {})
            if stage in stages and stages[stage].get('status') == 'completed':
                synced.append(entity)
        
        return synced
    
    def clear_entity_state(self, entity_type: str) -> None:
        """
        Clear state for an entity.
        
        Args:
            entity_type: Entity type to clear
        """
        if entity_type in self.state['entities']:
            del self.state['entities'][entity_type]
            self._save_state()
            logger.info(f"Cleared state for {entity_type}")
    
    def clear_all_state(self) -> None:
        """Clear all sync state."""
        self.state = {
            'last_updated': datetime.now().isoformat(),
            'entities': {},
            'pipeline_runs': []
        }
        self._save_state()
        logger.info("Cleared all sync state")
    
    def get_sync_report(self) -> Dict[str, Any]:
        """
        Generate sync status report.
        
        Returns:
            Report dictionary
        """
        report = {
            'last_updated': self.state.get('last_updated'),
            'total_entities': len(self.state['entities']),
            'total_pipeline_runs': len(self.state['pipeline_runs']),
            'entities': {}
        }
        
        for entity, info in self.state['entities'].items():
            stages = info.get('stages', {})
            report['entities'][entity] = {
                'completed_stages': list(stages.keys()),
                'latest_stage': max(stages.keys(), default=None),
                'row_counts': {
                    stage: info['stages'][stage].get('row_count', 0)
                    for stage in stages
                }
            }
        
        return report
