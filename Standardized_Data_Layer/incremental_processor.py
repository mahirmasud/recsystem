"""
Incremental processor for handling incremental data updates.
Supports delta processing, append-only ingestion, and change tracking.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


@dataclass
class IncrementalState:
    """State tracking for incremental processing."""
    entity_type: str
    last_processed_timestamp: Optional[str] = None
    last_processed_id: Optional[Any] = None
    total_rows_processed: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class IncrementalProcessor:
    """
    Handles incremental data processing and delta updates.
    
    Features:
    - Timestamp-based incremental processing
    - ID-based change detection
    - Append-only ingestion
    - Delta computation
    - State persistence
    """
    
    def __init__(self, state_path: Optional[str] = None):
        """
        Initialize incremental processor.
        
        Args:
            state_path: Path to store state files
        """
        self.state_path = Path(state_path) if state_path else Constants.OUTPUT_DIR / 'incremental_state'
        self.state_path.mkdir(parents=True, exist_ok=True)
        
        self.states: Dict[str, IncrementalState] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Load existing state from disk."""
        state_file = self.state_path / 'state.json'
        
        if state_file.exists():
            try:
                import json
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                for entity_type, state_data in data.items():
                    self.states[entity_type] = IncrementalState(**state_data)
                
                logger.info(f"Loaded incremental state for {len(self.states)} entities")
            except Exception as e:
                logger.warning(f"Failed to load incremental state: {e}")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        state_file = self.state_path / 'state.json'
        
        import json
        data = {
            entity_type: state.__dict__ 
            for entity_type, state in self.states.items()
        }
        
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug("Saved incremental state")
    
    def process_incremental(
        self,
        df: pd.DataFrame,
        entity_type: str,
        timestamp_column: Optional[str] = None,
        id_column: Optional[str] = None,
        mode: str = 'append'  # append, upsert, update
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process incremental data update.
        
        Args:
            df: New/incoming DataFrame
            entity_type: Entity type name
            timestamp_column: Column containing timestamps
            id_column: Column containing unique IDs
            mode: Processing mode (append, upsert, update)
        
        Returns:
            Tuple of (delta DataFrame, processing info)
        """
        logger.info(f"Processing incremental update for {entity_type} in {mode} mode")
        
        state = self.states.get(entity_type)
        
        if state is None:
            # First run - all data is new
            state = IncrementalState(entity_type=entity_type)
            self.states[entity_type] = state
            
            info = {
                'mode': mode,
                'new_rows': len(df),
                'updated_rows': 0,
                'skipped_rows': 0,
                'is_first_run': True
            }
            
            # Update state
            self._update_state_from_df(state, df, timestamp_column, id_column)
            self._save_state()
            
            return df, info
        
        # Filter to only new/changed records
        delta_df, info = self._compute_delta(
            df, state, timestamp_column, id_column, mode
        )
        
        if len(delta_df) > 0:
            # Update state
            self._update_state_from_df(state, delta_df, timestamp_column, id_column)
            self._save_state()
        
        logger.info(f"Incremental processing complete: {info['new_rows']} new, {info['updated_rows']} updated")
        return delta_df, info
    
    def _compute_delta(
        self,
        df: pd.DataFrame,
        state: IncrementalState,
        timestamp_column: Optional[str],
        id_column: Optional[str],
        mode: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute delta between current data and state."""
        new_rows = 0
        updated_rows = 0
        skipped_rows = 0
        
        if mode == 'append':
            # Simple append - filter by timestamp or take all
            if timestamp_column and state.last_processed_timestamp:
                mask = df[timestamp_column] > state.last_processed_timestamp
                delta_df = df[mask].copy()
                new_rows = len(delta_df)
                skipped_rows = len(df) - new_rows
            else:
                delta_df = df.copy()
                new_rows = len(df)
            
            info = {
                'mode': mode,
                'new_rows': new_rows,
                'updated_rows': 0,
                'skipped_rows': skipped_rows,
                'is_first_run': False
            }
            
            return delta_df, info
        
        elif mode == 'upsert':
            # Upsert - need to track by ID
            if not id_column:
                logger.warning("Upsert mode requires id_column, falling back to append")
                return df.copy(), {'mode': 'upsert', 'new_rows': len(df), 'updated_rows': 0, 'skipped_rows': 0}
            
            # For now, treat all as new (in production, would compare with existing)
            delta_df = df.copy()
            new_rows = len(df)
            
            info = {
                'mode': mode,
                'new_rows': new_rows,
                'updated_rows': 0,
                'skipped_rows': 0,
                'is_first_run': False
            }
            
            return delta_df, info
        
        else:  # update
            # Update mode - only process changed records
            if timestamp_column and state.last_processed_timestamp:
                mask = df[timestamp_column] > state.last_processed_timestamp
                delta_df = df[mask].copy()
                updated_rows = len(delta_df)
                skipped_rows = len(df) - updated_rows
            else:
                delta_df = pd.DataFrame(columns=df.columns)
                skipped_rows = len(df)
            
            info = {
                'mode': mode,
                'new_rows': 0,
                'updated_rows': updated_rows,
                'skipped_rows': skipped_rows,
                'is_first_run': False
            }
            
            return delta_df, info
    
    def _update_state_from_df(
        self,
        state: IncrementalState,
        df: pd.DataFrame,
        timestamp_column: Optional[str],
        id_column: Optional[str]
    ) -> None:
        """Update state based on processed DataFrame."""
        if len(df) == 0:
            return
        
        state.total_rows_processed += len(df)
        state.last_updated = datetime.now().isoformat()
        
        if timestamp_column and timestamp_column in df.columns:
            max_ts = df[timestamp_column].max()
            if pd.notna(max_ts):
                state.last_processed_timestamp = str(max_ts)
        
        if id_column and id_column in df.columns:
            max_id = df[id_column].max()
            if pd.notna(max_id):
                state.last_processed_id = max_id
    
    def get_state(self, entity_type: str) -> Optional[IncrementalState]:
        """Get current state for an entity."""
        return self.states.get(entity_type)
    
    def reset_state(self, entity_type: str) -> None:
        """Reset state for an entity."""
        if entity_type in self.states:
            del self.states[entity_type]
            self._save_state()
            logger.info(f"Reset incremental state for {entity_type}")
    
    def reset_all_states(self) -> None:
        """Reset all incremental states."""
        self.states = {}
        self._save_state()
        logger.info("Reset all incremental states")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all incremental processing states."""
        return {
            'total_entities': len(self.states),
            'entities': {
                entity_type: {
                    'last_processed_timestamp': state.last_processed_timestamp,
                    'last_processed_id': state.last_processed_id,
                    'total_rows_processed': state.total_rows_processed,
                    'last_updated': state.last_updated
                }
                for entity_type, state in self.states.items()
            }
        }
    
    def export_state(self, output_path: Optional[str] = None) -> Path:
        """
        Export current state to file.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path) if output_path else self.state_path / 'exported_state.json'
        
        import json
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'summary': self.get_processing_summary(),
            'states': {
                entity_type: state.__dict__
                for entity_type, state in self.states.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported incremental state to {output_path}")
        return output_path
