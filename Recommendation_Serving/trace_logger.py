"""
Trace Logger - Logs recommendation traces for debugging and auditing.

Provides detailed logging of recommendation requests, responses,
and processing stages for debugging, auditing, and analytics.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import logging
from shared.logger import get_logger
from shared.constants import Constants

logger = get_logger(__name__)


class TraceEntry:
    """Represents a single trace entry."""
    
    def __init__(
        self,
        trace_id: str,
        stage: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.trace_id = trace_id
        self.stage = stage
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trace_id': self.trace_id,
            'stage': self.stage,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'data': self.data
        }


class TraceLogger:
    """
    Logs recommendation traces for debugging and auditing.
    
    Features:
    - Request/response tracing
    - Stage-by-stage execution tracking
    - Performance timing
    - Error tracking
    - File-based persistence
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        persist_to_file: bool = True
    ):
        """
        Initialize the trace logger.
        
        Args:
            log_dir: Directory for trace logs
            enabled: Whether tracing is enabled
            persist_to_file: Whether to persist traces to file
        """
        self.enabled = enabled
        self.persist_to_file = persist_to_file
        
        if log_dir is None:
            log_dir = Constants.LOGS_DIR
        
        self.log_dir = log_dir
        self.trace_file = os.path.join(log_dir, 'recommendation_traces.jsonl')
        
        # In-memory trace buffer
        self._traces: Dict[str, List[TraceEntry]] = {}
        self._active_traces: Dict[str, datetime] = {}
        
        if persist_to_file and enabled:
            os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"TraceLogger initialized (enabled={enabled}, persist={persist_to_file})")
    
    def start_trace(
        self,
        trace_id: str,
        request_data: Dict[str, Any]
    ) -> None:
        """
        Start a new trace.
        
        Args:
            trace_id: Unique trace identifier
            request_data: Initial request data
        """
        if not self.enabled:
            return
        
        self._traces[trace_id] = []
        self._active_traces[trace_id] = datetime.now()
        
        entry = TraceEntry(
            trace_id=trace_id,
            stage='request',
            data=request_data
        )
        self._traces[trace_id].append(entry)
        
        logger.debug(f"Started trace {trace_id}")
    
    def log_stage(
        self,
        trace_id: str,
        stage: str,
        data: Dict[str, Any],
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Log a processing stage within a trace.
        
        Args:
            trace_id: Trace identifier
            stage: Stage name
            data: Stage-specific data
            duration_ms: Optional duration in milliseconds
        """
        if not self.enabled:
            return
        
        if trace_id not in self._traces:
            logger.warning(f"Trace {trace_id} not found")
            return
        
        entry = TraceEntry(
            trace_id=trace_id,
            stage=stage,
            data=data
        )
        entry.duration_ms = duration_ms
        
        self._traces[trace_id].append(entry)
        logger.debug(f"Logged stage {stage} for trace {trace_id}")
    
    def end_trace(
        self,
        trace_id: str,
        response_data: Dict[str, Any]
    ) -> None:
        """
        End a trace and persist if configured.
        
        Args:
            trace_id: Trace identifier
            response_data: Final response data
        """
        if not self.enabled:
            return
        
        if trace_id not in self._traces:
            return
        
        # Calculate total duration
        start_time = self._active_traces.get(trace_id)
        total_duration = None
        if start_time:
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            del self._active_traces[trace_id]
        
        # Add response entry
        entry = TraceEntry(
            trace_id=trace_id,
            stage='response',
            data=response_data
        )
        entry.duration_ms = total_duration
        self._traces[trace_id].append(entry)
        
        # Persist to file
        if self.persist_to_file:
            self._persist_trace(trace_id)
        
        logger.debug(f"Ended trace {trace_id}")
    
    def _persist_trace(self, trace_id: str) -> None:
        """Persist a trace to file."""
        entries = self._traces.get(trace_id, [])
        if not entries:
            return
        
        try:
            with open(self.trace_file, 'a') as f:
                trace_data = {
                    'trace_id': trace_id,
                    'entries': [entry.to_dict() for entry in entries],
                    'completed_at': datetime.now().isoformat()
                }
                f.write(json.dumps(trace_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist trace: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trace by ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Trace data or None
        """
        if trace_id not in self._traces:
            return None
        
        entries = self._traces[trace_id]
        return {
            'trace_id': trace_id,
            'entries': [entry.to_dict() for entry in entries],
            'is_complete': trace_id not in self._active_traces
        }
    
    def get_traces_for_user(
        self, 
        user_id: int, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent traces for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of traces to return
            
        Returns:
            List of trace data
        """
        user_traces = []
        
        for trace_id, entries in self._traces.items():
            # Check if any entry contains the user_id
            for entry in entries:
                if entry.data.get('user_id') == user_id:
                    user_traces.append({
                        'trace_id': trace_id,
                        'entries': [e.to_dict() for e in entries[:5]],  # First 5 entries
                        'entry_count': len(entries)
                    })
                    break
            
            if len(user_traces) >= limit:
                break
        
        return user_traces
    
    def get_error_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get traces that contain errors.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of error trace data
        """
        error_traces = []
        
        for trace_id, entries in self._traces.items():
            has_error = any(
                entry.data.get('error') or entry.stage == 'error'
                for entry in entries
            )
            
            if has_error:
                error_traces.append({
                    'trace_id': trace_id,
                    'entries': [e.to_dict() for e in entries],
                    'error_summary': self._extract_error_summary(entries)
                })
            
            if len(error_traces) >= limit:
                break
        
        return error_traces
    
    def _extract_error_summary(self, entries: List[TraceEntry]) -> str:
        """Extract error summary from trace entries."""
        for entry in entries:
            if entry.data.get('error'):
                return str(entry.data['error'])
        return "Unknown error"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from traces.
        
        Returns:
            Performance metrics dictionary
        """
        durations = []
        stage_durations: Dict[str, List[float]] = {}
        
        for entries in self._traces.values():
            for entry in entries:
                if entry.duration_ms is not None:
                    if entry.stage == 'response':
                        durations.append(entry.duration_ms)
                    else:
                        if entry.stage not in stage_durations:
                            stage_durations[entry.stage] = []
                        stage_durations[entry.stage].append(entry.duration_ms)
        
        stats = {
            'total_traces': len(self._traces),
            'active_traces': len(self._active_traces)
        }
        
        if durations:
            stats['avg_total_duration_ms'] = sum(durations) / len(durations)
            stats['p95_duration_ms'] = sorted(durations)[int(len(durations) * 0.95)]
            stats['max_duration_ms'] = max(durations)
        
        stats['stage_durations'] = {}
        for stage, durs in stage_durations.items():
            stats['stage_durations'][stage] = {
                'avg': sum(durs) / len(durs),
                'count': len(durs)
            }
        
        return stats
    
    def clear_old_traces(self, max_age_hours: int = 24) -> int:
        """
        Clear old traces from memory.
        
        Args:
            max_age_hours: Maximum age of traces to keep
            
        Returns:
            Number of traces cleared
        """
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        traces_to_remove = []
        
        for trace_id, entries in self._traces.items():
            if entries and entries[0].timestamp.timestamp() < cutoff:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self._traces[trace_id]
        
        logger.info(f"Cleared {len(traces_to_remove)} old traces")
        return len(traces_to_remove)
