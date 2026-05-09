"""Serving Monitor - Monitors recommendation serving performance."""
from typing import Dict, Any, List
from datetime import datetime
import logging
import json
import os
logger = logging.getLogger(__name__)

class ServingMonitor:
    """Monitors recommendation serving metrics."""
    def __init__(self, config: Dict[str, Any], log_dir: str = "output/logs"):
        self.config = config
        self.log_dir = log_dir
        self.metrics_log = os.path.join(log_dir, 'serving_metrics.jsonl')
        self.request_count = 0
        self.latencies = []
        self.error_count = 0
        
        os.makedirs(log_dir, exist_ok=True)
    
    def log_request(self, latency_ms: float, n_results: int, success: bool = True):
        """Log a request metric."""
        self.request_count += 1
        self.latencies.append(latency_ms)
        
        if not success:
            self.error_count += 1
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'latency_ms': latency_ms,
            'n_results': n_results,
            'success': success
        }
        
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serving statistics."""
        if not self.latencies:
            return {'request_count': 0}
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
            'avg_latency_ms': sum(self.latencies) / len(self.latencies),
            'p95_latency_ms': sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) > 20 else max(self.latencies),
            'p99_latency_ms': sorted(self.latencies)[int(len(self.latencies) * 0.99)] if len(self.latencies) > 100 else max(self.latencies)
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.request_count = 0
        self.latencies = []
        self.error_count = 0
