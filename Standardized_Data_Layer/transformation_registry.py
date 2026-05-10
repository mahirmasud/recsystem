"""
Transformation registry for managing reusable transformation rules.
Stores semantic conversion rules and maintains transformation lineage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from shared.logger import get_logger
from shared.constants import Constants


logger = get_logger(__name__)


@dataclass
class TransformationRule:
    """Definition of a reusable transformation rule."""
    rule_id: str
    name: str
    description: str
    source_pattern: str
    target_canonical: str
    transformation_type: str  # rename, normalize, compute, map_values
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = '1.0'


@dataclass
class TransformationRecord:
    """Record of a transformation applied to data."""
    record_id: str
    entity_type: str
    rule_id: str
    source_columns: List[str]
    target_columns: List[str]
    applied_at: str
    row_count: int
    success: bool
    error_message: Optional[str] = None


class TransformationRegistry:
    """
    Registry for reusable transformation rules and lineage.
    
    Features:
    - Register reusable transformations
    - Store semantic conversion rules
    - Maintain transformation lineage
    - Support schema evolution
    - Rule discovery and matching
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize transformation registry.
        
        Args:
            registry_path: Path to store registry files
        """
        self.registry_path = Path(registry_path) if registry_path else Constants.OUTPUT_DIR / 'transformation_registry'
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.rules: Dict[str, TransformationRule] = {}
        self.records: List[TransformationRecord] = []
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        rules_file = self.registry_path / 'rules.json'
        records_file = self.registry_path / 'records.json'
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    data = json.load(f)
                
                for rule_id, rule_data in data.items():
                    self.rules[rule_id] = TransformationRule(**rule_data)
                
                logger.info(f"Loaded {len(self.rules)} transformation rules")
            except Exception as e:
                logger.warning(f"Failed to load rules: {e}")
        
        if records_file.exists():
            try:
                with open(records_file, 'r') as f:
                    data = json.load(f)
                
                self.records = [TransformationRecord(**r) for r in data]
                logger.info(f"Loaded {len(self.records)} transformation records")
            except Exception as e:
                logger.warning(f"Failed to load records: {e}")
    
    def register_rule(
        self,
        name: str,
        description: str,
        source_pattern: str,
        target_canonical: str,
        transformation_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        rule_id: Optional[str] = None
    ) -> TransformationRule:
        """
        Register a new transformation rule.
        
        Args:
            name: Rule name
            description: Rule description
            source_pattern: Source column/pattern to match
            target_canonical: Target canonical form
            transformation_type: Type of transformation
            parameters: Additional parameters
            rule_id: Custom rule ID (auto-generated if None)
        
        Returns:
            Registered transformation rule
        """
        if rule_id is None:
            rule_id = f"rule_{name.lower().replace(' ', '_')}_{len(self.rules) + 1}"
        
        rule = TransformationRule(
            rule_id=rule_id,
            name=name,
            description=description,
            source_pattern=source_pattern,
            target_canonical=target_canonical,
            transformation_type=transformation_type,
            parameters=parameters or {}
        )
        
        self.rules[rule_id] = rule
        self._save_rules()
        
        logger.info(f"Registered transformation rule: {rule_id}")
        return rule
    
    def get_rule(self, rule_id: str) -> Optional[TransformationRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)
    
    def get_rules_by_type(self, transformation_type: str) -> List[TransformationRule]:
        """Get all rules of a specific type."""
        return [
            rule for rule in self.rules.values()
            if rule.transformation_type == transformation_type
        ]
    
    def find_matching_rules(self, source_column: str) -> List[TransformationRule]:
        """
        Find rules that match a source column pattern.
        
        Args:
            source_column: Source column name
        
        Returns:
            List of matching rules
        """
        matching = []
        source_lower = source_column.lower()
        
        for rule in self.rules.values():
            pattern_lower = rule.source_pattern.lower()
            
            # Exact match
            if pattern_lower == source_lower:
                matching.append(rule)
            # Pattern contains
            elif pattern_lower in source_lower or source_lower in pattern_lower:
                matching.append(rule)
        
        return matching
    
    def record_transformation(
        self,
        entity_type: str,
        rule_id: str,
        source_columns: List[str],
        target_columns: List[str],
        row_count: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> TransformationRecord:
        """
        Record a transformation application.
        
        Args:
            entity_type: Entity type transformed
            rule_id: Rule that was applied
            source_columns: Source columns involved
            target_columns: Target columns produced
            row_count: Number of rows transformed
            success: Whether transformation succeeded
            error_message: Error message if failed
        
        Returns:
            Created transformation record
        """
        record_id = f"record_{len(self.records) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = TransformationRecord(
            record_id=record_id,
            entity_type=entity_type,
            rule_id=rule_id,
            source_columns=source_columns,
            target_columns=target_columns,
            applied_at=datetime.now().isoformat(),
            row_count=row_count,
            success=success,
            error_message=error_message
        )
        
        self.records.append(record)
        
        # Keep only last 10000 records
        if len(self.records) > 10000:
            self.records = self.records[-10000:]
        
        self._save_records()
        logger.debug(f"Recorded transformation: {record_id}")
        return record
    
    def get_transformation_history(
        self,
        entity_type: Optional[str] = None,
        rule_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TransformationRecord]:
        """
        Get transformation history with optional filters.
        
        Args:
            entity_type: Filter by entity type
            rule_id: Filter by rule ID
            limit: Maximum records to return
        
        Returns:
            List of transformation records
        """
        filtered = self.records
        
        if entity_type:
            filtered = [r for r in filtered if r.entity_type == entity_type]
        
        if rule_id:
            filtered = [r for r in filtered if r.rule_id == rule_id]
        
        return filtered[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_rules = len(self.rules)
        total_records = len(self.records)
        successful = sum(1 for r in self.records if r.success)
        failed = total_records - successful
        
        rules_by_type = {}
        for rule in self.rules.values():
            t = rule.transformation_type
            rules_by_type[t] = rules_by_type.get(t, 0) + 1
        
        return {
            'total_rules': total_rules,
            'total_records': total_records,
            'successful_transformations': successful,
            'failed_transformations': failed,
            'rules_by_type': rules_by_type,
            'success_rate': successful / total_records if total_records > 0 else 1.0
        }
    
    def export_registry(self, output_path: Optional[str] = None) -> Path:
        """
        Export complete registry to file.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path) if output_path else self.registry_path / 'transformation_registry.json'
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'rules': {
                rule_id: rule.__dict__ for rule_id, rule in self.rules.items()
            },
            'recent_records': [
                record.__dict__ for record in self.records[-100:]
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported transformation registry to {output_path}")
        return output_path
    
    def _save_rules(self) -> None:
        """Save rules to disk."""
        rules_file = self.registry_path / 'rules.json'
        
        data = {rule_id: rule.__dict__ for rule_id, rule in self.rules.items()}
        
        with open(rules_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {len(self.rules)} rules")
    
    def _save_records(self) -> None:
        """Save records to disk."""
        records_file = self.registry_path / 'records.json'
        
        data = [record.__dict__ for record in self.records]
        
        with open(records_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {len(self.records)} records")
    
    def clear_rules(self) -> None:
        """Clear all rules."""
        self.rules = {}
        self._save_rules()
    
    def clear_records(self) -> None:
        """Clear all records."""
        self.records = []
        self._save_records()
