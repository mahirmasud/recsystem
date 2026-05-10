"""
Relationship builder for dynamic entity relationship standardization.
Detects and builds relational mappings between entities.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


@dataclass
class Relationship:
    """Definition of an entity relationship."""
    source_entity: str
    source_column: str
    target_entity: str
    target_column: str
    relationship_type: str  # one-to-many, many-to-many, etc.
    description: str = ""
    is_foreign_key: bool = True


@dataclass
class RelationshipGraph:
    """Graph of entity relationships."""
    relationships: List[Relationship] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        self.relationships.append(relationship)
        self.entities.add(relationship.source_entity)
        self.entities.add(relationship.target_entity)


class RelationshipBuilder:
    """
    Builds and manages entity relationships.
    
    Features:
    - Detect foreign key relationships from config
    - Build relationship graphs
    - Validate referential integrity
    - Support various relationship types
    - Generate join paths
    """
    
    # Relationship type constants
    ONE_TO_MANY = "one-to-many"
    MANY_TO_ONE = "many-to-one"
    ONE_TO_ONE = "one-to-one"
    MANY_TO_MANY = "many-to-many"
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize relationship builder.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
        self.relationships: List[Relationship] = []
        self.relationship_graph = RelationshipGraph()
    
    def build_from_config(self) -> List[Relationship]:
        """
        Build relationships from configuration foreign keys.
        
        Returns:
            List of detected relationships
        """
        logger.info("Building relationships from configuration")
        
        foreign_keys = self.config_reader.get_foreign_keys()
        
        for fk_reference, pk_reference in foreign_keys.items():
            try:
                relationship = self._parse_fk_reference(fk_reference, pk_reference)
                if relationship:
                    self.relationships.append(relationship)
                    self.relationship_graph.add_relationship(relationship)
            except Exception as e:
                logger.warning(f"Failed to parse FK reference {fk_reference}: {e}")
        
        logger.info(f"Built {len(self.relationships)} relationships from config")
        return self.relationships
    
    def _parse_fk_reference(
        self, 
        fk_reference: str, 
        pk_reference: str
    ) -> Optional[Relationship]:
        """Parse a foreign key reference string into a Relationship object."""
        try:
            # Parse format: "transactions.user_id" -> (transactions, user_id)
            fk_parts = fk_reference.split('.')
            pk_parts = pk_reference.split('.')
            
            if len(fk_parts) != 2 or len(pk_parts) != 2:
                return None
            
            source_entity = fk_parts[0]
            source_column = fk_parts[1]
            target_entity = pk_parts[0]
            target_column = pk_parts[1]
            
            # Infer relationship type based on column names
            rel_type = self._infer_relationship_type(source_column, target_column)
            
            relationship = Relationship(
                source_entity=source_entity,
                source_column=source_column,
                target_entity=target_entity,
                target_column=target_column,
                relationship_type=rel_type,
                description=f"{source_entity}.{source_column} -> {target_entity}.{target_column}"
            )
            
            logger.debug(f"Parsed relationship: {source_entity} -> {target_entity}")
            return relationship
            
        except Exception as e:
            logger.warning(f"Error parsing FK reference: {e}")
            return None
    
    def _infer_relationship_type(
        self, 
        source_column: str, 
        target_column: str
    ) -> str:
        """Infer relationship type from column names."""
        # If target is a primary key, it's typically many-to-one from source perspective
        if 'id' in target_column.lower():
            return self.MANY_TO_ONE
        return self.ONE_TO_MANY
    
    def detect_relationships_from_data(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> List[Relationship]:
        """
        Detect relationships by analyzing DataFrame columns.
        
        Args:
            dataframes: Dictionary of entity DataFrames
        
        Returns:
            List of detected relationships
        """
        logger.info("Detecting relationships from data patterns")
        
        detected = []
        
        # Look for common FK patterns
        id_columns = {}
        for entity, df in dataframes.items():
            for col in df.columns:
                if col.endswith('_id') or 'id' in col.lower():
                    if entity not in id_columns:
                        id_columns[entity] = []
                    id_columns[entity].append(col)
        
        # Match potential FKs
        for source_entity, source_ids in id_columns.items():
            for source_col in source_ids:
                # Try to find matching PK in other entities
                base_name = source_col.replace('_id', '')
                
                for target_entity, target_ids in id_columns.items():
                    if target_entity == source_entity:
                        continue
                    
                    # Check for exact match or pattern match
                    pk_candidate = f"{base_name}_id"
                    if pk_candidate in target_ids:
                        rel = Relationship(
                            source_entity=source_entity,
                            source_column=source_col,
                            target_entity=target_entity,
                            target_column=pk_candidate,
                            relationship_type=self.MANY_TO_ONE,
                            description=f"Detected: {source_col} references {pk_candidate}"
                        )
                        detected.append(rel)
                        logger.debug(f"Detected relationship: {rel.description}")
        
        return detected
    
    def validate_referential_integrity(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationship: Optional[Relationship] = None
    ) -> Dict[str, Any]:
        """
        Validate referential integrity for relationships.
        
        Args:
            dataframes: Dictionary of entity DataFrames
            relationship: Specific relationship to validate (None for all)
        
        Returns:
            Validation report
        """
        logger.info("Validating referential integrity")
        
        relationships_to_check = [relationship] if relationship else self.relationships
        violations = []
        
        for rel in relationships_to_check:
            if not rel:
                continue
                
            source_df = dataframes.get(rel.source_entity)
            target_df = dataframes.get(rel.target_entity)
            
            if source_df is None or target_df is None:
                logger.warning(f"Missing dataframe for {rel.source_entity} or {rel.target_entity}")
                continue
            
            if rel.source_column not in source_df.columns:
                violations.append({
                    'relationship': str(rel),
                    'violation': f"Source column {rel.source_column} not found"
                })
                continue
            
            if rel.target_column not in target_df.columns:
                violations.append({
                    'relationship': str(rel),
                    'violation': f"Target column {rel.target_column} not found"
                })
                continue
            
            # Check for orphan records
            source_ids = set(source_df[rel.source_column].dropna().unique())
            target_ids = set(target_df[rel.target_column].dropna().unique())
            
            orphan_ids = source_ids - target_ids
            
            if orphan_ids:
                violations.append({
                    'relationship': f"{rel.source_entity}.{rel.source_column} -> {rel.target_entity}.{rel.target_column}",
                    'violation': 'orphan_records',
                    'orphan_count': len(orphan_ids),
                    'sample_orphans': list(orphan_ids)[:10]
                })
                logger.warning(f"Found {len(orphan_ids)} orphan records in {rel.source_entity}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'relationships_checked': len(relationships_to_check)
        }
    
    def get_join_path(
        self,
        source_entity: str,
        target_entity: str
    ) -> Optional[List[Relationship]]:
        """
        Find a join path between two entities.
        
        Args:
            source_entity: Starting entity
            target_entity: Target entity
        
        Returns:
            List of relationships forming the path, or None if no path exists
        """
        logger.debug(f"Finding join path from {source_entity} to {target_entity}")
        
        # Simple BFS to find path
        from collections import deque
        
        queue = deque([(source_entity, [])])
        visited = {source_entity}
        
        while queue:
            current_entity, path = queue.popleft()
            
            if current_entity == target_entity:
                return path
            
            # Find outgoing relationships
            for rel in self.relationships:
                next_entity = None
                
                if rel.source_entity == current_entity:
                    next_entity = rel.target_entity
                elif rel.target_entity == current_entity:
                    next_entity = rel.source_entity
                
                if next_entity and next_entity not in visited:
                    visited.add(next_entity)
                    new_rel = Relationship(
                        source_entity=rel.source_entity,
                        source_column=rel.source_column,
                        target_entity=rel.target_entity,
                        target_column=rel.target_column,
                        relationship_type=rel.relationship_type
                    )
                    queue.append((next_entity, path + [new_rel]))
        
        return None
    
    def get_relationship_graph(self) -> RelationshipGraph:
        """Get the relationship graph."""
        return self.relationship_graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationships to dictionary format."""
        return {
            'relationships': [
                {
                    'source_entity': rel.source_entity,
                    'source_column': rel.source_column,
                    'target_entity': rel.target_entity,
                    'target_column': rel.target_column,
                    'relationship_type': rel.relationship_type,
                    'description': rel.description
                }
                for rel in self.relationships
            ],
            'entities': list(self.relationship_graph.entities)
        }
    
    def clear(self) -> None:
        """Clear all relationships."""
        self.relationships = []
        self.relationship_graph = RelationshipGraph()
