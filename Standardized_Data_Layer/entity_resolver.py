"""
Entity resolver for resolving entity identities across datasets.
Handles entity deduplication, identity matching, and resolution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


@dataclass
class EntityMatch:
    """Represents a matched entity identity."""
    entity_id: Any
    source_ids: List[Any]
    confidence: float
    match_type: str  # exact, fuzzy, probabilistic
    source_entity: str


class EntityResolver:
    """
    Resolves entity identities across datasets.
    
    Features:
    - Entity deduplication
    - Identity matching across sources
    - Confidence scoring
    - Multiple match strategies
    - Resolution reporting
    """
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize entity resolver.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
        self.resolved_entities: Dict[str, pd.DataFrame] = {}
        self.match_results: List[EntityMatch] = []
    
    def resolve_entity(
        self,
        df: pd.DataFrame,
        entity_type: str,
        id_column: Optional[str] = None,
        strategy: str = 'exact'
    ) -> pd.DataFrame:
        """
        Resolve entities in a DataFrame.
        
        Args:
            df: Input DataFrame
            entity_type: Entity type name
            id_column: Column to use for resolution (default: primary key)
            strategy: Resolution strategy ('exact', 'fuzzy', 'probabilistic')
        
        Returns:
            DataFrame with resolved entity IDs
        """
        logger.info(f"Resolving entities for {entity_type} using strategy: {strategy}")
        
        from Standardized_Data_Layer.canonical_schema import CanonicalSchema
        
        if id_column is None:
            try:
                id_column = CanonicalSchema.get_primary_key(entity_type)
            except KeyError:
                id_column = df.columns[0]  # Fallback to first column
        
        if id_column not in df.columns:
            logger.warning(f"ID column {id_column} not found, using first column")
            id_column = df.columns[0]
        
        if strategy == 'exact':
            resolved = self._resolve_exact(df, id_column)
        elif strategy == 'fuzzy':
            resolved = self._resolve_fuzzy(df, id_column, entity_type)
        else:
            resolved = self._resolve_exact(df, id_column)
        
        self.resolved_entities[entity_type] = resolved
        logger.info(f"Resolved {len(resolved)} unique entities for {entity_type}")
        return resolved
    
    def _resolve_exact(
        self, 
        df: pd.DataFrame, 
        id_column: str
    ) -> pd.DataFrame:
        """Resolve entities using exact ID matching."""
        # Group by ID and keep first occurrence
        resolved = df.drop_duplicates(subset=[id_column], keep='first')
        
        # Record matches
        grouped = df.groupby(id_column).size()
        for entity_id, count in grouped.items():
            if count > 1:
                self.match_results.append(EntityMatch(
                    entity_id=entity_id,
                    source_ids=[entity_id] * count,
                    confidence=1.0,
                    match_type='exact',
                    source_entity='deduplicated'
                ))
        
        return resolved
    
    def _resolve_fuzzy(
        self, 
        df: pd.DataFrame, 
        id_column: str,
        entity_type: str
    ) -> pd.DataFrame:
        """Resolve entities using fuzzy matching on attributes."""
        logger.debug("Applying fuzzy matching strategy")
        
        # For now, fall back to exact matching
        # In production, this would use string similarity algorithms
        return self._resolve_exact(df, id_column)
    
    def merge_entities(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        entity_type: str,
        id_column: Optional[str] = None,
        merge_strategy: str = 'primary_wins'
    ) -> pd.DataFrame:
        """
        Merge entities from two sources.
        
        Args:
            primary_df: Primary source DataFrame
            secondary_df: Secondary source DataFrame
            entity_type: Entity type name
            id_column: ID column for matching
            merge_strategy: How to handle conflicts ('primary_wins', 'secondary_wins', 'merge')
        
        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging entities for {entity_type} using strategy: {merge_strategy}")
        
        from Standardized_Data_Layer.canonical_schema import CanonicalSchema
        
        if id_column is None:
            try:
                id_column = CanonicalSchema.get_primary_key(entity_type)
            except KeyError:
                raise ValueError(f"Cannot determine ID column for {entity_type}")
        
        if id_column not in primary_df.columns or id_column not in secondary_df.columns:
            raise ValueError(f"ID column {id_column} not found in both DataFrames")
        
        # Combine DataFrames
        combined = pd.concat([primary_df, secondary_df], ignore_index=True)
        
        # Apply merge strategy
        if merge_strategy == 'primary_wins':
            # Keep first occurrence (from primary)
            merged = combined.drop_duplicates(subset=[id_column], keep='first')
        elif merge_strategy == 'secondary_wins':
            # Keep last occurrence (from secondary)
            merged = combined.drop_duplicates(subset=[id_column], keep='last')
        else:
            # Simple deduplication
            merged = combined.drop_duplicates(subset=[id_column], keep='first')
        
        logger.info(f"Merged {len(primary_df)} + {len(secondary_df)} -> {len(merged)} entities")
        return merged
    
    def find_duplicate_entities(
        self,
        df: pd.DataFrame,
        entity_type: str,
        match_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Find potential duplicate entities.
        
        Args:
            df: Input DataFrame
            entity_type: Entity type name
            match_columns: Columns to check for duplicates
        
        Returns:
            DataFrame of potential duplicates
        """
        logger.info(f"Finding duplicate entities for {entity_type}")
        
        from Standardized_Data_Layer.canonical_schema import CanonicalSchema
        
        if match_columns is None:
            # Default to non-ID columns
            try:
                pk = CanonicalSchema.get_primary_key(entity_type)
                match_columns = [col for col in df.columns if col != pk]
            except KeyError:
                match_columns = list(df.columns)
        
        # Find duplicates based on match columns
        duplicates_mask = df.duplicated(subset=match_columns, keep=False)
        duplicates = df[duplicates_mask].copy()
        
        logger.info(f"Found {len(duplicates)} potential duplicates")
        return duplicates
    
    def create_entity_mapping(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_id_col: str,
        target_id_col: str,
        match_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a mapping between source and target entity IDs.
        
        Args:
            source_df: Source entity DataFrame
            target_df: Target entity DataFrame
            source_id_col: Source ID column
            target_id_col: Target ID column
            match_columns: Columns to use for matching
        
        Returns:
            Mapping DataFrame with source_id and target_id
        """
        logger.info("Creating entity ID mapping")
        
        if match_columns is None:
            # Use all common columns except IDs
            common_cols = set(source_df.columns) & set(target_df.columns)
            match_columns = list(common_cols - {source_id_col, target_id_col})
        
        if not match_columns:
            logger.warning("No match columns available")
            return pd.DataFrame({'source_id': [], 'target_id': []})
        
        # Perform merge to find matches
        merged = pd.merge(
            source_df[[source_id_col] + match_columns],
            target_df[[target_id_col] + match_columns],
            on=match_columns,
            how='inner'
        )
        
        mapping = merged[[source_id_col, target_id_col]].drop_duplicates()
        mapping.columns = ['source_id', 'target_id']
        
        logger.info(f"Created mapping with {len(mapping)} pairs")
        return mapping
    
    def get_resolution_stats(self, entity_type: str) -> Dict[str, Any]:
        """
        Get statistics about entity resolution.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Statistics dictionary
        """
        resolved = self.resolved_entities.get(entity_type)
        
        if resolved is None:
            return {'status': 'not_resolved'}
        
        matches = [m for m in self.match_results if m.source_entity == entity_type or m.match_type == 'exact']
        
        return {
            'entity_type': entity_type,
            'resolved_count': len(resolved),
            'total_matches': len(matches),
            'duplicates_found': sum(1 for m in matches if len(m.source_ids) > 1),
            'status': 'resolved'
        }
    
    def get_all_resolved_entities(self) -> Dict[str, pd.DataFrame]:
        """Get all resolved entity DataFrames."""
        return self.resolved_entities
    
    def clear(self) -> None:
        """Clear all resolution results."""
        self.resolved_entities = {}
        self.match_results = []
