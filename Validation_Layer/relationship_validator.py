"""Relationship validator for checking referential integrity."""

import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
from shared.logger import get_logger


logger = get_logger(__name__)


class RelationshipValidator:
    """Validates referential integrity between related tables."""
    
    # Foreign key relationships between entities
    FOREIGN_KEY_RELATIONSHIPS = {
        'transactions': [
            {'column': 'user_id', 'references': ('users', 'user_id')},
            {'column': 'item_id', 'references': ('items', 'item_id')}
        ],
        'interactions': [
            {'column': 'user_id', 'references': ('users', 'user_id')},
            {'column': 'item_id', 'references': ('items', 'item_id')}
        ],
        'items': [
            {'column': 'category_id', 'references': ('categories', 'category_id')}
        ]
    }
    
    def __init__(self, relationships: Optional[Dict[str, List[Dict]]] = None):
        """Initialize relationship validator.
        
        Args:
            relationships: Custom foreign key relationships
        """
        self.relationships = relationships or self.FOREIGN_KEY_RELATIONSHIPS
    
    def validate(
        self,
        dataframes: Dict[str, pd.DataFrame],
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate referential integrity.
        
        Args:
            dataframes: Dictionary of all DataFrames
            entity_type: Optional specific entity to check
        
        Returns:
            Dictionary with issues list and passed flag
        """
        logger.info("Validating referential integrity")
        issues = []
        
        # Determine which entities to check
        entities_to_check = [entity_type] if entity_type else list(self.relationships.keys())
        
        for entity in entities_to_check:
            if entity not in dataframes:
                continue
            
            df = dataframes[entity]
            relationships = self.relationships.get(entity, [])
            
            for rel in relationships:
                fk_column = rel['column']
                ref_entity, ref_column = rel['references']
                
                # Skip if foreign key column doesn't exist
                if fk_column not in df.columns:
                    continue
                
                # Skip if referenced entity doesn't exist
                if ref_entity not in dataframes:
                    logger.warning(f"Referenced entity {ref_entity} not found for {entity}.{fk_column}")
                    continue
                
                ref_df = dataframes[ref_entity]
                if ref_column not in ref_df.columns:
                    logger.warning(f"Referenced column {ref_column} not found in {ref_entity}")
                    continue
                
                # Check for orphaned records
                issue = self._check_foreign_key(
                    df, fk_column,
                    ref_df, ref_column,
                    entity, ref_entity
                )
                if issue:
                    issues.append(issue)
        
        passed = len([i for i in issues if i['severity'] == 'critical']) == 0
        
        return {
            'issues': issues,
            'passed': passed,
            'orphaned_count': sum(i.get('orphaned_count', 0) for i in issues),
            'relationships_checked': len(entities_to_check)
        }
    
    def _check_foreign_key(
        self,
        child_df: pd.DataFrame,
        fk_column: str,
        parent_df: pd.DataFrame,
        pk_column: str,
        child_entity: str,
        parent_entity: str
    ) -> Optional[Dict[str, Any]]:
        """Check foreign key constraint.
        
        Args:
            child_df: Child DataFrame with foreign key
            fk_column: Foreign key column name
            parent_df: Parent DataFrame with primary key
            pk_column: Primary key column name
            child_entity: Child entity name
            parent_entity: Parent entity name
        
        Returns:
            Issue dict if violations found, None otherwise
        """
        # Get valid parent keys (excluding nulls)
        parent_keys = set(parent_df[pk_column].dropna().unique())
        
        # Find orphaned records (excluding null FK values)
        child_fk_values = child_df[fk_column].dropna()
        orphaned_mask = ~child_fk_values.isin(parent_keys)
        orphaned_records = child_df.loc[child_fk_values[orphaned_mask].index]
        
        orphaned_count = len(orphaned_records)
        
        if orphaned_count > 0:
            orphaned_percentage = (orphaned_count / len(child_df)) * 100
            
            # Get sample orphaned values
            sample_orphaned = orphaned_records[fk_column].head(10).tolist()
            
            severity = 'critical' if orphaned_percentage > 5 else 'warning'
            
            return {
                'validation_type': 'relationship',
                'constraint_type': 'foreign_key',
                'child_entity': child_entity,
                'child_column': fk_column,
                'parent_entity': parent_entity,
                'parent_column': pk_column,
                'issue': f"Found {orphaned_count} orphaned records in {child_entity}.{fk_column} referencing non-existent {parent_entity}.{pk_column} ({orphaned_percentage:.2f}%)",
                'severity': severity,
                'orphaned_count': int(orphaned_count),
                'orphaned_percentage': round(orphaned_percentage, 2),
                'sample_orphaned_values': sample_orphaned
            }
        
        return None
    
    def validate_bidirectional(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate bidirectional relationships.
        
        Checks that relationships make sense in both directions.
        
        Args:
            dataframes: Dictionary of all DataFrames
        
        Returns:
            Validation results
        """
        logger.info("Validating bidirectional relationships")
        issues = []
        
        # Check transaction-user relationship
        if 'transactions' in dataframes and 'users' in dataframes:
            issues.extend(self._check_user_transactions(
                dataframes['transactions'],
                dataframes['users']
            ))
        
        # Check interaction-user relationship
        if 'interactions' in dataframes and 'users' in dataframes:
            issues.extend(self._check_user_interactions(
                dataframes['interactions'],
                dataframes['users']
            ))
        
        return {
            'issues': issues,
            'passed': len(issues) == 0
        }
    
    def _check_user_transactions(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Check user-transaction relationship consistency."""
        issues = []
        
        if 'user_id' not in transactions_df.columns or 'user_id' not in users_df.columns:
            return issues
        
        # Users without transactions (informational)
        users_with_tx = set(transactions_df['user_id'].dropna().unique())
        all_users = set(users_df['user_id'].dropna().unique())
        users_without_tx = all_users - users_with_tx
        
        if len(users_without_tx) > 0 and len(users_without_tx) == len(all_users):
            issues.append({
                'validation_type': 'relationship',
                'constraint_type': 'business_logic',
                'issue': f"All {len(users_without_tx)} users have no transactions - verify data completeness",
                'severity': 'warning',
                'affected_count': len(users_without_tx)
            })
        
        return issues
    
    def _check_user_interactions(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Check user-interaction relationship consistency."""
        issues = []
        
        if 'user_id' not in interactions_df.columns or 'user_id' not in users_df.columns:
            return issues
        
        # Users without interactions (informational)
        users_with_interaction = set(interactions_df['user_id'].dropna().unique())
        all_users = set(users_df['user_id'].dropna().unique())
        users_without_interaction = all_users - users_with_interaction
        
        if len(users_without_interaction) > 0 and len(users_without_interaction) == len(all_users):
            issues.append({
                'validation_type': 'relationship',
                'constraint_type': 'business_logic',
                'issue': f"All {len(users_without_interaction)} users have no interactions - verify data completeness",
                'severity': 'warning',
                'affected_count': len(users_without_interaction)
            })
        
        return issues
    
    def get_relationship_summary(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get summary of relationship metrics.
        
        Args:
            dataframes: Dictionary of all DataFrames
        
        Returns:
            Summary statistics about relationships
        """
        summary = {
            'entities': {},
            'relationships': []
        }
        
        # Summarize each entity's relationships
        for entity, relationships in self.relationships.items():
            if entity not in dataframes:
                continue
            
            df = dataframes[entity]
            entity_summary = {
                'total_records': len(df),
                'foreign_keys': []
            }
            
            for rel in relationships:
                fk_column = rel['column']
                ref_entity, ref_column = rel['references']
                
                if fk_column not in df.columns:
                    continue
                
                fk_unique = df[fk_column].nunique()
                fk_nulls = df[fk_column].isna().sum()
                
                entity_summary['foreign_keys'].append({
                    'column': fk_column,
                    'references': f"{ref_entity}.{ref_column}",
                    'unique_values': int(fk_unique),
                    'null_count': int(fk_nulls)
                })
            
            summary['entities'][entity] = entity_summary
        
        # Summarize relationships
        for entity, relationships in self.relationships.items():
            for rel in relationships:
                summary['relationships'].append({
                    'from': f"{entity}.{rel['column']}",
                    'to': f"{rel['references'][0]}.{rel['references'][1]}"
                })
        
        return summary
