"""
Role mapper for dynamically assigning canonical roles to entities.
Maps source entities to user/item/interaction/etc. roles based on configuration.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from shared.logger import get_logger
from Standardized_Data_Layer.config_reader import ConfigReader


logger = get_logger(__name__)


@dataclass
class RoleAssignment:
    """Definition of a role assignment."""
    source_entity: str
    canonical_role: str  # user, item, interaction, transaction, etc.
    confidence: float
    assignment_reason: str


class RoleMapper:
    """
    Maps source entities to canonical recommendation roles.
    
    Features:
    - Dynamic role detection from config
    - Entity role inference
    - Multi-role support
    - Domain-agnostic mapping
    """
    
    # Canonical role types
    ROLE_USER = 'user'
    ROLE_ITEM = 'item'
    ROLE_INTERACTION = 'interaction'
    ROLE_TRANSACTION = 'transaction'
    ROLE_CATEGORY = 'category'
    ROLE_SESSION = 'session'
    ROLE_EVENT = 'event'
    ROLE_ACCOUNT = 'account'
    ROLE_PRODUCT = 'product'
    ROLE_SUBSCRIPTION = 'subscription'
    ROLE_ACTIVITY = 'activity'
    ROLE_BEHAVIOR = 'behavior'
    
    # Keywords that suggest entity roles
    ROLE_KEYWORDS = {
        ROLE_USER: ['user', 'customer', 'client', 'member', 'account', 'consumer'],
        ROLE_ITEM: ['item', 'product', 'sku', 'article', 'listing', 'asset'],
        ROLE_INTERACTION: ['interaction', 'engagement', 'action', 'click', 'view'],
        ROLE_TRANSACTION: ['transaction', 'order', 'purchase', 'sale', 'payment'],
        ROLE_CATEGORY: ['category', 'department', 'class', 'group', 'segment'],
        ROLE_SESSION: ['session', 'visit', 'browse_session'],
        ROLE_EVENT: ['event', 'log', 'activity', 'occurrence'],
        ROLE_ACCOUNT: ['account', 'subscription', 'membership'],
        ROLE_PRODUCT: ['product', 'item', 'good', 'merchandise'],
        ROLE_SUBSCRIPTION: ['subscription', 'plan', 'tier', 'membership'],
        ROLE_ACTIVITY: ['activity', 'action', 'task', 'operation'],
        ROLE_BEHAVIOR: ['behavior', 'pattern', 'habit', 'tendency'],
    }
    
    def __init__(self, config_reader: Optional[ConfigReader] = None):
        """
        Initialize role mapper.
        
        Args:
            config_reader: ConfigReader instance
        """
        self.config_reader = config_reader or ConfigReader()
        self.role_assignments: Dict[str, RoleAssignment] = {}
    
    def assign_roles_from_config(self) -> Dict[str, RoleAssignment]:
        """
        Assign canonical roles based on configuration.
        
        Returns:
            Dictionary of entity type to role assignment
        """
        logger.info("Assigning roles from configuration")
        
        trusted_mappings = self.config_reader.get_trusted_mappings()
        
        for entity_type, columns in trusted_mappings.items():
            # Infer role from entity type name and column patterns
            role = self._infer_role_from_entity_type(entity_type, columns)
            
            if role:
                assignment = RoleAssignment(
                    source_entity=entity_type,
                    canonical_role=role,
                    confidence=0.9,
                    assignment_reason=f"Inferred from entity type '{entity_type}'"
                )
                self.role_assignments[entity_type] = assignment
                logger.debug(f"Assigned role '{role}' to entity '{entity_type}'")
        
        return self.role_assignments
    
    def _infer_role_from_entity_type(
        self,
        entity_type: str,
        columns: Dict[str, Any]
    ) -> Optional[str]:
        """Infer canonical role from entity type name and columns."""
        entity_lower = entity_type.lower()
        
        # Direct match
        for role, keywords in self.ROLE_KEYWORDS.items():
            if any(keyword in entity_lower for keyword in keywords):
                return role
        
        # Check column patterns
        column_names = [col.lower() for col in columns.keys()]
        
        # User indicators
        if any('user_id' in c or 'customer_id' in c for c in column_names):
            return self.ROLE_USER
        
        # Item indicators
        if any('item_id' in c or 'product_id' in c for c in column_names):
            return self.ROLE_ITEM
        
        # Transaction indicators
        if any('transaction_id' in c or 'order_id' in c for c in column_names):
            if any('amount' in c or 'price' in c for c in column_names):
                return self.ROLE_TRANSACTION
        
        # Interaction indicators
        if any('interaction_id' in c or 'event_id' in c for c in column_names):
            if any('type' in c or 'action' in c for c in column_names):
                return self.ROLE_INTERACTION
        
        # Category indicators
        if any('category_id' in c for c in column_names):
            return self.ROLE_CATEGORY
        
        return None
    
    def infer_role_from_dataframe(
        self,
        df: pd.DataFrame,
        entity_name: str
    ) -> Optional[str]:
        """
        Infer canonical role from DataFrame structure.
        
        Args:
            df: Input DataFrame
            entity_name: Name of the entity
        
        Returns:
            Inferred role or None
        """
        logger.debug(f"Inferring role for entity '{entity_name}' from DataFrame")
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Score each role based on column matches
        role_scores: Dict[str, int] = {}
        
        for role, keywords in self.ROLE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                for col in columns_lower:
                    if keyword in col:
                        score += 1
            
            if score > 0:
                role_scores[role] = score
        
        if not role_scores:
            return None
        
        # Return highest scoring role
        best_role = max(role_scores, key=role_scores.get)
        logger.debug(f"Inferred role '{best_role}' with score {role_scores[best_role]}")
        return best_role
    
    def get_entities_by_role(self, role: str) -> List[str]:
        """
        Get all entities assigned to a specific role.
        
        Args:
            role: Canonical role name
        
        Returns:
            List of entity types with that role
        """
        return [
            entity for entity, assignment in self.role_assignments.items()
            if assignment.canonical_role == role
        ]
    
    def get_role_for_entity(self, entity_type: str) -> Optional[str]:
        """
        Get the canonical role for an entity type.
        
        Args:
            entity_type: Entity type name
        
        Returns:
            Canonical role or None
        """
        assignment = self.role_assignments.get(entity_type)
        return assignment.canonical_role if assignment else None
    
    def detect_user_item_pair(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Optional[tuple[str, str]]:
        """
        Detect the primary user and item entities for recommendations.
        
        Args:
            dataframes: Dictionary of entity DataFrames
        
        Returns:
            Tuple of (user_entity, item_entity) or None
        """
        logger.info("Detecting user-item entity pair")
        
        user_candidates = []
        item_candidates = []
        
        for entity_type, df in dataframes.items():
            role = self.infer_role_from_dataframe(df, entity_type)
            
            if role == self.ROLE_USER:
                user_candidates.append(entity_type)
            elif role == self.ROLE_ITEM:
                item_candidates.append(entity_type)
        
        # Prefer entities named 'users' or 'items'
        user_entity = next((e for e in user_candidates if e == 'users'), 
                          user_candidates[0] if user_candidates else None)
        item_entity = next((e for e in item_candidates if e == 'items'),
                          item_candidates[0] if item_candidates else None)
        
        if user_entity and item_entity:
            logger.info(f"Detected user-item pair: {user_entity}, {item_entity}")
            return (user_entity, item_entity)
        
        return None
    
    def get_all_role_assignments(self) -> Dict[str, Dict[str, Any]]:
        """Get all role assignments as dictionary."""
        return {
            entity: {
                'canonical_role': assignment.canonical_role,
                'confidence': assignment.confidence,
                'reason': assignment.assignment_reason
            }
            for entity, assignment in self.role_assignments.items()
        }
    
    def clear_assignments(self) -> None:
        """Clear all role assignments."""
        self.role_assignments = {}
