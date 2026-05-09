"""
Canonical schema definitions for the recommendation system.
Defines the standard column names and types for each entity type.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ColumnDefinition:
    """Definition of a canonical column."""
    name: str
    dtype: str
    required: bool = True
    description: str = ""
    default_value: Any = None


@dataclass
class EntitySchema:
    """Schema definition for an entity type."""
    name: str
    columns: Dict[str, ColumnDefinition]
    primary_key: str
    description: str = ""


class CanonicalSchema:
    """
    Manages canonical schema definitions for all entity types.
    
    Defines the standard structure for:
    - users
    - items
    - transactions
    - interactions
    - categories
    """
    
    # User entity schema
    USERS_SCHEMA = EntitySchema(
        name="users",
        primary_key="user_id",
        description="User/customer master data",
        columns={
            "user_id": ColumnDefinition("user_id", "int64", True, "Unique user identifier"),
            "email": ColumnDefinition("email", "string", False, "User email address"),
            "registration_date": ColumnDefinition("registration_date", "datetime64[ns]", False, "Account creation date"),
            "country": ColumnDefinition("country", "string", False, "Country code"),
            "age": ColumnDefinition("age", "int64", False, "User age"),
            "gender": ColumnDefinition("gender", "string", False, "User gender"),
        }
    )
    
    # Item entity schema
    ITEMS_SCHEMA = EntitySchema(
        name="items",
        primary_key="item_id",
        description="Product/item master data",
        columns={
            "item_id": ColumnDefinition("item_id", "int64", True, "Unique item identifier"),
            "item_name": ColumnDefinition("item_name", "string", False, "Item/product name"),
            "category_id": ColumnDefinition("category_id", "int64", False, "Category identifier"),
            "price": ColumnDefinition("price", "float64", False, "Current price"),
            "cost": ColumnDefinition("cost", "float64", False, "Cost basis"),
            "brand": ColumnDefinition("brand", "string", False, "Brand name"),
            "stock_quantity": ColumnDefinition("stock_quantity", "int64", False, "Available inventory"),
        }
    )
    
    # Transaction entity schema
    TRANSACTIONS_SCHEMA = EntitySchema(
        name="transactions",
        primary_key="transaction_id",
        description="Purchase transaction records",
        columns={
            "transaction_id": ColumnDefinition("transaction_id", "int64", True, "Unique transaction identifier"),
            "user_id": ColumnDefinition("user_id", "int64", True, "User who made purchase"),
            "item_id": ColumnDefinition("item_id", "int64", True, "Item purchased"),
            "timestamp": ColumnDefinition("timestamp", "datetime64[ns]", True, "Transaction timestamp"),
            "quantity": ColumnDefinition("quantity", "int64", True, "Quantity purchased"),
            "unit_price": ColumnDefinition("unit_price", "float64", True, "Price per unit"),
            "discount": ColumnDefinition("discount", "float64", False, "Discount amount", 0.0),
            "total_amount": ColumnDefinition("total_amount", "float64", True, "Total transaction amount"),
        }
    )
    
    # Interaction entity schema
    INTERACTIONS_SCHEMA = EntitySchema(
        name="interactions",
        primary_key="interaction_id",
        description="User-item interaction events",
        columns={
            "interaction_id": ColumnDefinition("interaction_id", "int64", True, "Unique interaction identifier"),
            "user_id": ColumnDefinition("user_id", "int64", True, "User who interacted"),
            "item_id": ColumnDefinition("item_id", "int64", True, "Item interacted with"),
            "timestamp": ColumnDefinition("timestamp", "datetime64[ns]", True, "Interaction timestamp"),
            "interaction_type": ColumnDefinition("interaction_type", "string", True, "Type of interaction"),
            "session_id": ColumnDefinition("session_id", "string", False, "Session identifier"),
            "device_type": ColumnDefinition("device_type", "string", False, "Device used"),
        }
    )
    
    # Category entity schema
    CATEGORIES_SCHEMA = EntitySchema(
        name="categories",
        primary_key="category_id",
        description="Product category hierarchy",
        columns={
            "category_id": ColumnDefinition("category_id", "int64", True, "Unique category identifier"),
            "category_name": ColumnDefinition("category_name", "string", True, "Category name"),
            "parent_category_id": ColumnDefinition("parent_category_id", "int64", False, "Parent category", None),
            "level": ColumnDefinition("level", "int64", False, "Hierarchy level", 1),
        }
    )
    
    # All schemas registry
    SCHEMAS: Dict[str, EntitySchema] = {
        "users": USERS_SCHEMA,
        "items": ITEMS_SCHEMA,
        "transactions": TRANSACTIONS_SCHEMA,
        "interactions": INTERACTIONS_SCHEMA,
        "categories": CATEGORIES_SCHEMA,
    }
    
    @classmethod
    def get_schema(cls, entity_type: str) -> EntitySchema:
        """
        Get schema for an entity type.
        
        Args:
            entity_type: Name of entity type
        
        Returns:
            EntitySchema for the entity type
        
        Raises:
            KeyError: If entity type not found
        """
        if entity_type not in cls.SCHEMAS:
            raise KeyError(f"Unknown entity type: {entity_type}")
        return cls.SCHEMAS[entity_type]
    
    @classmethod
    def get_columns(cls, entity_type: str) -> List[str]:
        """Get list of canonical column names for entity type."""
        schema = cls.get_schema(entity_type)
        return list(schema.columns.keys())
    
    @classmethod
    def get_dtype(cls, entity_type: str, column: str) -> str:
        """Get expected data type for a column."""
        schema = cls.get_schema(entity_type)
        if column not in schema.columns:
            raise KeyError(f"Unknown column {column} for entity {entity_type}")
        return schema.columns[column].dtype
    
    @classmethod
    def get_all_dtypes(cls, entity_type: str) -> Dict[str, str]:
        """Get all column dtypes for entity type."""
        schema = cls.get_schema(entity_type)
        return {col: col_def.dtype for col, col_def in schema.columns.items()}
    
    @classmethod
    def get_required_columns(cls, entity_type: str) -> List[str]:
        """Get list of required columns for entity type."""
        schema = cls.get_schema(entity_type)
        return [col for col, col_def in schema.columns.items() if col_def.required]
    
    @classmethod
    def get_primary_key(cls, entity_type: str) -> str:
        """Get primary key column for entity type."""
        schema = cls.get_schema(entity_type)
        return schema.primary_key
    
    @classmethod
    def validate_entity_type(cls, entity_type: str) -> bool:
        """Check if entity type is valid."""
        return entity_type in cls.SCHEMAS
    
    @classmethod
    def get_all_entity_types(cls) -> List[str]:
        """Get list of all entity types."""
        return list(cls.SCHEMAS.keys())
