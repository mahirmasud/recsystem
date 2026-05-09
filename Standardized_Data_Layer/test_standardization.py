"""
Test suite for the Standardized Data Layer.
"""

import pandas as pd
import pytest
from Standardized_Data_Layer.canonical_schema import CanonicalSchema
from Standardized_Data_Layer.config_reader import ConfigReader
from Standardized_Data_Layer.mapping_applier import MappingApplier
from Standardized_Data_Layer.dataframe_standardizer import DataFrameStandardizer
from Standardized_Data_Layer.null_handler import NullHandler
from Standardized_Data_Layer.schema_validator import SchemaValidator


class TestCanonicalSchema:
    """Tests for canonical schema definitions."""
    
    def test_get_schema(self):
        """Test getting schema for entity type."""
        schema = CanonicalSchema.get_schema('users')
        assert schema.name == 'users'
        assert schema.primary_key == 'user_id'
    
    def test_get_columns(self):
        """Test getting columns for entity type."""
        columns = CanonicalSchema.get_columns('items')
        assert 'item_id' in columns
        assert 'price' in columns
    
    def test_validate_entity_type(self):
        """Test entity type validation."""
        assert CanonicalSchema.validate_entity_type('users') is True
        assert CanonicalSchema.validate_entity_type('invalid') is False


class TestMappingApplier:
    """Tests for mapping applier."""
    
    def test_apply_mapping(self):
        """Test applying column mappings."""
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email_address': ['a@test.com', 'b@test.com', 'c@test.com']
        })
        
        applier = MappingApplier()
        result = applier.apply_mapping(df, 'users')
        
        assert 'user_id' in result.columns
        assert 'email' in result.columns
    
    def test_convert_dtypes(self):
        """Test dtype conversion."""
        df = pd.DataFrame({
            'user_id': ['1', '2', '3'],
            'age': ['25', '30', '35']
        })
        
        applier = MappingApplier()
        result = applier.convert_dtypes(df, 'users')
        
        assert result['user_id'].dtype in ['int64', 'int32']


class TestNullHandler:
    """Tests for null handler."""
    
    def test_handle_nulls_default(self):
        """Test handling nulls with default strategy."""
        df = pd.DataFrame({
            'user_id': [1, 2, None],
            'country': ['US', None, 'UK']
        })
        
        handler = NullHandler(strategy='default')
        result = handler.handle_all(df, 'users', check_required=False)
        
        assert result['user_id'].isnull().sum() == 0
        assert result['country'].isnull().sum() == 0
    
    def test_null_report(self):
        """Test null report generation."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [None, None, None]
        })
        
        handler = NullHandler()
        report = handler.get_null_report(df)
        
        assert report['total_rows'] == 3
        assert 'col1' in report['columns_with_nulls']
        assert 'col2' in report['columns_with_nulls']


class TestSchemaValidator:
    """Tests for schema validator."""
    
    def test_validate_success(self):
        """Test successful validation."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'email': ['a@test.com', 'b@test.com', 'c@test.com']
        })
        
        validator = SchemaValidator()
        is_valid, report = validator.validate(df, 'users')
        
        assert is_valid is True
        assert report['error_count'] == 0
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'email': ['a@test.com', 'b@test.com']
        })
        
        validator = SchemaValidator()
        is_valid, report = validator.validate(df, 'users')
        
        # user_id is required but missing
        assert is_valid is False or len(report['errors']) > 0


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()
