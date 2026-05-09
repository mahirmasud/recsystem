"""Test suite for the Validation Layer."""

import pandas as pd
import pytest
from Validation_Layer.null_validator import NullValidator
from Validation_Layer.type_validator import TypeValidator
from Validation_Layer.range_validator import RangeValidator
from Validation_Layer.uniqueness_validator import UniquenessValidator


class TestNullValidator:
    """Tests for null validator."""
    
    def test_no_nulls(self):
        """Test DataFrame with no nulls."""
        df = pd.DataFrame({'user_id': [1, 2, 3], 'email': ['a', 'b', 'c']})
        validator = NullValidator()
        result = validator.validate(df, 'users')
        
        assert result['passed'] is True
        assert len(result['issues']) == 0
    
    def test_with_nulls(self):
        """Test DataFrame with nulls."""
        df = pd.DataFrame({'user_id': [1, None, 3], 'email': ['a', 'b', 'c']})
        validator = NullValidator()
        result = validator.validate(df, 'users')
        
        assert result['passed'] is False
        assert len(result['issues']) == 1


class TestRangeValidator:
    """Tests for range validator."""
    
    def test_valid_ranges(self):
        """Test values within valid ranges."""
        df = pd.DataFrame({'quantity': [1, 5, 10], 'price': [10.0, 20.0, 30.0]})
        validator = RangeValidator()
        result = validator.validate(df, 'transactions')
        
        assert result['passed'] is True
    
    def test_negative_values(self):
        """Test negative value detection."""
        df = pd.DataFrame({'quantity': [-1, 5, 10], 'price': [10.0, 20.0, 30.0]})
        validator = RangeValidator()
        result = validator.validate(df, 'transactions')
        
        assert result['passed'] is False


class TestUniquenessValidator:
    """Tests for uniqueness validator."""
    
    def test_unique_keys(self):
        """Test unique primary keys."""
        df = pd.DataFrame({'user_id': [1, 2, 3], 'email': ['a', 'b', 'c']})
        validator = UniquenessValidator()
        result = validator.validate(df, 'users')
        
        assert result['passed'] is True
    
    def test_duplicate_keys(self):
        """Test duplicate primary key detection."""
        df = pd.DataFrame({'user_id': [1, 1, 3], 'email': ['a', 'b', 'c']})
        validator = UniquenessValidator()
        result = validator.validate(df, 'users')
        
        assert result['passed'] is False
        assert len(result['issues']) > 0


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()
