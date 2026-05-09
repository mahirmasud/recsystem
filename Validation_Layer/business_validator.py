"""Business validator for domain-specific rules."""

import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from shared.logger import get_logger
from shared.config import Config


logger = get_logger(__name__)


class BusinessValidator:
    """Validates business-specific rules and constraints."""
    
    # Default business rules per entity type
    DEFAULT_BUSINESS_RULES = {
        'users': [
            {
                'name': 'valid_email_format',
                'column': 'email',
                'rule': lambda x: x.str.contains('@', na=False),
                'message': 'Invalid email format',
                'severity': 'warning'
            },
            {
                'name': 'reasonable_age',
                'column': 'age',
                'rule': lambda x: (x >= 13) & (x <= 120),
                'message': 'Age outside reasonable range (13-120)',
                'severity': 'warning'
            }
        ],
        'items': [
            {
                'name': 'positive_price',
                'column': 'price',
                'rule': lambda x: x >= 0,
                'message': 'Price must be non-negative',
                'severity': 'critical'
            },
            {
                'name': 'valid_stock',
                'column': 'stock_quantity',
                'rule': lambda x: x >= 0,
                'message': 'Stock quantity must be non-negative',
                'severity': 'critical'
            }
        ],
        'transactions': [
            {
                'name': 'positive_amount',
                'column': 'amount',
                'rule': lambda x: x > 0,
                'message': 'Transaction amount must be positive',
                'severity': 'critical'
            },
            {
                'name': 'valid_quantity',
                'column': 'quantity',
                'rule': lambda x: x > 0,
                'message': 'Quantity must be positive',
                'severity': 'critical'
            },
            {
                'name': 'discount_less_than_amount',
                'columns': ['discount', 'amount'],
                'rule': lambda df: df['discount'] <= df['amount'],
                'message': 'Discount cannot exceed transaction amount',
                'severity': 'critical'
            }
        ],
        'interactions': [
            {
                'name': 'valid_interaction_type',
                'column': 'interaction_type',
                'rule': lambda x: x.isin(['view', 'click', 'purchase', 'cart_add', 'wishlist_add']),
                'message': 'Unknown interaction type',
                'severity': 'warning'
            }
        ]
    }
    
    def __init__(self, custom_rules: Optional[Dict[str, List[Dict]]] = None):
        """Initialize business validator.
        
        Args:
            custom_rules: Custom business rules to add/override defaults
        """
        self.rules = {**self.DEFAULT_BUSINESS_RULES}
        if custom_rules:
            for entity, entity_rules in custom_rules.items():
                if entity not in self.rules:
                    self.rules[entity] = []
                self.rules[entity].extend(entity_rules)
    
    def validate(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> Dict[str, Any]:
        """Validate business rules for an entity.
        
        Args:
            df: DataFrame to validate
            entity_type: Type of entity
        
        Returns:
            Dictionary with issues list and passed flag
        """
        logger.info(f"Validating business rules for {entity_type}")
        issues = []
        
        entity_rules = self.rules.get(entity_type, [])
        
        for rule in entity_rules:
            issue = self._apply_rule(df, rule, entity_type)
            if issue:
                issues.append(issue)
        
        passed = len([i for i in issues if i['severity'] == 'critical']) == 0
        
        return {
            'issues': issues,
            'passed': passed,
            'rules_checked': len(entity_rules),
            'rules_failed': len(issues)
        }
    
    def _apply_rule(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
        entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Apply a single business rule.
        
        Args:
            df: DataFrame to validate
            rule: Rule definition
            entity_type: Entity type
        
        Returns:
            Issue dict if violations found, None otherwise
        """
        rule_name = rule.get('name', 'unknown')
        severity = rule.get('severity', 'warning')
        message = rule.get('message', f'Rule {rule_name} violated')
        
        try:
            # Single column rule
            if 'column' in rule and 'rule' in rule:
                column = rule['column']
                if column not in df.columns:
                    return None
                
                rule_func = rule['rule']
                mask = rule_func(df[column])
                
                # Invert mask to find violations
                violations = ~mask
                violation_count = violations.sum()
                
                if violation_count > 0:
                    violation_percentage = (violation_count / len(df)) * 100
                    
                    return {
                        'validation_type': 'business',
                        'rule_name': rule_name,
                        'column': column,
                        'issue': f"{message} - {violation_count} violations ({violation_percentage:.2f}%)",
                        'severity': severity,
                        'violation_count': int(violation_count),
                        'violation_percentage': round(violation_percentage, 2)
                    }
            
            # Multi-column rule
            elif 'columns' in rule and 'rule' in rule:
                columns = rule['columns']
                if not all(col in df.columns for col in columns):
                    return None
                
                rule_func = rule['rule']
                mask = rule_func(df)
                
                violations = ~mask
                violation_count = violations.sum()
                
                if violation_count > 0:
                    violation_percentage = (violation_count / len(df)) * 100
                    
                    return {
                        'validation_type': 'business',
                        'rule_name': rule_name,
                        'columns': columns,
                        'issue': f"{message} - {violation_count} violations ({violation_percentage:.2f}%)",
                        'severity': severity,
                        'violation_count': int(violation_count),
                        'violation_percentage': round(violation_percentage, 2)
                    }
        
        except Exception as e:
            logger.error(f"Error applying rule {rule_name}: {e}")
            return {
                'validation_type': 'business',
                'rule_name': rule_name,
                'issue': f"Error evaluating rule: {str(e)}",
                'severity': 'warning',
                'error': str(e)
            }
        
        return None
    
    def add_rule(
        self,
        entity_type: str,
        name: str,
        column: str,
        rule_func: Callable,
        message: str,
        severity: str = 'warning'
    ):
        """Add a custom business rule.
        
        Args:
            entity_type: Entity type to apply rule to
            name: Rule name
            column: Column to validate
            rule_func: Function that returns boolean Series (True = valid)
            message: Error message for violations
            severity: Severity level (warning or critical)
        """
        if entity_type not in self.rules:
            self.rules[entity_type] = []
        
        self.rules[entity_type].append({
            'name': name,
            'column': column,
            'rule': rule_func,
            'message': message,
            'severity': severity
        })
        
        logger.info(f"Added business rule '{name}' for {entity_type}")
    
    def validate_cross_entity(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate cross-entity business rules.
        
        Args:
            dataframes: Dictionary of all DataFrames
        
        Returns:
            Validation results
        """
        logger.info("Validating cross-entity business rules")
        issues = []
        
        # Rule: Transaction amount should match item price * quantity
        if all(k in dataframes for k in ['transactions', 'items']):
            issues.extend(self._check_transaction_pricing(
                dataframes['transactions'],
                dataframes['items']
            ))
        
        # Rule: Items should have at least one interaction or transaction
        if all(k in dataframes for k in ['items', 'transactions', 'interactions']):
            issues.extend(self._check_item_activity(
                dataframes['items'],
                dataframes['transactions'],
                dataframes['interactions']
            ))
        
        return {
            'issues': issues,
            'passed': len([i for i in issues if i['severity'] == 'critical']) == 0
        }
    
    def _check_transaction_pricing(
        self,
        transactions_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Check that transaction amounts align with item prices."""
        issues = []
        
        if 'item_id' not in transactions_df.columns or 'price' not in items_df.columns:
            return issues
        
        if 'quantity' not in transactions_df.columns or 'amount' not in transactions_df.columns:
            return issues
        
        # Merge to get item prices
        merged = transactions_df.merge(
            items_df[['item_id', 'price']],
            on='item_id',
            how='left',
            suffixes=('', '_item')
        )
        
        # Calculate expected amount
        merged['expected_amount'] = merged['price'] * merged['quantity']
        
        # Check for significant discrepancies (>10% difference)
        merged['diff_ratio'] = abs(merged['amount'] - merged['expected_amount']) / merged['expected_amount'].replace(0, 1)
        discrepancies = merged[merged['diff_ratio'] > 0.1]
        
        if len(discrepancies) > 0:
            discrepancy_pct = (len(discrepancies) / len(transactions_df)) * 100
            
            issues.append({
                'validation_type': 'business',
                'rule_name': 'transaction_pricing_consistency',
                'issue': f"{len(discrepancies)} transactions have amounts differing >10% from expected price*quantity ({discrepancy_pct:.2f}%)",
                'severity': 'warning',
                'violation_count': len(discrepancies),
                'violation_percentage': round(discrepancy_pct, 2)
            })
        
        return issues
    
    def _check_item_activity(
        self,
        items_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Check for items with no activity."""
        issues = []
        
        if 'item_id' not in items_df.columns:
            return issues
        
        active_items = set()
        
        if 'item_id' in transactions_df.columns:
            active_items.update(transactions_df['item_id'].dropna().unique())
        
        if 'item_id' in interactions_df.columns:
            active_items.update(interactions_df['item_id'].dropna().unique())
        
        all_items = set(items_df['item_id'].dropna().unique())
        inactive_items = all_items - active_items
        
        if len(inactive_items) > 0:
            inactive_pct = (len(inactive_items) / len(all_items)) * 100 if all_items else 0
            
            severity = 'warning' if inactive_pct < 50 else 'critical'
            
            issues.append({
                'validation_type': 'business',
                'rule_name': 'item_activity_check',
                'issue': f"{len(inactive_items)} items have no transactions or interactions ({inactive_pct:.2f}%)",
                'severity': severity,
                'violation_count': len(inactive_items),
                'violation_percentage': round(inactive_pct, 2)
            })
        
        return issues
