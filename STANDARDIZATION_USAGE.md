# Module 7 - Standardized Data Layer Usage Guide

## Overview

The Standardized Data Layer converts ANY structured relational dataset into canonical ML-ready recommendation datasets using the schema intelligence defined in `output/rec_config.json`.

## Quick Start

### 1. Generate Sample Data and Run Standardization

```bash
python cli.py standardize --config output/rec_config.json --sample
```

This will:
- Generate sample data based on your configuration mappings
- Transform it to canonical schema
- Validate the results
- Export parquet files to `output/standardized/`

### 2. Process Your Actual Data Files

```bash
python cli.py standardize \
  --config output/rec_config.json \
  --files users=data/customers.csv items=data/products.csv transactions=data/orders.csv
```

Format: `entity_type=file_path` (space-separated for multiple files)

### 3. Use Python API Directly

```python
from Standardized_Data_Layer.run import StandardizationPipeline

# Initialize pipeline
pipeline = StandardizationPipeline('output/rec_config.json')

# Option A: Load from files
results = pipeline.run_from_files({
    'users': 'data/customers.csv',
    'items': 'data/products.csv',
    'transactions': 'data/orders.csv'
})

# Option B: Provide DataFrames directly
import pandas as pd
source_data = {
    'users': pd.read_csv('data/customers.csv'),
    'items': pd.read_csv('data/products.csv')
}
results = pipeline.run(source_data)
```

## CLI Options

```bash
python cli.py standardize --help
```

Available options:
- `--config, -c`: Path to rec_config.json (default: output/rec_config.json)
- `--sample`: Generate and process sample data from configuration
- `--files`: Source data files (format: entity_type=file_path)
- `--no-validate`: Skip validation after standardization
- `--no-write`: Skip writing output files
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress non-error output

## Output Files

After successful standardization, you'll find:

### Standardized Datasets (`output/standardized/`)
- `users.parquet` - Canonical user/customer data
- `items.parquet` - Canonical item/product data
- `transactions.parquet` - Transaction/order data
- `interactions.parquet` - User interaction events
- `categories.parquet` - Category/hierarchy data

### Metadata & Lineage (`output/reports/`)
- `data_lineage.json` - Complete transformation lineage
- `schema_metadata.json` - Schema versioning info
- `transformation_registry.json` - Applied transformations

## Example Output

```
============================================================
STANDARDIZATION RESULTS
============================================================
Success: True
Entities processed: 5
Duration: 0.12 seconds
Total rows: 25
Errors: 0
Warnings: 0

Output files written to: /workspace/output/standardized
============================================================
```

## Supported Domains

The system dynamically adapts to:
- Ecommerce (customers, products, orders)
- Fintech (accounts, transactions, transfers)
- Healthcare (patients, treatments, appointments)
- EdTech (students, courses, enrollments)
- SaaS (users, subscriptions, usage_events)
- CRM (contacts, leads, opportunities)
- ERP (employees, departments, projects)
- And any other structured relational dataset

## Configuration-Driven

All transformations are driven by `output/rec_config.json`:
- Schema mappings (source → canonical columns)
- Entity relationships (foreign keys)
- Feature assignments
- Interaction signal weights
- Business semantics
- Validation rules

No hardcoded schemas - everything adapts dynamically!

## Next Steps

After standardization, you can:
1. Validate: `python cli.py validate --config output/rec_config.json`
2. Generate Features: `python cli.py generate-features --config output/rec_config.json`
3. Train Models: `python cli.py train-model --config output/rec_config.json`
4. Get Recommendations: `python cli.py recommend --user 1001`

