# Intelligent Schema Mapping & Business Meaning System

A production-ready, modular recommendation system with intelligent schema mapping, business meaning understanding, and hybrid deep learning recommendations.

## Architecture Overview

This system implements **Modules 7-12** of an intelligent recommendation pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Orchestrator                          │
│                         (cli.py)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Module 7        │ │  Module 8        │ │  Module 9        │
│  Standardized    │ │  Validation      │ │  Feature         │
│  Data Layer      │ │  Layer           │ │  Engineering     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Module 10       │
                    │  Recommendation  │
                    │  Engine          │
                    │  - Candidate Gen │
                    │  - Ranking       │
                    │  - Re-Ranking    │
                    │  - Evaluation    │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Module 11       │ │  Module 12       │ │  Shared          │
│  Rule Engine     │ │  Serving         │ │  Utilities       │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Project Structure

```
├── Business_Meaning/          # Domain loading & semantic mapping (pre-built)
├── ecommerce/                 # Sample e-commerce data (pre-built)
├── Mapping_Engine/            # Schema mapping engine (pre-built)
├── output/
│   ├── rec_config.json        # Primary configuration source
│   ├── standardized/          # Module 7 output
│   ├── validated/             # Module 8 output
│   ├── features/              # Module 9 output
│   ├── models/                # Module 10 trained models
│   ├── recommendations/       # Module 10/12 outputs
│   ├── reports/               # Validation & evaluation reports
│   └── logs/                  # System logs
│
├── Standardized_Data_Layer/   # MODULE 7
│   ├── canonical_schema.py    # Canonical data model definitions
│   ├── config_reader.py       # Configuration loader
│   ├── mapping_applier.py     # Apply confirmed mappings
│   ├── dataframe_standardizer.py
│   ├── metric_builder.py      # Derived metrics (CLV, margins)
│   ├── lineage_manager.py     # Track data transformations
│   ├── null_handler.py        # Handle missing values
│   ├── schema_validator.py    # Schema validation
│   ├── dataset_writer.py      # Parquet export
│   ├── sync_manager.py        # Pipeline synchronization
│   ├── run.py                 # Pipeline entry point
│   └── test_standardization.py
│
├── Validation_Layer/          # MODULE 8
│   ├── validator.py           # Main orchestrator
│   ├── null_validator.py      # Null/missing value checks
│   ├── type_validator.py      # Data type validation
│   ├── range_validator.py     # Numeric range checks
│   ├── uniqueness_validator.py # Primary key & duplicate detection
│   ├── relationship_validator.py # Referential integrity
│   ├── business_validator.py  # Domain-specific rules
│   ├── severity_classifier.py # Issue severity scoring
│   ├── score_calculator.py    # Quality score calculation
│   ├── invalid_handler.py     # Invalid record handling
│   ├── report_generator.py    # Report generation
│   ├── run.py                 # Pipeline entry point
│   └── test_validation.py
│
├── Feature_Engineering/       # MODULE 9
│   ├── entity_builder.py      # Entity construction
│   ├── feature_generator.py   # Main feature orchestration
│   ├── aggregation_features.py # Aggregation-based features
│   ├── temporal_features.py   # Time-based features
│   ├── behavioral_features.py # User behavior patterns
│   ├── recency_features.py    # Recency signals
│   ├── frequency_features.py  # Frequency signals
│   ├── monetary_features.py   # Monetary value features
│   ├── category_affinity.py   # Category preferences
│   ├── customer_value.py      # CLV calculations
│   ├── feature_store.py       # Feature storage & retrieval
│   ├── feature_selector.py    # Feature selection
│   ├── feature_versioning.py  # Version tracking
│   ├── run.py                 # Pipeline entry point
│   └── test_features.py
│
├── Recommendation_Engine/     # MODULE 10
│   ├── Candidate_Generation/
│   │   ├── collaborative_filtering.py
│   │   ├── matrix_factorization.py
│   │   ├── three_tower_model.py  # Three-Tower architecture
│   │   ├── context_tower.py      # Context encoding
│   │   ├── ann_search.py         # Approximate nearest neighbor
│   │   ├── user_tower.py         # User embeddings
│   │   ├── item_tower.py         # Item embeddings
│   │   ├── popularity_engine.py  # Popularity fallback
│   │   ├── embedding_retrieval.py
│   │   └── candidate_manager.py
│   │
│   ├── Ranking/
│   │   ├── dlrm_ranker.py        # Deep Learning Recommendation Model
│   │   ├── xgboost_ranker.py     # XGBoost ranking
│   │   ├── lightgbm_ranker.py    # LightGBM ranking
│   │   ├── scoring_engine.py
│   │   ├── interaction_ranker.py
│   │   └── ranker.py
│   │
│   ├── ReRanking/
│   │   ├── diversity.py          # Diversity optimization
│   │   ├── freshness.py          # Freshness boosting
│   │   ├── margin_booster.py     # Margin-based boosting
│   │   ├── business_booster.py   # Business rule boosting
│   │   ├── cold_start.py         # Cold start handling
│   │   ├── exploration.py        # Exploration vs exploitation
│   │   └── reranker.py
│   │
│   ├── Evaluation/
│   │   ├── precision.py
│   │   ├── recall.py
│   │   ├── map_metric.py
│   │   ├── ndcg.py
│   │   ├── ctr.py
│   │   ├── diversity_score.py
│   │   ├── coverage.py
│   │   └── evaluator.py
│   │
│   ├── model_registry.py      # Model versioning & registry
│   ├── trainer.py             # Training pipelines
│   ├── inference.py           # Inference engine
│   ├── recommendation_manager.py
│   ├── run.py                 # CLI entry point
│   └── test_recommendation.py
│
├── Rule_Engine/               # MODULE 11
│   ├── rule_loader.py         # YAML rule loading
│   ├── rule_parser.py         # Rule parsing
│   ├── filter_rules.py        # Filtering rules
│   ├── boost_rules.py         # Score boosting rules
│   ├── conditional_rules.py   # Conditional logic
│   ├── context_rules.py       # Context-aware rules
│   ├── campaign_rules.py      # Campaign-specific rules
│   ├── rule_executor.py       # Rule execution engine
│   ├── score_adjuster.py      # Score adjustment logic
│   ├── explanation_logger.py  # Explainability logging
│   ├── chain_executor.py      # Rule chain execution
│   ├── run.py                 # CLI entry point
│   └── test_rules.py
│
├── Recommendation_Serving/    # MODULE 12
│   ├── request_parser.py      # Request parsing & validation
│   ├── recommendation_service.py # Core service orchestration
│   ├── session_handler.py     # Session state management
│   ├── cache_handler.py       # Caching layer
│   ├── recommendation_formatter.py # Output formatting
│   ├── explanation_generator.py # Human-readable explanations
│   ├── trace_logger.py        # Debug/audit tracing
│   ├── realtime_recommendation.py # Real-time serving
│   ├── batch_recommendation.py # Batch processing
│   ├── export_manager.py      # Export management
│   ├── run.py                 # CLI entry point
│   └── test_serving.py
│
├── shared/                    # Shared utilities
│   ├── config.py              # Configuration management
│   ├── constants.py           # System constants
│   ├── logger.py              # Structured logging
│   ├── file_loader.py         # File I/O utilities
│   ├── dataframe_utils.py     # DataFrame helpers
│   ├── yaml_loader.py         # YAML utilities
│   ├── metrics.py             # Metric calculations
│   └── exceptions.py          # Custom exceptions
│
├── cli.py                     # GLOBAL CLI ORCHESTRATOR
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py status
```

## Quick Start

### 1. Standardize Data (Module 7)

Convert mapped client data into canonical format:

```bash
python cli.py standardize --config output/rec_config.json
```

This creates:
- `output/standardized/users.parquet`
- `output/standardized/items.parquet`
- `output/standardized/transactions.parquet`
- `output/standardized/interactions.parquet`
- `output/standardized/categories.parquet`

### 2. Validate Data (Module 8)

Validate standardized datasets:

```bash
python cli.py validate --config output/rec_config.json
```

Outputs:
- `output/validated/` - Cleaned datasets
- `output/reports/validation_report.json` - Quality report

### 3. Generate Features (Module 9)

Generate ML-ready features:

```bash
python cli.py generate-features \
  --transactions output/standardized/transactions.parquet \
  --interactions output/standardized/interactions.parquet \
  --output output/features
```

Creates:
- `output/features/user_features.parquet`
- `output/features/item_features.parquet`
- `output/features/interaction_features.parquet`

### 4. Train Models (Module 10)

Train recommendation models:

```bash
python cli.py train-model \
  --model-types xgboost,lightgbm \
  --epochs 10 \
  --batch-size 256 \
  --learning-rate 0.001
```

Saves models to `output/models/`.

### 5. Generate Recommendations (Module 10 + 12)

Get personalized recommendations for a user:

```bash
python cli.py recommend --user 1001 --top-k 10 --format json
```

Or batch recommendations:

```bash
python cli.py batch-recommend users.csv --top-k 10 --output recommendations.json
```

### 6. Apply Business Rules (Module 11)

Apply filtering, boosting, and campaign rules:

```bash
python cli.py apply-rules \
  --rules output/rules.yaml \
  --input recommendations.json \
  --output final_recommendations.json
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `standardize` | Module 7: Convert to canonical format |
| `validate` | Module 8: Validate datasets |
| `generate-features` | Module 9: Generate ML features |
| `train-model` | Module 10: Train recommendation models |
| `recommend` | Generate recommendations for a user |
| `batch-recommend` | Generate recommendations for multiple users |
| `apply-rules` | Module 11: Apply business rules |
| `status` | Show system status |

### Common Options

- `--config, -c`: Path to `rec_config.json` (default: `output/rec_config.json`)
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress non-error output

## Module Details

### Module 7: Standardized Data Layer

**Goal**: Convert mapped client data into canonical recommendation-ready datasets.

**Key Components**:
- `canonical_schema.py`: Defines canonical data models (users, items, transactions, interactions)
- `mapping_applier.py`: Applies confirmed schema mappings from `rec_config.json`
- `metric_builder.py`: Creates derived metrics (net_sales, profit_margin, CLV)
- `lineage_manager.py`: Tracks transformation history for auditability

**Output**: Parquet files in `output/standardized/`

### Module 8: Validation Layer

**Goal**: Validate standardized datasets before ML processing.

**Validation Checks**:
- Null validation (critical vs optional columns)
- Data type validation
- Uniqueness checks (primary keys, duplicates)
- Range validation (numeric bounds)
- Relationship integrity (foreign keys)
- Business rule validation

**Outputs**:
- Quality scores (0-100 scale)
- Severity-classified issues (critical, warning, info)
- Invalid record reports

### Module 9: Feature Engineering

**Goal**: Generate ML-ready features for recommendation models.

**Feature Types**:
- **Aggregation**: Group-by statistics (sum, mean, count)
- **Temporal**: Time-based patterns (hour, day, seasonality)
- **Behavioral**: User action sequences
- **Recency**: Days since last activity
- **Frequency**: Activity counts over time windows
- **Monetary**: Spending patterns
- **Category Affinity**: Preference scores per category
- **Customer Lifetime Value**: Predicted future value

**Features**:
- Feature versioning for reproducibility
- Feature metadata tracking
- Feature selection capabilities

### Module 10: Recommendation Engine

**Architecture**: Modern hybrid deep learning recommendation system.

#### Candidate Generation (Retrieval)
- **Three-Tower Model**: User, Item, and Context towers for embedding learning
- **Collaborative Filtering**: Matrix factorization fallback
- **Popularity Engine**: Trending items fallback
- **ANN Search**: Approximate nearest neighbor for large-scale retrieval

#### Ranking
- **DLRM**: Deep Learning Recommendation Model for personalized scoring
- **XGBoost/LightGBM**: Gradient boosting fallbacks
- **Interaction Scoring**: User-item interaction modeling

#### Re-Ranking
- **Diversity**: Avoid repetitive recommendations
- **Freshness**: Promote new items
- **Margin Booster**: Prioritize high-margin products
- **Cold Start**: Handle new users/items
- **Exploration**: Balance exploration vs exploitation

#### Evaluation Metrics
- Precision@K, Recall@K
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- CTR simulation
- Diversity score
- Coverage score

### Module 11: Rule Engine

**Goal**: Apply business rules after ML recommendations.

**Rule Types**:
- **Filtering Rules**: Remove ineligible items
- **Boosting Rules**: Increase scores for strategic items
- **Conditional Rules**: Context-dependent logic
- **Campaign Rules**: Time-limited promotions
- **Context Rules**: Device, location, time-based rules

**Features**:
- YAML-driven configuration
- Rule chaining with conditions
- Explainability logging
- Score adjustment tracking

### Module 12: Recommendation Serving

**Goal**: CLI-based recommendation serving with caching and export.

**Capabilities**:
- Single-user real-time recommendations
- Batch recommendations from CSV
- Multiple export formats (JSON, CSV, Parquet)
- Caching layer with TTL
- Session state management
- Trace logging for debugging
- Human-readable explanations

## Configuration

The primary configuration file is `output/rec_config.json`, which contains:

- Trusted field mappings
- Business meanings
- Metric definitions
- Canonical role assignments
- User-confirmed schema mappings

## Example Workflow

```bash
# 1. Check system status
python cli.py status

# 2. Standardize source data
python cli.py standardize

# 3. Validate standardized data
python cli.py validate

# 4. Generate features
python cli.py generate-features

# 5. Train models
python cli.py train-model --model-types xgboost,lightgbm,dlrm

# 6. Get recommendations for a user
python cli.py recommend --user 1001 --top-k 10

# 7. Apply business rules
python cli.py apply-rules --rules rules.yaml --input recs.json

# 8. Export final recommendations
python cli.py recommend --user 1001 --output final_recs.json --format json
```

## Testing

Run tests for individual modules:

```bash
# Module 7 tests
python -m pytest Standardized_Data_Layer/test_standardization.py -v

# Module 8 tests
python -m pytest Validation_Layer/test_validation.py -v

# Module 9 tests
python -m pytest Feature_Engineering/test_features.py -v

# Module 10 tests
python -m pytest Recommendation_Engine/test_recommendation.py -v

# Module 11 tests
python -m pytest Rule_Engine/test_rules.py -v

# Module 12 tests
python -m pytest Recommendation_Serving/test_serving.py -v
```

## Architecture Decisions

### Why CLI-Based?
- Simplicity for initial deployment
- Easy integration into existing pipelines
- Backend-ready design for future migration
- Clear separation of concerns

### Why Parquet-First?
- Columnar storage for efficient analytics
- Schema enforcement
- Compression for large datasets
- Native support in pandas/polars/spark

### Why Modular Design?
- Independent testing of components
- Easy replacement of algorithms
- Scalable team development
- Future microservices migration path

### Why Three-Tower Architecture?
- Separates user, item, and context modeling
- Enables efficient candidate retrieval
- Supports cold-start scenarios
- Industry-standard for large-scale recommendations

## Scalability Roadmap

### Current State (CLI)
- Single-machine execution
- File-based data exchange
- Synchronous processing

### Phase 1: Parallel Processing
- Add multiprocessing support
- Batch parallelization
- Distributed feature computation

### Phase 2: Backend Integration
- Migrate to FastAPI
- Async recommendation serving
- Redis caching layer
- PostgreSQL for metadata

### Phase 3: Distributed Systems
- Apache Spark for feature engineering
- Kafka for real-time events
- Kubernetes deployment
- Model serving with TorchServe/Triton

## Contributing

1. Follow the modular architecture pattern
2. Write unit tests for new components
3. Use structured logging
4. Document public APIs with docstrings
5. Keep CLI files as orchestrators only (no business logic)

## License

See LICENSE file.

## Support

For issues or questions, please open an issue on GitHub.
