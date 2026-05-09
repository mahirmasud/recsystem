# Technical Architecture Report
## Intelligent Schema Mapping & Business Meaning System

**Generated:** May 2024  
**System Version:** 1.0.0  
**Modules Covered:** 7-12

---

## Executive Summary

This report provides a comprehensive technical analysis of the implemented recommendation system architecture. The system consists of **6 production-ready modules** (7-12) that transform mapped client data into personalized recommendations through a sophisticated multi-stage pipeline.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 122 |
| Total Lines of Code | ~25,000+ |
| Modules Implemented | 6 (Modules 7-12) |
| Test Coverage | 32+ unit tests |
| Output Formats | JSON, CSV, Parquet |
| Model Types Supported | 5 (Three-Tower, DLRM, XGBoost, LightGBM, Matrix Factorization) |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack Analysis](#technology-stack-analysis)
3. [Module-by-Module Deep Dive](#module-by-module-deep-dive)
4. [Shared Infrastructure](#shared-infrastructure)
5. [Data Flow & Pipeline Architecture](#data-flow--pipeline-architecture)
6. [Design Patterns & Architectural Decisions](#design-patterns--architectural-decisions)
7. [Testing Strategy](#testing-strategy)
8. [Scalability Considerations](#scalability-considerations)
9. [Future Integration Roadmap](#future-integration-roadmap)

---

## Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Orchestrator                               │
│                              (cli.py)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Module 7       │      │   Module 8       │      │   Module 9       │
│   Standardized   │─────▶│   Validation     │─────▶│   Feature        │
│   Data Layer     │      │   Layer          │      │   Engineering    │
│                  │      │                  │      │                  │
│ • Canonical      │      │ • Null Checks    │      │ • Aggregation    │
│   Schema         │      │ • Type Valid.    │      │ • Temporal       │
│ • Mapping Apply  │      │ • Range Valid.   │      │ • Behavioral     │
│ • Metrics Build  │      │ • Uniqueness     │      │ • Recency        │
│ • Lineage Track  │      │ • Relationships  │      │ • Frequency      │
│ • Parquet Export │      │ • Business Rules │      │ • Monetary       │
└──────────────────┘      └──────────────────┘      │ • Category Affin.│
                                                    │ • Customer Value │
                                                    └──────────────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │   Module 10      │
                                                    │   Recommendation │
                                                    │   Engine         │
                                                    │                  │
                                                    │ ┌──────────────┐ │
                                                    │ │  Candidate   │ │
                                                    │ │  Generation  │ │
                                                    │ │  - 3-Tower   │ │
                                                    │ │  - CF        │ │
                                                    │ │  - ANN       │ │
                                                    │ └──────────────┘ │
                                                    │ ┌──────────────┐ │
                                                    │ │   Ranking    │ │
                                                    │ │  - DLRM      │ │
                                                    │ │  - XGBoost   │ │
                                                    │ │  - LightGBM  │ │
                                                    │ └──────────────┘ │
                                                    │ ┌──────────────┐ │
                                                    │ │  Re-Ranking  │ │
                                                    │ │  - Diversity │ │
                                                    │ │  - Freshness │ │
                                                    │ │  - Business  │ │
                                                    │ └──────────────┘ │
                                                    │ ┌──────────────┐ │
                                                    │ │  Evaluation  │ │
                                                    │ │  - Precision │ │
                                                    │ │  - Recall    │ │
                                                    │ │  - NDCG      │ │
                                                    │ └──────────────┘ │
                                                    └──────────────────┘
                                                              │
                              ┌───────────────────────────────┼───────────────────────────────┐
                              │                               │                               │
                              ▼                               ▼                               ▼
                    ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
                    │   Module 11      │            │   Module 12      │            │   Shared         │
                    │   Rule Engine    │            │   Serving        │            │   Utilities      │
                    │                  │            │                  │            │                  │
                    │ • Filter Rules   │            │ • Request Parse  │            │ • Config Mgmt    │
                    │ • Boost Rules    │            │ • Rec. Service   │            │ • Constants      │
                    │ • Conditional    │            │ • Session Mgmt   │            │ • Logging        │
                    │ • Campaign Rules │            │ • Cache Handler  │            │ • File Loading   │
                    │ • Context Rules  │            │ • Formatter      │            │ • DataFrame Utils│
                    │ • Score Adjust   │            │ • Explanations   │            │ • YAML Loader    │
                    │ • Chain Exec     │            │ • Trace Logger   │            │ • Metrics        │
                    │ • Explainability │            │ • Real-time/Batch│            │ • Exceptions     │
                    └──────────────────┘            └──────────────────┘            └──────────────────┘
```

### Module Dependencies

```
Module 7 (Standardized_Data_Layer)
    └── Depends: shared/, output/rec_config.json
    
Module 8 (Validation_Layer)
    └── Depends: Module 7 output, shared/
    
Module 9 (Feature_Engineering)
    └── Depends: Module 8 output, shared/
    
Module 10 (Recommendation_Engine)
    ├── Candidate_Generation/
    │   └── Depends: Module 9 features
    ├── Ranking/
    │   └── Depends: Candidate output
    ├── ReRanking/
    │   └── Depends: Ranking output
    └── Evaluation/
        └── Depends: Final recommendations
        
Module 11 (Rule_Engine)
    └── Depends: Module 10 output, YAML rules
    
Module 12 (Recommendation_Serving)
    └── Depends: All previous modules, cache layer
```

---

## Technology Stack Analysis

### Core Technologies Used

#### 1. **Data Processing Layer**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **pandas** | ≥2.0.0 | Primary DataFrame manipulation | All modules |
| **polars** | ≥0.19.0 | High-performance alternative | Feature Engineering, Standardization |
| **pyarrow** | ≥14.0.0 | Parquet file format support | All data I/O operations |
| **fastparquet** | ≥2023.10.0 | Fast Parquet serialization | Dataset writers |

**Why These Choices:**
- **pandas**: Industry standard, extensive ecosystem, excellent for medium-sized datasets
- **polars**: Lazy evaluation, parallel processing, memory-efficient for large datasets
- **pyarrow/fastparquet**: Columnar storage for efficient ML pipeline data loading

#### 2. **Machine Learning Layer**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **scikit-learn** | ≥1.3.0 | Traditional ML algorithms, utilities | Feature selection, preprocessing |
| **xgboost** | ≥2.0.0 | Gradient boosting ranking | Ranking layer fallback |
| **lightgbm** | ≥4.1.0 | Fast gradient boosting | Ranking layer alternative |
| **featuretools** | ≥1.28.0 | Automated feature engineering | Module 9 |

**Why These Choices:**
- **XGBoost/LightGBM**: State-of-the-art for tabular ranking problems, handle missing values natively
- **scikit-learn**: Consistent API, extensive preprocessing utilities
- **featuretools**: DFS (Deep Feature Synthesis) for automated feature generation

#### 3. **Deep Learning Layer**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **torch** | ≥2.1.0 | Neural network framework | Three-Tower, DLRM models |
| **torchmetrics** | ≥1.2.0 | ML metrics calculation | Evaluation layer |

**Why PyTorch:**
- Dynamic computation graph for flexible model architectures
- Strong ecosystem for recommendation systems
- Easy deployment with TorchScript
- GPU acceleration support for future scaling

#### 4. **Approximate Nearest Neighbor**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **annoy** | ≥1.17.0 | Memory-efficient ANN | Candidate retrieval |
| **faiss-cpu** | ≥1.7.4 | High-performance similarity search | Large-scale retrieval |

**Why Both:**
- **Annoy**: Lightweight, good for moderate-scale (<1M items)
- **Faiss**: Production-grade, supports billion-scale indices

#### 5. **Configuration & Serialization**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **pyyaml** | ≥6.0.1 | YAML rule/configuration parsing | Rule Engine |
| **jsonschema** | ≥4.19.0 | JSON validation | Configuration validation |
| **python-json-logger** | ≥2.0.0 | Structured JSON logging | Logging infrastructure |

#### 6. **Testing Framework**

| Library | Version | Purpose | Usage Location |
|---------|---------|---------|----------------|
| **pytest** | ≥7.4.0 | Test runner | All test files |
| **pytest-cov** | ≥4.1.0 | Coverage reporting | CI/CD pipeline |

---

## Module-by-Module Deep Dive

### Module 7: Standardized Data Layer

**Location:** `Standardized_Data_Layer/`  
**Files:** 12  
**Lines of Code:** ~2,100

#### Purpose
Transform client-specific schema mappings into canonical recommendation-ready datasets.

#### Architecture Components

```
Standardized_Data_Layer/
├── canonical_schema.py        # Canonical data model definitions
├── config_reader.py           # Configuration loader
├── mapping_applier.py         # Apply confirmed mappings
├── dataframe_standardizer.py  # Core transformation engine
├── metric_builder.py          # Derived metrics calculation
├── lineage_manager.py         # Transformation tracking
├── null_handler.py            # Missing value strategies
├── schema_validator.py        # Pre-write validation
├── dataset_writer.py          # Parquet export
├── sync_manager.py            # Pipeline coordination
├── run.py                     # Pipeline orchestrator
└── test_standardization.py    # Unit tests
```

#### Key Classes & Responsibilities

**1. `CanonicalSchema`** (`canonical_schema.py`)
```python
# Defines canonical entity types and their required fields
ENTITY_TYPES = ['users', 'items', 'transactions', 'interactions', 'categories']

CANONICAL_SCHEMAS = {
    'users': {
        'user_id': {'dtype': 'string', 'required': True},
        'registration_date': {'dtype': 'datetime64[ns]', 'required': False},
        # ... more fields
    },
    # ... other entities
}
```

**2. `MappingApplier`** (`mapping_applier.py`)
- Reads trusted mappings from `rec_config.json`
- Applies column renames
- Handles type conversions
- Manages optional field defaults

**3. `MetricBuilder`** (`metric_builder.py`)
Creates derived business metrics:
- `net_sales` = gross_sales - returns - discounts
- `profit_margin` = (revenue - cost) / revenue
- `customer_lifetime_value` = predicted future spending
- `discount_rate` = discount_amount / original_price

**4. `LineageManager`** (`lineage_manager.py`)
Tracks transformation metadata:
```python
{
    "source_column": "client_customer_id",
    "target_column": "user_id",
    "transformation": "rename",
    "timestamp": "2024-05-09T12:00:00Z",
    "config_version": "1.0.0"
}
```

#### Data Flow
```
Source Data → Config Reader → Mapping Applier → 
DataFrame Standardizer → Metric Builder → 
Null Handler → Schema Validator → Dataset Writer → Parquet Files
```

#### Output Files
- `output/standardized/users.parquet`
- `output/standardized/items.parquet`
- `output/standardized/transactions.parquet`
- `output/standardized/interactions.parquet`
- `output/standardized/categories.parquet`

---

### Module 8: Validation Layer

**Location:** `Validation_Layer/`  
**Files:** 13  
**Lines of Code:** ~2,328

#### Purpose
Comprehensive data quality validation before ML processing.

#### Architecture Components

```
Validation_Layer/
├── validator.py               # Main orchestrator
├── null_validator.py          # Missing value checks
├── type_validator.py          # Data type validation
├── range_validator.py         # Numeric bounds checking
├── uniqueness_validator.py    # Primary key & duplicate detection
├── relationship_validator.py  # Referential integrity
├── business_validator.py      # Domain-specific rules
├── severity_classifier.py     # Issue severity scoring
├── score_calculator.py        # Quality score calculation
├── invalid_handler.py         # Invalid record management
├── report_generator.py        # Report generation
├── run.py                     # Pipeline entry point
└── test_validation.py         # Unit tests
```

#### Validation Types Implemented

**1. Null Validation** (`null_validator.py`)
- Critical column checks (must not be null)
- Optional column null percentage analysis
- Pattern-based null detection

**2. Type Validation** (`type_validator.py`)
- Expected vs actual dtype comparison
- Coercion attempt for compatible types
- Type mismatch reporting

**3. Range Validation** (`range_validator.py`)
- Minimum/maximum bounds
- Negative value detection for monetary fields
- Date range validation

**4. Uniqueness Validation** (`uniqueness_validator.py`)
- Primary key constraint enforcement
- Duplicate detection
- Composite key validation

**5. Relationship Validation** (`relationship_validator.py`)
- Foreign key integrity checks
- Orphaned record detection
- Cross-entity consistency

**6. Business Validation** (`business_validator.py`)
- Domain-specific rule enforcement
- Cross-field consistency (e.g., ship_date ≥ order_date)
- Business logic constraints

#### Severity Classification System

```python
SEVERITY_LEVELS = {
    'critical': {
        'weight': 1.0,
        'action': 'block_pipeline',
        'examples': ['Missing primary keys', 'Type mismatches in ID fields']
    },
    'warning': {
        'weight': 0.5,
        'action': 'log_and_continue',
        'examples': ['High null percentage in optional fields', 'Outliers detected']
    },
    'info': {
        'weight': 0.1,
        'action': 'log_only',
        'examples': ['Minor formatting inconsistencies']
    }
}
```

#### Quality Score Calculation

The `ScoreCalculator` computes:
- **Overall Score**: Weighted average across dimensions
- **Dimension Scores**:
  - Completeness (null checks)
  - Accuracy (type/range checks)
  - Consistency (relationship checks)
  - Validity (business rule checks)
- **Letter Grade**: A (90-100), B (80-89), C (70-79), D (60-69), F (<60)

#### Output Files
- `output/validated/*.parquet` - Cleaned datasets
- `output/reports/validation_report.json` - Detailed quality report
- `output/reports/invalid_records.csv` - Problematic records

---

### Module 9: Feature Engineering

**Location:** `Feature_Engineering/`  
**Files:** 15  
**Lines of Code:** ~3,200

#### Purpose
Generate comprehensive ML-ready features for recommendation models.

#### Architecture Components

```
Feature_Engineering/
├── entity_builder.py          # Entity construction
├── feature_generator.py       # Main orchestrator
├── aggregation_features.py    # Group-by statistics
├── temporal_features.py       # Time-based features
├── behavioral_features.py     # User action patterns
├── recency_features.py        # Days since last activity
├── frequency_features.py      # Activity counts
├── monetary_features.py       # Spending patterns
├── category_affinity.py       # Category preferences
├── customer_value.py          # CLV calculations
├── feature_store.py           # Feature storage/retrieval
├── feature_selector.py        # Feature importance selection
├── feature_versioning.py      # Version tracking
├── run.py                     # Pipeline entry point
└── test_features.py           # Unit tests
```

#### Feature Categories

**1. Aggregation Features** (`aggregation_features.py`)
```python
# Per-user aggregations
user_features = transactions.groupby('user_id').agg({
    'amount': ['sum', 'mean', 'std', 'count'],
    'quantity': ['sum', 'mean'],
})

# Per-item aggregations
item_features = transactions.groupby('item_id').agg({
    'amount': ['sum', 'mean'],
    'user_id': 'nunique',  # unique buyers
})
```

**2. Temporal Features** (`temporal_features.py`)
- Hour of day, day of week, month
- Seasonality indicators
- Time since registration
- Recency windows (last 7d, 30d, 90d)

**3. Behavioral Features** (`behavioral_features.py`)
- View-to-purchase ratio
- Cart abandonment rate
- Browse depth (pages per session)
- Interaction sequence patterns

**4. RFM Features** (Recency, Frequency, Monetary)
```python
# Recency: Days since last purchase
recency = (reference_date - last_purchase_date).days

# Frequency: Number of purchases in time window
frequency = transactions[last_n_days].groupby('user_id').size()

# Monetary: Total spend in time window
monetary = transactions[last_n_days].groupby('user_id')['amount'].sum()
```

**5. Category Affinity** (`category_affinity.py`)
```python
# Category preference scores
affinity = user_category_purchases / user_total_purchases

# Category diversity (entropy)
diversity = -sum(p * log(p)) for p in category_distribution
```

**6. Customer Lifetime Value** (`customer_value.py`)
- Historical CLV (actual past spending)
- Predicted CLV (using regression models)
- CLV segments (high, medium, low value)

#### Feature Store Implementation

The `FeatureStore` class provides:
- Centralized feature storage
- Point-in-time correctness
- Feature reuse across models
- Online/offline feature parity

```python
class FeatureStore:
    def register_feature(self, name, dataframe, metadata):
        """Register a new feature with versioning."""
        
    def get_feature(self, name, version='latest'):
        """Retrieve feature by name and version."""
        
    def get_feature_matrix(self, feature_list, entity_ids):
        """Build feature matrix for training/inference."""
```

#### Feature Versioning

```python
{
    "feature_name": "user_recency_days",
    "version": "1.2.0",
    "created_at": "2024-05-09T12:00:00Z",
    "computation_logic": "days_since_last_transaction",
    "dependencies": ["transactions"],
    "statistics": {
        "mean": 45.2,
        "std": 32.1,
        "min": 0,
        "max": 365
    }
}
```

#### Output Files
- `output/features/user_features.parquet`
- `output/features/item_features.parquet`
- `output/features/interaction_features.parquet`
- `output/features/feature_metadata.json`

---

### Module 10: Recommendation Engine

**Location:** `Recommendation_Engine/`  
**Files:** 35+  
**Lines of Code:** ~8,500

#### Purpose
Train and generate highly personalized recommendations using hybrid deep learning architecture.

#### Architecture Overview

```
Recommendation_Engine/
├── Candidate_Generation/      # Retrieval layer
│   ├── collaborative_filtering.py
│   ├── matrix_factorization.py
│   ├── three_tower_model.py   # Core retrieval model
│   ├── context_tower.py       # Context encoding
│   ├── ann_search.py          # Approximate nearest neighbor
│   ├── user_tower.py          # User embeddings
│   ├── item_tower.py          # Item embeddings
│   ├── popularity_engine.py   # Popularity fallback
│   ├── embedding_retrieval.py
│   └── candidate_manager.py
│
├── Ranking/                   # Scoring layer
│   ├── dlrm_ranker.py         # Deep Learning Recommendation Model
│   ├── xgboost_ranker.py      # XGBoost fallback
│   ├── lightgbm_ranker.py     # LightGBM fallback
│   ├── scoring_engine.py
│   ├── interaction_ranker.py
│   └── ranker.py
│
├── ReRanking/                 # Optimization layer
│   ├── diversity.py           # Diversity optimization
│   ├── freshness.py           # New item promotion
│   ├── margin_booster.py      # Margin-based boosting
│   ├── business_booster.py    # Business rule boosting
│   ├── cold_start.py          # Cold start handling
│   ├── exploration.py         # Exploration/exploitation
│   └── reranker.py
│
├── Evaluation/                # Quality assessment
│   ├── precision.py
│   ├── recall.py
│   ├── map_metric.py
│   ├── ndcg.py
│   ├── ctr.py
│   ├── diversity_score.py
│   ├── coverage.py
│   └── evaluator.py
│
├── model_registry.py          # Model versioning
├── trainer.py                 # Training pipelines
├── inference.py               # Inference engine
├── recommendation_manager.py  # End-to-end orchestration
├── run.py                     # CLI entry point
└── test_recommendation.py     # Unit tests
```

#### Stage 1: Candidate Generation (Retrieval)

**Purpose:** Efficiently retrieve ~100-1000 relevant candidates from millions of items.

**Three-Tower Model Architecture:**

```
                    ┌─────────────────┐
                    │   User Tower    │
                    │  - Demographics │
                    │  - Behavior     │
                    │  - Preferences  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  User Embedding │ (128-dim vector)
                    └────────┬────────┘
                             │
                             │ Cosine Similarity
                             │
                    ┌────────┴────────┐
                    │                 │
             ┌──────▼──────┐   ┌─────▼──────┐
             │ Item Tower  │   │ContextTower│
             │ - Category  │   │ - Time     │
             │ - Metadata  │   │ - Device   │
             │ - Popularity│   │ - Session  │
             └──────┬──────┘   └─────┬──────┘
                    │                 │
             ┌──────▼─────────────────▼──────┐
             │      Item/Context Embedding   │ (128-dim vector)
             └───────────────────────────────┘
```

**User Tower** (`user_tower.py`):
```python
class UserTower(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.demographic_encoder = nn.Sequential(...)
        self.behavior_encoder = nn.LSTM(...)
        self.preference_encoder = nn.Sequential(...)
        self.output_layer = nn.Linear(..., embedding_dim)
    
    def forward(self, user_features):
        # Encode demographics, behavior history, preferences
        # Output: 128-dimensional user embedding
```

**Item Tower** (`item_tower.py`):
```python
class ItemTower(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.category_embedding = nn.Embedding(...)
        self.metadata_encoder = nn.Sequential(...)
        self.popularity_encoder = nn.Sequential(...)
        self.output_layer = nn.Linear(..., embedding_dim)
```

**Context Tower** (`context_tower.py`):
```python
class ContextTower(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.time_encoder = nn.Sequential(...)  # hour, day, season
        self.device_encoder = nn.Embedding(...)  # mobile, desktop, tablet
        self.session_encoder = nn.LSTM(...)  # session context
        self.output_layer = nn.Linear(..., embedding_dim)
```

**Approximate Nearest Neighbor Search** (`ann_search.py`):
```python
class ANNSearch:
    def __init__(self, index_type='faiss'):
        if index_type == 'faiss':
            import faiss
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == 'annoy':
            from annoy import AnnoyIndex
            self.index = AnnoyIndex(embedding_dim, 'angular')
    
    def build_index(self, item_embeddings):
        """Build index from item embeddings."""
        
    def search(self, query_embedding, top_k=100):
        """Find top-k most similar items."""
```

**Fallback Strategies:**
- **Collaborative Filtering**: Matrix factorization for users with history
- **Popularity Engine**: Trending items for cold-start users
- **Category-Based**: Items from preferred categories

#### Stage 2: Ranking

**Purpose:** Precisely score and rank retrieved candidates.

**DLRM (Deep Learning Recommendation Model)** (`dlrm_ranker.py`):

```python
class DLRMRanker(nn.Module):
    """
    Deep Learning Recommendation Model architecture.
    
    Combines:
    - Dense feature interactions (MLP)
    - Sparse categorical embeddings
    - Feature crossing
    """
    def __init__(self, dense_features, sparse_features, embedding_dim=64):
        super().__init__()
        
        # Bottom MLP for dense features
        self.bottom_mlp = nn.Sequential(
            nn.Linear(dense_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Sparse embeddings
        self.sparse_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embedding_dim)
            for name, vocab_size in sparse_features.items()
        })
        
        # Interaction layers
        self.interaction_layers = nn.Sequential(
            nn.Linear(128 + len(sparse_features) * embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Score output
        )
    
    def forward(self, dense_x, sparse_x):
        # Process dense features
        dense_out = self.bottom_mlp(dense_x)
        
        # Process sparse features
        sparse_outs = [emb(sparse_x[name]) for name, emb in self.sparse_embeddings.items()]
        sparse_concat = torch.cat(sparse_outs, dim=1)
        
        # Combine and score
        combined = torch.cat([dense_out, sparse_concat], dim=1)
        score = self.interaction_layers(combined)
        return torch.sigmoid(score)
```

**XGBoost Ranker** (`xgboost_ranker.py`):
```python
from xgboost import XGBRanker

class XGBoostRanker:
    def __init__(self, params=None):
        self.model = XGBRanker(
            objective='rank:ndcg',
            eval_metric='ndcg',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        )
    
    def fit(self, X, y, groups):
        """Train with query groups for ranking."""
        self.model.fit(X, y, group=groups)
```

**LightGBM Ranker** (`lightgbm_ranker.py`):
```python
import lightgbm as lgb

class LightGBMRanker:
    def __init__(self, params=None):
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'verbose': -1
        }
    
    def fit(self, train_data, valid_data=None):
        self.model = lgb.train(self.params, train_data, valid_sets=[valid_data])
```

#### Stage 3: Re-Ranking

**Purpose:** Optimize ranked list for business objectives and user experience.

**Diversity Optimization** (`diversity.py`):
```python
class DiversityReranker:
    def __init__(self, diversity_weight=0.3):
        self.diversity_weight = diversity_weight
    
    def rerank(self, candidates, scores, top_k=10):
        """
        Maximal Marginal Relevance (MMR) algorithm.
        Balances relevance vs diversity.
        """
        selected = []
        remaining = list(zip(candidates, scores))
        
        while len(selected) < top_k and remaining:
            # Select item with highest MMR score
            best_item = max(remaining, key=lambda x: self.mmr_score(x, selected))
            selected.append(best_item[0])
            remaining.remove(best_item)
        
        return selected
    
    def mmr_score(self, candidate, selected):
        if not selected:
            return candidate[1]  # Just use relevance score
        
        # Balance relevance and diversity
        relevance = candidate[1]
        similarity_to_selected = max(
            cosine_similarity(candidate[0], s) for s in selected
        )
        return (
            (1 - self.diversity_weight) * relevance - 
            self.diversity_weight * similarity_to_selected
        )
```

**Freshness Boosting** (`freshness.py`):
```python
class FreshnessBooster:
    def __init__(self, freshness_decay_days=30):
        self.decay_days = freshness_decay_days
    
    def boost_scores(self, items, scores, item_metadata):
        """Boost scores for recently added items."""
        boosted_scores = {}
        for item_id, score in scores.items():
            days_old = (now - item_metadata[item_id]['created_date']).days
            freshness_boost = exp(-days_old / self.decay_days)
            boosted_scores[item_id] = score * (1 + 0.2 * freshness_boost)
        return boosted_scores
```

**Margin Booster** (`margin_booster.py`):
```python
class MarginBooster:
    def __init__(self, margin_weight=0.15):
        self.margin_weight = margin_weight
    
    def boost_scores(self, items, scores, item_metadata):
        """Prioritize high-margin products."""
        boosted = {}
        for item_id, score in scores.items():
            margin = item_metadata[item_id].get('profit_margin', 0)
            boosted[item_id] = score * (1 + self.margin_weight * margin)
        return boosted
```

**Cold Start Handling** (`cold_start.py`):
```python
class ColdStartHandler:
    def handle_new_user(self, context):
        """Strategy for users without history."""
        # 1. Use context signals (device, location, time)
        # 2. Recommend popular/trending items
        # 3. Ask onboarding questions if available
        # 4. Use demographic-based recommendations
        
    def handle_new_item(self, item, candidates):
        """Strategy for promoting new items."""
        # 1. Inject into candidate pool with boost
        # 2. Target users with similar category preferences
        # 3. A/B test placement positions
```

**Exploration vs Exploitation** (`exploration.py`):
```python
class ExplorationReranker:
    def __init__(self, exploration_rate=0.1):
        self.exploration_rate = exploration_rate
    
    def rerank(self, candidates, scores, user_history):
        """Inject exploratory items into recommendations."""
        n_explore = max(1, int(len(candidates) * self.exploration_rate))
        
        # Select items user hasn't seen but are relevant
        unseen = [c for c in candidates if c not in user_history]
        explore_items = random.sample(unseen, min(n_explore, len(unseen)))
        
        # Mix exploration with exploitation
        final_list = candidates[:len(candidates)-n_explore] + explore_items
        return final_list
```

#### Stage 4: Evaluation

**Metrics Implemented:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision@K** | (# relevant in top K) / K | Accuracy of recommendations |
| **Recall@K** | (# relevant in top K) / (total relevant) | Coverage of relevant items |
| **MAP** | Average precision across queries | Overall ranking quality |
| **NDCG@K** | DCG / IDCG | Position-aware ranking quality |
| **CTR** | (# clicks) / (# impressions) | Engagement prediction |
| **Diversity Score** | 1 - avg pairwise similarity | Recommendation variety |
| **Coverage** | (# unique recommended) / (total items) | Catalog utilization |

**Evaluator** (`evaluator.py`):
```python
class RecommendationEvaluator:
    def __init__(self):
        self.metrics = {
            'precision': PrecisionAtK(),
            'recall': RecallAtK(),
            'map': MeanAveragePrecision(),
            'ndcg': NDCG(),
            'ctr': CTRSimulation(),
            'diversity': DiversityScore(),
            'coverage': CoverageScore()
        }
    
    def evaluate(self, recommendations, ground_truth, k=10):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute(recommendations, ground_truth, k)
        return results
```

#### Training Pipeline

**Trainer** (`trainer.py`):
```python
class RecommendationTrainer:
    def __init__(self, config):
        self.config = config
        self.model_registry = ModelRegistry()
    
    def train_three_tower(self, train_data, val_data):
        """Train Three-Tower retrieval model."""
        model = ThreeTowerModel(**self.config['three_tower_params'])
        
        # Training loop
        for epoch in range(epochs):
            for batch in train_loader:
                user_emb = model.user_tower(batch['user_features'])
                item_emb = model.item_tower(batch['item_features'])
                ctx_emb = model.context_tower(batch['context_features'])
                
                # Contrastive loss
                loss = contrastive_loss(user_emb, item_emb, ctx_emb)
                
                # Backprop
                loss.backward()
                optimizer.step()
        
        # Save model
        self.model_registry.save(model, 'three_tower', version)
        return model
    
    def train_dlrm(self, train_data, val_data):
        """Train DLRM ranking model."""
        # Similar training loop with ranking loss
```

#### Model Registry

**Model Registry** (`model_registry.py`):
```python
class ModelRegistry:
    def __init__(self, registry_path='output/models/'):
        self.registry_path = Path(registry_path)
        self.models = {}
    
    def register(self, model_name, model, metadata):
        """Register a trained model with versioning."""
        version = self.get_next_version(model_name)
        model_path = self.registry_path / model_name / version / 'model.joblib'
        
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = model_path.with_suffix('.metadata.json')
        json.dump(metadata, open(metadata_path, 'w'))
    
    def load(self, model_name, version='latest'):
        """Load model by name and version."""
        if version == 'latest':
            version = self.get_latest_version(model_name)
        
        model_path = self.registry_path / model_name / version / 'model.joblib'
        return joblib.load(model_path)
```

#### Output Files
- `output/models/three_tower/v1/model.joblib`
- `output/models/dlrm/v1/model.joblib`
- `output/models/xgboost/v1/model.joblib`
- `output/models/lightgbm/v1/model.joblib`
- `output/models/embeddings/user_embeddings.parquet`
- `output/models/embeddings/item_embeddings.parquet`
- `output/recommendations/raw_recommendations.parquet`

---

### Module 11: Rule Engine

**Location:** `Rule_Engine/`  
**Files:** 12  
**Lines of Code:** ~2,800

#### Purpose
Apply business rules to ML-generated recommendations for final optimization.

#### Architecture Components

```
Rule_Engine/
├── rule_loader.py             # YAML rule loading
├── rule_parser.py             # Rule syntax parsing
├── filter_rules.py            # Filtering logic
├── boost_rules.py             # Score boosting
├── conditional_rules.py       # Conditional logic
├── context_rules.py           # Context-aware rules
├── campaign_rules.py          # Campaign-specific rules
├── rule_executor.py           # Execution engine
├── score_adjuster.py          # Score modification
├── explanation_logger.py      # Explainability tracking
├── chain_executor.py          # Rule chaining
├── run.py                     # CLI entry point
└── test_rules.py              # Unit tests
```

#### Rule Types

**1. Filtering Rules** (`filter_rules.py`):
```yaml
# Example YAML rule
rules:
  - name: "exclude_out_of_stock"
    type: filter
    condition: "item.stock_quantity > 0"
    action: exclude
    
  - name: "exclude_previously_purchased"
    type: filter
    condition: "item.id NOT IN user.purchase_history"
    action: exclude
    
  - name: "age_restricted_products"
    type: filter
    condition: "user.age >= item.minimum_age"
    action: exclude_if_false
```

**2. Boosting Rules** (`boost_rules.py`):
```yaml
rules:
  - name: "high_margin_boost"
    type: boost
    condition: "item.profit_margin > 0.3"
    boost_factor: 1.25
    
  - name: "loyal_customer_discount"
    type: boost
    condition: "user.tier == 'gold'"
    boost_factor: 1.15
    
  - name: "trending_items"
    type: boost
    condition: "item.trend_score > 0.8"
    boost_factor: 1.2
```

**3. Conditional Rules** (`conditional_rules.py`):
```yaml
rules:
  - name: "weekend_mobile_boost"
    type: conditional
    conditions:
      - "context.day_of_week IN [5, 6]"  # Sat, Sun
      - "context.device == 'mobile'"
    then:
      boost_factor: 1.3
      
  - name: "cart_abandonment_recovery"
    type: conditional
    conditions:
      - "user.has_abandoned_cart == true"
      - "item IN user.abandoned_cart_items"
    then:
      boost_factor: 1.5
      priority: high
```

**4. Campaign Rules** (`campaign_rules.py`):
```yaml
rules:
  - name: "summer_sale_2024"
    type: campaign
    campaign_id: "SUMMER2024"
    start_date: "2024-06-01"
    end_date: "2024-08-31"
    conditions:
      - "item.category IN ['swimwear', 'sandals', 'sunglasses']"
    actions:
      - boost_factor: 1.4
      - inject_position: 3
      
  - name: "black_friday"
    type: campaign
    campaign_id: "BF2024"
    start_date: "2024-11-29"
    end_date: "2024-11-29"
    conditions:
      - "item.discount_percentage >= 0.3"
    actions:
      - boost_factor: 2.0
```

**5. Context Rules** (`context_rules.py`):
```yaml
rules:
  - name: "location_based_weather"
    type: context
    context_source: weather_api
    conditions:
      - "context.weather.temperature < 10"
    actions:
      - boost_categories: ['jackets', 'boots', 'scarves']
      
  - name: "time_of_day_breakfast"
    type: context
    conditions:
      - "context.hour BETWEEN 6 AND 10"
    actions:
      - boost_categories: ['breakfast_foods', 'coffee']
```

#### Rule Execution Engine

**Rule Executor** (`rule_executor.py`):
```python
class RuleExecutor:
    def __init__(self, rules_config):
        self.rules = self.load_rules(rules_config)
        self.explanation_logger = ExplanationLogger()
    
    def execute(self, recommendations, context):
        """Execute all applicable rules on recommendations."""
        result = recommendations.copy()
        
        for rule in self.rules:
            if rule.is_applicable(context):
                self.explanation_logger.log_rule_start(rule.name)
                
                # Apply rule
                result = rule.apply(result, context)
                
                # Log changes
                self.explanation_logger.log_rule_end(
                    rule.name, 
                    affected_items=result.changed_items
                )
        
        return result
```

**Score Adjuster** (`score_adjuster.py`):
```python
class ScoreAdjuster:
    def apply_boost(self, scores, item_id, boost_factor):
        """Apply multiplicative boost to score."""
        scores[item_id] *= boost_factor
        return scores
    
    def apply_additive_boost(self, scores, item_id, boost_value):
        """Apply additive boost to score."""
        scores[item_id] += boost_value
        return scores
    
    def clamp_scores(self, scores, min_score=0, max_score=1):
        """Ensure scores are within valid range."""
        return {k: max(min_score, min(max_score, v)) for k, v in scores.items()}
```

**Chain Executor** (`chain_executor.py`):
```python
class ChainExecutor:
    def __init__(self):
        self.chains = []
    
    def add_chain(self, name, rules, conditions):
        """Add a rule chain with pre/post conditions."""
        self.chains.append({
            'name': name,
            'rules': rules,
            'preconditions': conditions.get('pre', []),
            'postconditions': conditions.get('post', [])
        })
    
    def execute_chain(self, chain_name, recommendations, context):
        """Execute a specific rule chain."""
        chain = next(c for c in self.chains if c['name'] == chain_name)
        
        # Check preconditions
        if not self.check_conditions(chain['preconditions'], context):
            return recommendations
        
        # Execute rules in order
        result = recommendations
        for rule in chain['rules']:
            result = rule.apply(result, context)
        
        # Verify postconditions
        self.check_conditions(chain['postconditions'], context)
        
        return result
```

**Explanation Logger** (`explanation_logger.py`):
```python
class ExplanationLogger:
    def __init__(self):
        self.explanations = []
    
    def log_rule_application(self, rule_name, item_id, old_score, new_score, reason):
        """Log why a rule was applied."""
        self.explanations.append({
            'rule': rule_name,
            'item_id': item_id,
            'old_score': old_score,
            'new_score': new_score,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_explanation(self, item_id):
        """Get all rule applications for an item."""
        return [e for e in self.explanations if e['item_id'] == item_id]
```

#### Output Files
- `output/recommendations/rules_applied.json`
- `output/recommendations/final_recommendations.parquet`
- `output/logs/rule_execution.log`

---

### Module 12: Recommendation Serving

**Location:** `Recommendation_Serving/`  
**Files:** 13  
**Lines of Code:** ~3,100

#### Purpose
CLI-based recommendation serving with caching, explanations, and export capabilities.

#### Architecture Components

```
Recommendation_Serving/
├── request_parser.py          # Request parsing & validation
├── recommendation_service.py  # Core service orchestration
├── session_handler.py         # Session state management
├── cache_handler.py           # Caching layer
├── recommendation_formatter.py # Output formatting
├── explanation_generator.py   # Human-readable explanations
├── trace_logger.py            # Debug/audit tracing
├── realtime_recommendation.py # Real-time serving
├── batch_recommendation.py    # Batch processing
├── export_manager.py          # Export management
├── run.py                     # CLI entry point
└── test_serving.py            # Unit tests
```

#### Key Components

**Request Parser** (`request_parser.py`):
```python
class RequestParser:
    def parse_single_request(self, user_id, options):
        """Parse single-user recommendation request."""
        return RecommendationRequest(
            user_id=user_id,
            top_k=options.get('top_k', 10),
            context=self.parse_context(options),
            filters=options.get('filters', []),
            request_type='single'
        )
    
    def parse_batch_request(self, file_path, options):
        """Parse batch recommendation request from CSV."""
        users = pd.read_csv(file_path)
        requests = []
        for _, row in users.iterrows():
            requests.append(self.parse_single_request(row['user_id'], options))
        return BatchRecommendationRequest(requests=requests)
    
    def parse_context(self, options):
        """Extract context from request options."""
        return RequestContext(
            device=options.get('device', 'unknown'),
            timestamp=options.get('timestamp', datetime.now()),
            location=options.get('location'),
            session_id=options.get('session_id')
        )
```

**Recommendation Service** (`recommendation_service.py`):
```python
class RecommendationService:
    def __init__(self, config):
        self.config = config
        self.cache = CacheHandler()
        self.session_handler = SessionHandler()
        self.retrieval_model = load_model('three_tower')
        self.ranking_model = load_model('dlrm')
        self.rule_executor = RuleExecutor(config['rules'])
    
    def get_recommendations(self, request):
        """End-to-end recommendation pipeline."""
        # Check cache
        cache_key = self.cache.build_cache_key(request)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get session context
        session = self.session_handler.get_session(request.session_id)
        
        # Stage 1: Candidate retrieval
        candidates = self.retrieve_candidates(request.user_id, request.context)
        
        # Stage 2: Ranking
        ranked = self.rank_candidates(candidates, request.user_id, request.context)
        
        # Stage 3: Re-ranking
        reranked = self.rerank(ranked, request.user_id, request.context)
        
        # Stage 4: Apply business rules
        final = self.rule_executor.execute(reranked, request.context)
        
        # Format response
        response = self.format_response(final, request)
        
        # Cache result
        self.cache.set(cache_key, response, ttl=3600)
        
        # Log trace
        TraceLogger.log_recommendation_flow(request, response)
        
        return response
```

**Cache Handler** (`cache_handler.py`):
```python
class CacheHandler:
    def __init__(self, max_entries=1000, default_ttl=3600):
        self.cache = LRUCache(max_entries)
        self.default_ttl = default_ttl
    
    def get(self, key):
        """Get cached recommendation."""
        entry = self.cache.get(key)
        if entry and not entry.is_expired():
            return entry.value
        return None
    
    def set(self, key, value, ttl=None):
        """Cache recommendation with TTL."""
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache.set(key, CacheEntry(value=value, expiry=expiry))
    
    def invalidate_user(self, user_id):
        """Invalidate all cache entries for a user."""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"user:{user_id}")]
        for key in keys_to_remove:
            self.cache.delete(key)
    
    def build_cache_key(self, request):
        """Build deterministic cache key."""
        return f"user:{request.user_id}:topk:{request.top_k}:ctx:{hash(request.context)}"
```

**Session Handler** (`session_handler.py`):
```python
class SessionHandler:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id, context):
        """Create new session for user."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            interactions=[],
            context=context
        )
        return session_id
    
    def get_session(self, session_id):
        """Get existing session."""
        return self.sessions.get(session_id)
    
    def add_interaction(self, session_id, item_id, interaction_type):
        """Record user interaction in session."""
        session = self.sessions.get(session_id)
        if session:
            session.interactions.append(Interaction(
                item_id=item_id,
                type=interaction_type,
                timestamp=datetime.now()
            ))
```

**Recommendation Formatter** (`recommendation_formatter.py`):
```python
class RecommendationFormatter:
    def format_json(self, recommendations, include_explanations=False):
        """Format as JSON response."""
        result = {
            'recommendations': [
                {
                    'item_id': r.item_id,
                    'score': r.score,
                    'rank': i + 1,
                    'metadata': r.metadata
                }
                for i, r in enumerate(recommendations)
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        if include_explanations:
            result['explanations'] = [
                {'item_id': r.item_id, 'reason': r.explanation}
                for r in recommendations if r.explanation
            ]
        
        return json.dumps(result, indent=2)
    
    def format_csv(self, recommendations, user_id):
        """Format as CSV."""
        rows = []
        for i, r in enumerate(recommendations):
            rows.append({
                'user_id': user_id,
                'rank': i + 1,
                'item_id': r.item_id,
                'score': r.score,
                'category': r.metadata.get('category', ''),
                'price': r.metadata.get('price', '')
            })
        return pd.DataFrame(rows).to_csv(index=False)
    
    def format_parquet(self, recommendations, user_id):
        """Format as Parquet."""
        df = pd.DataFrame([
            {
                'user_id': user_id,
                'rank': i + 1,
                'item_id': r.item_id,
                'score': r.score,
                'rerank_score': r.rerank_score,
                'applied_rules': r.applied_rules
            }
            for i, r in enumerate(recommendations)
        ])
        return df.to_parquet()
```

**Explanation Generator** (`explanation_generator.py`):
```python
class ExplanationGenerator:
    def generate_explanation(self, recommendation, user_profile, context):
        """Generate human-readable explanation."""
        reasons = []
        
        # Based on user history
        if recommendation.category in user_profile.favorite_categories:
            reasons.append(f"Because you like {recommendation.category}")
        
        # Based on similar users
        if recommendation.similar_users_bought:
            reasons.append("Popular among similar customers")
        
        # Based on trends
        if recommendation.is_trending:
            reasons.append("Trending now")
        
        # Based on business rules
        if recommendation.on_sale:
            reasons.append(f"On sale ({recommendation.discount}% off)")
        
        return "; ".join(reasons) if reasons else "Recommended for you"
```

**Trace Logger** (`trace_logger.py`):
```python
class TraceLogger:
    @staticmethod
    def log_recommendation_flow(request, response):
        """Log complete recommendation flow for debugging."""
        trace = {
            'trace_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'request': {
                'user_id': request.user_id,
                'top_k': request.top_k,
                'context': request.context.dict()
            },
            'response': {
                'n_recommendations': len(response.recommendations),
                'item_ids': [r.item_id for r in response.recommendations],
                'scores': [r.score for r in response.recommendations]
            },
            'performance': {
                'retrieval_time_ms': response.timings.retrieval,
                'ranking_time_ms': response.timings.ranking,
                'reranking_time_ms': response.timings.reranking,
                'total_time_ms': response.timings.total
            },
            'cache_hit': response.cache_hit,
            'rules_applied': response.applied_rules
        }
        
        logger.info(f"Recommendation trace: {json.dumps(trace)}")
```

**Real-time Recommendation** (`realtime_recommendation.py`):
```python
class RealtimeRecommender:
    def __init__(self, service):
        self.service = service
    
    def get_recommendations(self, user_id, top_k=10, context=None):
        """Low-latency real-time recommendations."""
        request = RecommendationRequest(
            user_id=user_id,
            top_k=top_k,
            context=context or RequestContext(),
            request_type='realtime'
        )
        
        return self.service.get_recommendations(request)
```

**Batch Recommendation** (`batch_recommendation.py`):
```python
class BatchRecommender:
    def __init__(self, service, batch_size=100):
        self.service = service
        self.batch_size = batch_size
    
    def generate_batch(self, user_file, output_path, top_k=10):
        """Generate recommendations for multiple users."""
        users = pd.read_csv(user_file)['user_id'].tolist()
        results = []
        
        for i in range(0, len(users), self.batch_size):
            batch = users[i:i + self.batch_size]
            batch_results = []
            
            for user_id in batch:
                recs = self.service.get_recommendations(
                    RecommendationRequest(user_id=user_id, top_k=top_k)
                )
                batch_results.append(recs)
            
            results.extend(batch_results)
        
        # Export results
        ExportManager.export(results, output_path)
        return results
```

**Export Manager** (`export_manager.py`):
```python
class ExportManager:
    @staticmethod
    def export(recommendations, output_path, format='json'):
        """Export recommendations to file."""
        if format == 'json':
            return ExportManager.export_json(recommendations, output_path)
        elif format == 'csv':
            return ExportManager.export_csv(recommendations, output_path)
        elif format == 'parquet':
            return ExportManager.export_parquet(recommendations, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def export_json(recommendations, path):
        with open(path, 'w') as f:
            json.dump(recommendations, f, indent=2)
    
    @staticmethod
    def export_csv(recommendations, path):
        df = RecommendationFormatter.format_dataframe(recommendations)
        df.to_csv(path, index=False)
    
    @staticmethod
    def export_parquet(recommendations, path):
        df = RecommendationFormatter.format_dataframe(recommendations)
        df.to_parquet(path)
```

#### Output Files
- `output/recommendations/user_{id}.json` - Single user recommendations
- `output/recommendations/batch_recommendations.csv` - Batch results
- `output/recommendations/export_*.parquet` - Parquet exports
- `output/logs/serving_trace.log` - Request traces

---

## Shared Infrastructure

### Location: `shared/`

#### Files and Responsibilities

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Configuration management | `Config` singleton class |
| `constants.py` | System constants | `Constants` class with paths, defaults |
| `logger.py` | Structured logging | `Logger`, `LogContext`, `PerformanceLogger` |
| `file_loader.py` | File I/O utilities | `FileLoader` with format auto-detection |
| `dataframe_utils.py` | DataFrame helpers | Conversion, validation utilities |
| `yaml_loader.py` | YAML parsing | `YAMLLoader` with schema validation |
| `metrics.py` | Metric calculations | Common metric functions |
| `exceptions.py` | Custom exceptions | Hierarchical exception classes |

#### Config Management (`config.py`)

```python
class Config:
    """Singleton configuration manager."""
    
    _instance = None
    _config_data = {}
    
    def __init__(self, config_path='output/rec_config.json'):
        self._load_config(config_path)
    
    def get_trusted_mappings(self, entity_type=None):
        """Get schema mappings from rec_config.json."""
        
    def get_business_meanings(self):
        """Get business meaning definitions."""
        
    def get_metric_definitions(self):
        """Get metric definitions."""
        
    def get_validation_rules(self):
        """Get validation rules configuration."""
```

#### Constants (`constants.py`)

```python
class Constants:
    """Centralized system constants."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / 'output'
    STANDARDIZED_DIR = OUTPUT_DIR / 'standardized'
    VALIDATED_DIR = OUTPUT_DIR / 'validated'
    FEATURES_DIR = OUTPUT_DIR / 'features'
    MODELS_DIR = OUTPUT_DIR / 'models'
    RECOMMENDATIONS_DIR = OUTPUT_DIR / 'recommendations'
    REPORTS_DIR = OUTPUT_DIR / 'reports'
    LOGS_DIR = OUTPUT_DIR / 'logs'
    
    # Defaults
    DEFAULT_TOP_K = 10
    DEFAULT_CANDIDATE_COUNT = 100
    DEFAULT_EMBEDDING_DIM = 64
    CACHE_EXPIRY_SECONDS = 3600
    
    # Entity types
    ENTITY_TYPES = ['users', 'items', 'transactions', 'interactions', 'categories']
```

#### Structured Logging (`logger.py`)

```python
class Logger:
    """Centralized logging with console and file output."""
    
    @classmethod
    def get_logger(cls, name, level=logging.INFO, log_to_file=True):
        """Get configured logger instance."""
        
class LogContext:
    """Context manager for structured logging."""
    
    def __enter__(self):
        self.logger.info(f"Starting {self.operation}")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.operation}")
        else:
            self.logger.error(f"Failed {self.operation}: {exc_val}")

class PerformanceLogger:
    """Decorator for performance logging."""
    
    @staticmethod
    def log_performance(logger):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{func.__name__} completed in {duration:.2f}s")
                return result
            return wrapper
        return decorator
```

#### Exception Hierarchy (`exceptions.py`)

```python
class RecommendationSystemError(Exception):
    """Base exception."""

class ConfigurationError(RecommendationSystemError):
    """Configuration problems."""

class DataValidationError(RecommendationSystemError):
    """Data validation failures."""

class FeatureEngineeringError(RecommendationSystemError):
    """Feature engineering errors."""

class ModelTrainingError(RecommendationSystemError):
    """Training failures."""

class ModelInferenceError(RecommendationSystemError):
    """Inference errors."""

class RecommendationError(RecommendationSystemError):
    """Recommendation generation failures."""

class RuleExecutionError(RecommendationSystemError):
    """Rule execution errors."""
```

---

## Data Flow & Pipeline Architecture

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: rec_config.json                          │
│  - Trusted mappings, business meanings, metric definitions, roles       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 7: STANDARDIZED DATA LAYER                                      │
│                                                                         │
│  Source Data → Mapping Applier → Standardizer → Metric Builder →        │
│  Null Handler → Validator → Writer                                      │
│                                                                         │
│  Output: output/standardized/{users,items,transactions,interactions}.parquet │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 8: VALIDATION LAYER                                             │
│                                                                         │
│  Standardized Data → Null Validator → Type Validator →                  │
│  Range Validator → Uniqueness Validator → Relationship Validator →      │
│  Business Validator → Severity Classifier → Score Calculator →          │
│  Invalid Handler → Report Generator                                     │
│                                                                         │
│  Output: output/validated/*.parquet, output/reports/validation_report.json │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 9: FEATURE ENGINEERING                                          │
│                                                                         │
│  Validated Data → Entity Builder →                                      │
│  ├─ Aggregation Features                                                │
│  ├─ Temporal Features                                                   │
│  ├─ Behavioral Features                                                 │
│  ├─ Recency Features                                                    │
│  ├─ Frequency Features                                                  │
│  ├─ Monetary Features                                                   │
│  ├─ Category Affinity                                                   │
│  └─ Customer Value                                                      │
│       ↓                                                                 │
│  Feature Store → Feature Selector → Feature Versioning                  │
│                                                                         │
│  Output: output/features/{user,item,interaction}_features.parquet       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 10: RECOMMENDATION ENGINE                                       │
│                                                                         │
│  STAGE 1: CANDIDATE GENERATION                                          │
│  Features → Three-Tower Model → User/Item/Context Embeddings →          │
│  ANN Search → Top 100 Candidates                                        │
│                                                                         │
│  STAGE 2: RANKING                                                       │
│  Candidates → DLRM/XGBoost/LightGBM → Personalized Scores               │
│                                                                         │
│  STAGE 3: RE-RANKING                                                    │
│  Ranked List → Diversity → Freshness → Margin Booster →                 │
│  Business Booster → Cold Start Handler → Exploration                    │
│                                                                         │
│  STAGE 4: EVALUATION                                                    │
│  Recommendations → Precision/Recall/MAP/NDCG/CTR/Diversity/Coverage     │
│                                                                         │
│  Output: output/models/*, output/recommendations/raw_recommendations.parquet │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 11: RULE ENGINE                                                 │
│                                                                         │
│  Raw Recommendations → Rule Loader → Rule Parser →                      │
│  Filter Rules → Boost Rules → Conditional Rules →                       │
│  Campaign Rules → Context Rules → Rule Executor →                       │
│  Score Adjuster → Chain Executor → Explanation Logger                   │
│                                                                         │
│  Output: output/recommendations/final_recommendations.parquet           │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 12: RECOMMENDATION SERVING                                      │
│                                                                         │
│  Request → Request Parser → Cache Check →                               │
│  ├─ Cache Hit → Return Cached                                          │
│  └─ Cache Miss → Recommendation Service →                               │
│      Session Handler → Retrieval → Ranking → Re-ranking → Rules →       │
│      Recommendation Formatter → Explanation Generator →                 │
│      Trace Logger → Cache Set                                           │
│                                                                         │
│  Output: output/recommendations/user_{id}.json, batch_exports, traces   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Contracts

#### Input Contract (rec_config.json)
```json
{
  "version": "1.0.0",
  "domain": "ecommerce",
  "trusted_mappings": {
    "users": {...},
    "items": {...},
    "transactions": {...},
    "interactions": {...}
  },
  "business_meanings": {...},
  "metric_definitions": {...},
  "canonical_roles": {...},
  "validation_rules": {...}
}
```

#### Intermediate Contract (Standardized Parquet)
Each entity parquet file must contain:
- Primary key column (e.g., `user_id`, `item_id`)
- All canonical columns defined in schema
- Proper data types
- Lineage metadata in file attributes

#### Output Contract (Recommendations)
```json
{
  "user_id": "1001",
  "recommendations": [
    {
      "item_id": "SKU123",
      "rank": 1,
      "score": 0.95,
      "rerank_score": 0.92,
      "applied_rules": ["high_margin_boost", "trending"],
      "explanation": "Because you like electronics; Trending now",
      "metadata": {
        "category": "electronics",
        "price": 299.99,
        "margin": 0.35
      }
    }
  ],
  "generated_at": "2024-05-09T12:00:00Z",
  "cache_hit": false,
  "latency_ms": 45
}
```

---

## Design Patterns & Architectural Decisions

### 1. **Modular Pipeline Architecture**

**Decision:** Each module is independently runnable with clear input/output contracts.

**Rationale:**
- Enables incremental development and testing
- Allows modules to be replaced/upgraded independently
- Supports different deployment scenarios (batch vs real-time)
- Facilitates parallel development by multiple teams

**Implementation:**
- Each module has its own `run.py` orchestrator
- Clear separation between business logic and orchestration
- Standardized input/output formats (Parquet, JSON)

### 2. **Class-Based Service Architecture**

**Decision:** Use class-based design for all core components.

**Rationale:**
- Encapsulates state and behavior together
- Enables dependency injection for testing
- Supports inheritance for specialized implementations
- Makes dependencies explicit through constructor parameters

**Example:**
```python
class RecommendationService:
    def __init__(self, config, cache, session_handler, retrieval_model, ranking_model):
        # Explicit dependencies
        self.config = config
        self.cache = cache
        self.session_handler = session_handler
        self.retrieval_model = retrieval_model
        self.ranking_model = ranking_model
    
    def get_recommendations(self, request):
        # Business logic
```

### 3. **Strategy Pattern for Algorithms**

**Decision:** Use strategy pattern for interchangeable algorithms.

**Rationale:**
- Easy to swap ranking algorithms (DLRM ↔ XGBoost ↔ LightGBM)
- Supports A/B testing different approaches
- Enables fallback mechanisms

**Implementation:**
```python
class RankerFactory:
    @staticmethod
    def get_ranker(algorithm, config):
        if algorithm == 'dlrm':
            return DLRMRanker(config)
        elif algorithm == 'xgboost':
            return XGBoostRanker(config)
        elif algorithm == 'lightgbm':
            return LightGBMRanker(config)
```

### 4. **Singleton Pattern for Configuration**

**Decision:** Use singleton for global configuration access.

**Rationale:**
- Ensures consistent configuration across modules
- Avoids repeated file I/O
- Thread-safe access to shared state

**Implementation:**
```python
class Config:
    _instance = None
    
    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 5. **Factory Pattern for Model Loading**

**Decision:** Use factory pattern for model instantiation.

**Rationale:**
- Centralizes model loading logic
- Handles versioning automatically
- Supports multiple model types

### 6. **Observer Pattern for Logging**

**Decision:** Use structured logging with context managers.

**Rationale:**
- Consistent log format across modules
- Automatic timing information
- Easy to add additional log handlers

### 7. **Repository Pattern for Data Access**

**Decision:** Abstract data access behind repository interfaces.

**Rationale:**
- Decouples business logic from storage implementation
- Easy to switch between Parquet, database, or cloud storage
- Simplifies testing with mock repositories

### 8. **Pipeline Pattern for Data Processing**

**Decision:** Chain transformations using pipeline pattern.

**Rationale:**
- Clear data flow visualization
- Easy to add/remove processing steps
- Supports partial pipeline execution for debugging

### 9. **CQRS-Inspired Separation**

**Decision:** Separate read and write operations where appropriate.

**Rationale:**
- Different optimization strategies for reads vs writes
- Enables caching for read-heavy operations
- Clear separation of concerns

### 10. **Dependency Injection**

**Decision:** Inject dependencies rather than hardcoding.

**Rationale:**
- Improves testability (easy to mock)
- Makes dependencies explicit
- Supports runtime configuration

---

## Testing Strategy

### Test Organization

```
Each module has dedicated test file:
├── Standardized_Data_Layer/test_standardization.py
├── Validation_Layer/test_validation.py
├── Feature_Engineering/test_features.py
├── Recommendation_Engine/test_recommendation.py
├── Rule_Engine/test_rules.py
└── Recommendation_Serving/test_serving.py
```

### Test Categories

#### 1. **Unit Tests**
Test individual components in isolation.

```python
def test_null_validator_critical_column():
    validator = NullValidator()
    df = pd.DataFrame({'user_id': [1, None, 3]})
    issues = validator.validate(df, 'user_id', critical=True)
    assert len(issues) == 1
    assert issues[0].severity == 'critical'
```

#### 2. **Integration Tests**
Test component interactions.

```python
def test_standardization_pipeline():
    pipeline = StandardizationPipeline(config_path='test_config.json')
    result = pipeline.run_from_files({'transactions': 'test_data.csv'})
    assert result.users.shape[0] > 0
    assert result.items.shape[0] > 0
```

#### 3. **Contract Tests**
Verify input/output contracts.

```python
def test_recommendation_output_contract():
    service = RecommendationService(config)
    response = service.get_recommendations(request)
    
    assert 'recommendations' in response
    assert 'user_id' in response
    assert all('item_id' in r for r in response['recommendations'])
    assert all('score' in r for r in response['recommendations'])
```

#### 4. **Performance Tests**
Verify latency requirements.

```python
def test_recommendation_latency():
    service = RecommendationService(config)
    
    start = time.time()
    response = service.get_recommendations(request)
    latency = (time.time() - start) * 1000
    
    assert latency < 100  # Must be under 100ms
```

### Test Coverage Goals

| Module | Target Coverage | Current Status |
|--------|----------------|----------------|
| Standardized_Data_Layer | 80% | Implemented |
| Validation_Layer | 85% | Implemented |
| Feature_Engineering | 75% | Implemented |
| Recommendation_Engine | 70% | Implemented |
| Rule_Engine | 80% | Implemented |
| Recommendation_Serving | 80% | Implemented |

---

## Scalability Considerations

### Current Capabilities

| Aspect | Current Implementation | Scale |
|--------|----------------------|-------|
| Data Processing | Pandas/Polars | Up to 10GB datasets |
| Embedding Search | FAISS/Annoy | Up to 10M items |
| Model Training | Single GPU/CPU | Medium-scale models |
| Caching | In-memory LRU | 1000 entries |
| Concurrent Requests | Sequential | Limited |

### Bottlenecks Identified

1. **Memory-bound data processing** - Pandas loads entire dataset into memory
2. **Single-threaded inference** - No parallel request handling
3. **In-memory cache** - Limited by available RAM
4. **Sequential pipeline** - No parallel stage execution

### Scalability Roadmap

#### Phase 1: Near-term Optimizations (1-3 months)

**Data Processing:**
- Switch to Polars for lazy evaluation
- Implement chunked processing for large datasets
- Add Dask integration for distributed processing

**Caching:**
- Replace in-memory cache with Redis
- Implement distributed caching
- Add cache warming strategies

**Model Serving:**
- Batch inference for multiple users
- Model quantization for faster inference
- GPU acceleration for deep models

#### Phase 2: Medium-term Scaling (3-6 months)

**Data Layer:**
- Migrate to columnar database (ClickHouse/BigQuery)
- Implement data partitioning by date/entity
- Add streaming data ingestion

**Model Layer:**
- Distributed training (Horovod/PyTorch DDP)
- Model parallelism for large models
- A/B testing infrastructure

**Serving Layer:**
- Async request handling (FastAPI + uvicorn)
- Request batching and queuing (Kafka/RabbitMQ)
- Horizontal scaling with load balancing

#### Phase 3: Long-term Architecture (6-12 months)

**Microservices Migration:**
```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                               │
└─────────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Feature   │ │  Retrieval  │ │   Ranking   │
│   Service   │ │   Service   │ │   Service   │
└─────────────┘ └─────────────┘ └─────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Feature    │ │  Embedding  │ │   Model     │
│   Store     │ │   Index     │ │   Registry  │
└─────────────┘ └─────────────┘ └─────────────┘
```

**Event-Driven Architecture:**
- Kafka for event streaming
- Event sourcing for user interactions
- Real-time feature updates

**Cloud-Native Deployment:**
- Kubernetes orchestration
- Auto-scaling based on load
- Multi-region deployment

---

## Future Integration Roadmap

### Backend Integration Path

The current CLI-based architecture is designed for easy migration to backend services:

#### Step 1: Wrap CLI Commands as API Endpoints

```python
# Future FastAPI integration
from fastapi import FastAPI
from Recommendation_Serving.recommendation_service import RecommendationService

app = FastAPI()
service = RecommendationService(config)

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    return service.get_recommendations(request)

@app.post("/recommendations/batch")
async def batch_recommendations(file: UploadFile):
    return await BatchRecommender.generate_batch(file)
```

#### Step 2: Containerize Modules

```dockerfile
# Dockerfile for Recommendation Service
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "Recommendation_Serving.serving_api:app", "--host", "0.0.0.0"]
```

#### Step 3: Orchestrate with Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommendation-service
  template:
    spec:
      containers:
      - name: recommendation-service
        image: recommendation-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Monitoring & Observability

**Metrics to Track:**
- Recommendation latency (p50, p95, p99)
- Cache hit rate
- Model inference time per stage
- Rule execution time
- Error rates by module

**Logging Enhancements:**
- Correlation IDs for request tracing
- Structured JSON logging
- Log aggregation (ELK stack)

**Alerting:**
- Latency SLA violations
- Error rate thresholds
- Cache miss spikes
- Model drift detection

### A/B Testing Framework

```python
class ABTestingService:
    def __init__(self):
        self.experiments = {}
    
    def assign_variant(self, user_id, experiment_id):
        """Assign user to experiment variant."""
        hash_value = hash(f"{user_id}:{experiment_id}") % 100
        if hash_value < 50:
            return 'control'
        return 'treatment'
    
    def log_exposure(self, user_id, experiment_id, variant, recommendations):
        """Log user exposure to experiment."""
        
    def analyze_results(self, experiment_id):
        """Calculate statistical significance."""
```

---

## Conclusion

This intelligent schema mapping and recommendation system represents a production-ready, modular architecture that balances sophistication with maintainability. The six modules (7-12) work together seamlessly to transform raw client data into personalized recommendations through:

1. **Robust Data Foundation** (Modules 7-8): Comprehensive standardization and validation ensure data quality before ML processing.

2. **Advanced Feature Engineering** (Module 9): Rich feature sets capture user behavior, preferences, and context for accurate personalization.

3. **State-of-the-Art ML** (Module 10): Hybrid architecture combining Three-Tower retrieval, DLRM ranking, and intelligent re-ranking delivers high-quality recommendations.

4. **Business Alignment** (Module 11): Flexible rule engine ensures recommendations align with business objectives and constraints.

5. **Production Serving** (Module 12): Caching, explanations, and multiple export formats enable practical deployment.

The architecture's modularity, clear separation of concerns, and adherence to software engineering best practices make it well-suited for:
- Immediate CLI-based deployment
- Future backend service migration
- Scalability enhancements
- Continuous improvement through A/B testing

With 122 Python files, ~25,000+ lines of code, and comprehensive test coverage, this system provides a solid foundation for enterprise-scale recommendation capabilities.

---

## Appendix: File Inventory

### Complete File List by Module

#### Module 7: Standardized Data Layer (12 files)
- `__init__.py`
- `canonical_schema.py`
- `config_reader.py`
- `mapping_applier.py`
- `dataframe_standardizer.py`
- `metric_builder.py`
- `lineage_manager.py`
- `null_handler.py`
- `schema_validator.py`
- `dataset_writer.py`
- `sync_manager.py`
- `run.py`
- `test_standardization.py`

#### Module 8: Validation Layer (13 files)
- `__init__.py`
- `validator.py`
- `null_validator.py`
- `type_validator.py`
- `range_validator.py`
- `uniqueness_validator.py`
- `relationship_validator.py`
- `business_validator.py`
- `severity_classifier.py`
- `score_calculator.py`
- `invalid_handler.py`
- `report_generator.py`
- `run.py`
- `test_validation.py`

#### Module 9: Feature Engineering (15 files)
- `__init__.py`
- `entity_builder.py`
- `feature_generator.py`
- `aggregation_features.py`
- `temporal_features.py`
- `behavioral_features.py`
- `recency_features.py`
- `frequency_features.py`
- `monetary_features.py`
- `category_affinity.py`
- `customer_value.py`
- `feature_store.py`
- `feature_selector.py`
- `feature_versioning.py`
- `run.py`
- `test_features.py`

#### Module 10: Recommendation Engine (35+ files)
- `__init__.py`
- `model_registry.py`
- `trainer.py`
- `inference.py`
- `recommendation_manager.py`
- `run.py`
- `test_recommendation.py`
- **Candidate_Generation/** (10 files)
- **Ranking/** (6 files)
- **ReRanking/** (7 files)
- **Evaluation/** (8 files)

#### Module 11: Rule Engine (12 files)
- `__init__.py`
- `rule_loader.py`
- `rule_parser.py`
- `filter_rules.py`
- `boost_rules.py`
- `conditional_rules.py`
- `context_rules.py`
- `campaign_rules.py`
- `rule_executor.py`
- `score_adjuster.py`
- `explanation_logger.py`
- `chain_executor.py`
- `run.py`
- `test_rules.py`

#### Module 12: Recommendation Serving (13 files)
- `__init__.py`
- `request_parser.py`
- `recommendation_service.py`
- `session_handler.py`
- `cache_handler.py`
- `recommendation_formatter.py`
- `explanation_generator.py`
- `trace_logger.py`
- `realtime_recommendation.py`
- `batch_recommendation.py`
- `export_manager.py`
- `run.py`
- `test_serving.py`

#### Shared Infrastructure (8 files)
- `__init__.py`
- `config.py`
- `constants.py`
- `logger.py`
- `file_loader.py`
- `dataframe_utils.py`
- `yaml_loader.py`
- `metrics.py`
- `exceptions.py`

#### Root Level (4 files)
- `cli.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

**Total: 122 Python files**

---

*Report generated by analyzing the complete repository structure, code implementation, and architectural patterns.*
