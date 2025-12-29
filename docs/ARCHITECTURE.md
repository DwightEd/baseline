# Architecture Documentation

## Overview

The Hallucination Detection Framework v4 is designed with industrial-grade robustness, maintainability, and extensibility in mind. It follows a modular architecture with clear separation of concerns.

## Design Principles

### 1. High Cohesion, Low Coupling
Each module handles a specific responsibility and communicates with other modules through well-defined interfaces.

### 2. Registry Pattern
All major components (datasets, models, feature extractors, methods) use a generic registry pattern that enables:
- Runtime dynamic registration
- Decorator-based registration
- Alias support
- Instance caching

### 3. Type Safety
Pydantic models ensure strict type validation for all configurations and data structures.

### 4. Configuration-Code Separation
All parameters are defined in YAML configuration files, with environment variable support.

## Module Dependency Graph

```
                    ┌─────────┐
                    │  core   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌─────▼─────┐
    │datasets │    │ models  │    │ features  │
    └────┬────┘    └────┬────┘    └─────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                    ┌────▼────┐
                    │ methods │
                    └────┬────┘
                         │
                    ┌────▼─────┐
                    │evaluation│
                    └──────────┘
```

## Core Module (`src/core/`)

### Registry (`registry.py`)
```python
class Registry(Generic[T]):
    """Generic registry for component registration."""
    
    def register(name, aliases=None, **metadata)
    def get(name) -> Type[T]
    def create(name, *args, **kwargs) -> T
    def get_or_create(name, *args, **kwargs) -> T
    def list_registered() -> List[str]
```

### Global Registries
- `DATASET_REGISTRY` - Dataset parsers
- `MODEL_REGISTRY` - Model loaders
- `FEATURE_EXTRACTOR_REGISTRY` - Feature extractors
- `METHOD_REGISTRY` - Detection methods
- `EVALUATOR_REGISTRY` - Evaluators

### Types (`types.py`)
All Pydantic models for type validation:

```
DataSample
├── id: str
├── question: str
├── answer: str
├── gold_answer: str
├── context: Optional[str]
├── task_type: TaskType
├── split: Optional[SplitType]
├── label: Optional[int]
└── metadata: Dict[str, Any]

LabeledSample(DataSample)
├── has_hallucination: bool
├── hallucination_spans: List[HallucinationSpan]
└── source_model: Optional[str]

ExtractedFeatures
├── sample_id: str
├── prompt_length: int
├── response_length: int
├── total_length: int
├── attention_features: Optional[Dict]
├── hidden_state_features: Optional[Dict]
├── token_prob_features: Optional[Dict]
└── metadata: Dict[str, Any]
```

### Configuration Hierarchy
```
PipelineConfig
├── project_name: str
├── random_seed: int
├── device: str
├── log_level: str
├── dataset: DatasetConfig
├── model: ModelConfig
├── feature_extraction: FeatureExtractionConfig
├── generation: GenerationConfig
├── training: TrainingConfig
├── evaluation: EvaluationConfig
├── output_dir: Path
└── cache_dir: Path
```

### Exception Hierarchy
```
HallucinationDetectionError
├── ConfigurationError
├── DatasetError
│   ├── DatasetNotFoundError
│   └── DatasetParseError
├── ModelError
│   ├── ModelLoadError
│   └── ModelInferenceError
├── FeatureExtractionError
├── MethodError
│   └── MethodNotFittedError
├── EvaluationError
├── CacheError
├── ValidationError
└── PipelineError
```

## Datasets Module (`src/datasets/`)

### Interface
```python
class BaseDatasetParser(ABC):
    def parse() -> Iterator[DataSample]
    def load_all(max_samples=None) -> List[DataSample]
    def filter(split=None, task_type=None, predicate=None) -> Iterator[DataSample]
    def get_statistics() -> Dict[str, Any]
```

### Registered Parsers
| Name | Aliases | Format |
|------|---------|--------|
| `json` | - | JSON |
| `jsonl` | `jsonlines` | JSON Lines |
| `csv` | `tsv` | CSV/TSV |
| `parquet` | - | Parquet |
| `ragtruth` | `rag_truth`, `RAGTruth` | RAGTruth specific |
| `gsm8k` | `math`, `gsm` | GSM8K |
| `triviaqa` | - | TriviaQA |
| `truthfulqa` | - | TruthfulQA |
| `halueval` | `halu_eval` | HaluEval |
| `coqa` | - | CoQA |
| `nqopen` | `natural_questions`, `nq` | Natural Questions |
| `squadv2` | `squad`, `squad2` | SQuAD v2.0 |

## Models Module (`src/models/`)

### Interface
```python
@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: ModelConfig
    device: torch.device
    
    def generate(texts, generation_config=None) -> List[str]
    def forward(input_ids, attention_mask=None, ...) -> Dict[str, Any]
    def encode(text) -> torch.Tensor
    def decode(token_ids) -> str
```

### Model Manager
```python
class ModelManager:
    def load(name_or_config, cache=True) -> LoadedModel
    def unload(name) -> bool
    def unload_all() -> None
    def list_loaded() -> List[str]
    def get_memory_usage() -> Dict[str, Any]
```

## Features Module (`src/features/`)

### Extraction Modes
1. **Teacher Forcing**: Concatenate prompt + answer, extract features from response portion
2. **Generation**: Generate response, extract features during generation

### Feature Types
```python
@dataclass
class AttentionFeatures:
    weights: Optional[torch.Tensor]
    eigenvalues: Optional[torch.Tensor]
    laplacian_eigenvalues: Optional[torch.Tensor]
    row_entropy: Optional[torch.Tensor]

@dataclass
class HiddenStateFeatures:
    states: Optional[torch.Tensor]
    pca_reduced: Optional[torch.Tensor]
    pooled: Optional[torch.Tensor]
    layer_norms: Optional[torch.Tensor]

@dataclass
class TokenProbFeatures:
    probs: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]
    perplexity: Optional[float]
    top_k_probs: Optional[torch.Tensor]
```

### Registered Extractors
| Name | Aliases | Description |
|------|---------|-------------|
| `default` | `standard`, `full` | Full extraction |
| `attention_only` | - | Attention only |
| `probability_only` | - | Token probs only |

## Methods Module (`src/methods/`)

### Interface
```python
class BaseMethod(ABC):
    def fit(features, labels, val_features=None, val_labels=None) -> MethodMetrics
    def predict(features) -> Prediction
    def predict_batch(features) -> List[Prediction]
    def save(path) -> None
    def load(path) -> None
    
    @property
    def is_fitted(self) -> bool
```

### Registered Methods
| Name | Aliases | Description |
|------|---------|-------------|
| `lapeigvals` | `laplacian_eigenvalues`, `spectral` | Laplacian eigenvalue analysis |
| `entropy` | `token_entropy` | Token entropy based |
| `perplexity` | - | Perplexity thresholding |
| `random_forest` | `rf` | Random Forest classifier |
| `ensemble` | - | Ensemble of methods |

### Prediction Output
```python
@dataclass
class Prediction:
    score: float      # Probability of hallucination
    label: int        # Binary label (0/1)
    confidence: float # Prediction confidence
    metadata: Dict[str, Any]
```

## Evaluation Module (`src/evaluation/`)

### Metrics
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under PR Curve)
- Accuracy
- F1 Score
- Precision
- Recall
- Specificity
- Confusion Matrix
- ROC Curve
- PR Curve
- Bootstrap Confidence Intervals

### Evaluator
```python
class Evaluator:
    def evaluate(predictions, labels) -> EvaluationResult
    def find_optimal_threshold(predictions, labels, metric="f1") -> Tuple[float, float]
    def compare_methods(method_predictions, labels) -> Dict[str, EvaluationResult]
```

## Data Flow

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│  Dataset │───▶│   Model  │───▶│  Feature  │───▶│  Method  │───▶│ Evaluation │
│  Parser  │    │  Loader  │    │ Extractor │    │ Detector │    │   Result   │
└──────────┘    └──────────┘    └───────────┘    └──────────┘    └────────────┘
     │               │               │                │                │
     ▼               ▼               ▼                ▼                ▼
DataSample      LoadedModel   ExtractedFeatures  Prediction    EvaluationResult
```

## Configuration System

### Environment Variable Support
```yaml
path: "${VAR_NAME}"           # Required variable
path: "${VAR_NAME:default}"   # With default value
```

### Environment-specific Overrides
```yaml
default:
  batch_size: 32

development:
  batch_size: 8     # Override for dev

production:
  batch_size: 64    # Override for prod
```

### Deep Merge Behavior
```python
# Base config
{"a": 1, "b": {"c": 2, "d": 3}}

# Override config
{"b": {"c": 10}, "e": 5}

# Result
{"a": 1, "b": {"c": 10, "d": 3}, "e": 5}
```

## Caching System

### Feature Cache
- Stores PyTorch tensors efficiently
- Hit/miss statistics
- Automatic key generation

### Metadata Cache
- JSON-serializable data
- Configuration snapshots
- Index files

### Cache Key Computation
```python
def compute_hash(*args, **kwargs) -> str:
    # Stable hash for reproducibility
    # Handles complex nested structures
```

## Extension Points

### Adding a New Dataset Parser
1. Create class inheriting `BaseDatasetParser`
2. Implement `component_name()` and `parse()` methods
3. Register with `@DATASET_REGISTRY.register("name")`

### Adding a New Detection Method
1. Create class inheriting `BaseMethod`
2. Implement required abstract methods
3. Register with `@METHOD_REGISTRY.register("name")`

### Adding a New Feature Extractor
1. Create class inheriting `BaseFeatureExtractor`
2. Implement `extract()` method
3. Register with `@FEATURE_EXTRACTOR_REGISTRY.register("name")`
