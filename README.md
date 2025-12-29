# Hallucination Detection Framework v4

A professional, extensible framework for LLM hallucination detection with modular architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hallucination Detection                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Datasets │  │  Models  │  │ Features │  │     Methods      │ │
│  │ Registry │  │ Registry │  │ Registry │  │     Registry     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────────▼─────────┐ │
│  │ RAGTruth │  │HuggingFace│ │ Standard │  │   LapEigvals     │ │
│  │ GSM8K    │  │  Loader  │  │ Attention│  │   Entropy        │ │
│  │ TriviaQA │  │  Manager │  │ Probs    │  │   Perplexity     │ │
│  │ HaluEval │  └──────────┘  └──────────┘  │   RandomForest   │ │
│  │ CoQA     │                              │   Ensemble       │ │
│  └──────────┘                              └──────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                         Core Module                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │  Registry  │ │   Types    │ │   Config   │ │    Cache     │  │
│  │  Pattern   │ │  Pydantic  │ │  Manager   │ │   System     │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Registry Pattern**: Dynamic component registration with ABC base classes
- **Pydantic Validation**: Strict type checking and schema validation
- **Modular Configuration**: Environment-aware config with env var support
- **Feature Caching**: Avoid redundant computation
- **Multi-format Support**: JSON, JSONL, CSV, Parquet datasets
- **DVC Integration**: Reproducible ML pipelines

## Project Structure

```
hallucination-detection-v4/
├── src/
│   ├── core/           # Registry, types, config, cache, logging
│   ├── datasets/       # Dataset parsers (RAGTruth, GSM8K, etc.)
│   ├── models/         # Model loading and management
│   ├── features/       # Feature extraction
│   ├── methods/        # Detection methods
│   └── evaluation/     # Metrics and evaluation
├── config/
│   └── config.yaml     # Main configuration
├── scripts/            # CLI scripts
├── tests/              # Unit and integration tests
├── docs/               # Documentation
├── dvc.yaml            # DVC pipeline
├── params.yaml         # DVC parameters
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Environment

```bash
export RAGTRUTH_DATA_DIR=/path/to/ragtruth/dataset
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

### 2. Extract Features

```bash
python scripts/extract_features.py \
    --dataset ragtruth \
    --data_path $RAGTRUTH_DATA_DIR \
    --output_dir outputs/features \
    --mode teacher_forcing \
    --max_samples 1000
```

### 3. Train Methods

```bash
python scripts/train.py \
    --features_dir outputs/features \
    --methods lapeigvals entropy perplexity ensemble \
    --output_dir outputs/models
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --features_dir outputs/features \
    --models_dir outputs/models \
    --output_dir outputs/results
```

### Using DVC Pipeline

```bash
dvc repro
```

## Usage Examples

### Custom Dataset Parser

```python
from src.core import DATASET_REGISTRY
from src.datasets import BaseDatasetParser

@DATASET_REGISTRY.register("my_dataset")
class MyDatasetParser(BaseDatasetParser):
    @classmethod
    def component_name(cls) -> str:
        return "my_dataset"
    
    def parse(self):
        # Your implementation
        pass
```

### Custom Detection Method

```python
from src.core import METHOD_REGISTRY
from src.methods import BaseMethod, Prediction

@METHOD_REGISTRY.register("my_method")
class MyMethod(BaseMethod):
    @classmethod
    def component_name(cls) -> str:
        return "my_method"
    
    def fit(self, features, labels, **kwargs):
        # Training implementation
        pass
    
    def predict(self, features):
        return Prediction(score=0.5, label=0, confidence=0.8)
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
```

### Feature Extraction

```python
from src.features import create_extractor
from src.core import FeatureExtractionConfig, ExtractionMode

config = FeatureExtractionConfig(
    mode=ExtractionMode.TEACHER_FORCING,
    attention={"enabled": True, "compute_eigenvalues": True},
    token_probs={"enabled": True, "compute_entropy": True}
)

extractor = create_extractor(config)
features = extractor.extract(model, sample)
```

### Evaluation

```python
from src.evaluation import Evaluator, EvaluationResult

evaluator = Evaluator()
result = evaluator.evaluate(predictions, labels)

print(result.summary())
# AUROC:      0.8500
# AUPRC:      0.8000
# F1 Score:   0.7500
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAGTRUTH_DATA_DIR` | RAGTruth dataset path | `./data/ragtruth` |
| `MODEL_NAME` | Model name/path | `meta-llama/Llama-3.1-8B-Instruct` |
| `MODEL_PATH` | Local model path | - |
| `OUTPUT_DIR` | Output directory | `./outputs` |
| `CACHE_DIR` | Cache directory | `./cache` |

### Config File Structure

```yaml
default:
  dataset:
    name: ragtruth
    path: ${RAGTRUTH_DATA_DIR}
  
  model:
    name: ${MODEL_NAME}
    dtype: float16
  
  feature_extraction:
    mode: teacher_forcing
    attention:
      enabled: true
      compute_eigenvalues: true

development:
  dataset:
    max_samples: 100

production:
  feature_extraction:
    batch_size: 4
```

## Available Components

### Datasets
- `ragtruth` - RAGTruth hallucination dataset
- `gsm8k` - GSM8K math reasoning
- `triviaqa` - TriviaQA
- `truthfulqa` - TruthfulQA
- `halueval` - HaluEval
- `coqa` - CoQA dialogue
- `nqopen` - Natural Questions
- `squadv2` - SQuAD v2.0

### Detection Methods
- `lapeigvals` - Laplacian eigenvalues (spectral)
- `entropy` - Token entropy
- `perplexity` - Sequence perplexity
- `random_forest` - Random Forest classifier
- `ensemble` - Ensemble of methods

### Feature Extractors
- `default` - Full feature extraction
- `attention_only` - Attention features only
- `probability_only` - Token probability only

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## License

MIT License
