"""Hallucination Detection Framework v4.

Enhanced with:
- Flexible attention layer selection
- Batch task_type processing
- Lookback Lens hallucination annotation
- API-based hallucination verification
- lapeigvals-compatible output format
"""
__version__ = "4.1.0"
__author__ = "Hallucination Detection Team"

# Core modules (always available)
from . import core
from . import datasets
from . import annotators

# Optional modules (require additional dependencies)
def _lazy_import(name):
    """Lazy import for optional modules."""
    import importlib
    try:
        return importlib.import_module(f".{name}", __package__)
    except ImportError as e:
        import warnings
        warnings.warn(f"Module '{name}' not available: {e}")
        return None

# Try to import torch-dependent modules
try:
    from . import models
    from . import features
    from . import methods
    from . import evaluation
    from . import pipelines
    from . import outputs
except ImportError:
    models = None
    features = None
    methods = None
    evaluation = None
    pipelines = None
    outputs = None

__all__ = [
    "core",
    "datasets",
    "annotators",
    "models",
    "features",
    "methods",
    "evaluation",
    "pipelines",
    "outputs",
    "__version__",
]
