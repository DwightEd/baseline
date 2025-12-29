"""Hallucination annotation module.

This module provides methods for detecting and annotating hallucinations
in model-generated text, including:
- Lookback Lens: Attention-based detection (Chuang et al.)
- API-based: Using external LLMs for verification
- Gold labels: Using dataset's existing annotations
"""
from .base import (
    BaseAnnotator,
    AnnotationResult,
    AnnotationConfig,
    AnnotationMethod,
)
from .lookback_lens import (
    LookbackLensAnnotator,
    LookbackLensConfig,
)
from .api_annotator import (
    APIAnnotator,
    APIAnnotationConfig,
)

__all__ = [
    # Base
    "BaseAnnotator",
    "AnnotationResult",
    "AnnotationConfig",
    "AnnotationMethod",
    # Lookback Lens
    "LookbackLensAnnotator",
    "LookbackLensConfig",
    # API
    "APIAnnotator",
    "APIAnnotationConfig",
]
