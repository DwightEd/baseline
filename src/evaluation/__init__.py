"""Evaluation module for hallucination detection."""
from .metrics import (
    EvaluationResult,
    Evaluator,
    evaluate_predictions,
    compute_auroc,
    compute_f1,
)

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "evaluate_predictions",
    "compute_auroc",
    "compute_f1",
]
