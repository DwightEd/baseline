"""Evaluation metrics and utilities."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import get_logger, EvaluationConfig

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    auroc: float = 0.0
    auprc: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    threshold: float = 0.5
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    pr_curve: Optional[Dict[str, List[float]]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
            "confidence_intervals": self.confidence_intervals,
            "per_class_metrics": self.per_class_metrics,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'EvaluationResult':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"AUROC:      {self.auroc:.4f}",
            f"AUPRC:      {self.auprc:.4f}",
            f"Accuracy:   {self.accuracy:.4f}",
            f"F1 Score:   {self.f1:.4f}",
            f"Precision:  {self.precision:.4f}",
            f"Recall:     {self.recall:.4f}",
            f"Specificity:{self.specificity:.4f}",
            f"Threshold:  {self.threshold:.4f}",
        ]
        
        if self.confusion_matrix:
            lines.append("-" * 50)
            lines.append("Confusion Matrix:")
            lines.append(f"  TN={self.confusion_matrix[0][0]}, FP={self.confusion_matrix[0][1]}")
            lines.append(f"  FN={self.confusion_matrix[1][0]}, TP={self.confusion_matrix[1][1]}")
        
        if self.confidence_intervals:
            lines.append("-" * 50)
            lines.append("Confidence Intervals (95%):")
            for metric, (low, high) in self.confidence_intervals.items():
                lines.append(f"  {metric}: [{low:.4f}, {high:.4f}]")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class Evaluator:
    """Evaluates detection predictions against ground truth."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        predictions: List[Any],
        labels: List[int],
    ) -> EvaluationResult:
        """Evaluate predictions against labels."""
        scores = np.array([p.score if hasattr(p, 'score') else p for p in predictions])
        pred_labels = np.array([
            p.label if hasattr(p, 'label') else int(p >= self.config.threshold)
            for p in predictions
        ])
        true_labels = np.array(labels)
        
        result = EvaluationResult(
            threshold=self.config.threshold,
            metadata={"n_samples": len(predictions)}
        )
        
        if len(set(true_labels)) < 2:
            logger.warning("Only one class present in labels, some metrics unavailable")
            result.accuracy = float(accuracy_score(true_labels, pred_labels))
            result.metadata["warning"] = "single_class"
            return result
        
        result.auroc = float(roc_auc_score(true_labels, scores))
        result.auprc = float(average_precision_score(true_labels, scores))
        result.accuracy = float(accuracy_score(true_labels, pred_labels))
        result.f1 = float(f1_score(true_labels, pred_labels, zero_division=0))
        result.precision = float(precision_score(true_labels, pred_labels, zero_division=0))
        result.recall = float(recall_score(true_labels, pred_labels, zero_division=0))
        
        cm = confusion_matrix(true_labels, pred_labels)
        result.confusion_matrix = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        fpr, tpr, thresholds = roc_curve(true_labels, scores)
        result.roc_curve = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
        
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(true_labels, scores)
        result.pr_curve = {
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
            "thresholds": pr_thresholds.tolist()
        }
        
        if self.config.bootstrap_samples > 0:
            result.confidence_intervals = self._bootstrap_ci(
                scores, true_labels,
                self.config.bootstrap_samples,
                self.config.confidence_level
            )
        
        result.per_class_metrics = self._compute_per_class_metrics(true_labels, pred_labels, scores)
        
        return result
    
    def _bootstrap_ci(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_samples: int,
        confidence: float
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for key metrics."""
        n = len(scores)
        
        aurocs = []
        auprcs = []
        
        rng = np.random.default_rng(42)
        
        for _ in range(n_samples):
            idx = rng.choice(n, size=n, replace=True)
            
            if len(set(labels[idx])) < 2:
                continue
            
            try:
                aurocs.append(roc_auc_score(labels[idx], scores[idx]))
                auprcs.append(average_precision_score(labels[idx], scores[idx]))
            except Exception:
                continue
        
        result = {}
        alpha = (1 - confidence) / 2
        
        if aurocs:
            result["auroc"] = (
                float(np.percentile(aurocs, alpha * 100)),
                float(np.percentile(aurocs, (1 - alpha) * 100))
            )
        
        if auprcs:
            result["auprc"] = (
                float(np.percentile(auprcs, alpha * 100)),
                float(np.percentile(auprcs, (1 - alpha) * 100))
            )
        
        return result
    
    def _compute_per_class_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
        result = {}
        
        for label in sorted(set(true_labels)):
            mask = true_labels == label
            
            if mask.sum() == 0:
                continue
            
            correct = (pred_labels[mask] == label).sum()
            total = mask.sum()
            
            result[f"class_{label}"] = {
                "count": int(total),
                "accuracy": float(correct / total),
                "mean_score": float(scores[mask].mean()),
                "std_score": float(scores[mask].std()),
            }
        
        return result
    
    def find_optimal_threshold(
        self,
        predictions: List[Any],
        labels: List[int],
        metric: str = "f1",
        search_range: Tuple[float, float] = (0.1, 0.9),
        n_steps: int = 17
    ) -> Tuple[float, float]:
        """Find optimal classification threshold."""
        scores = np.array([p.score if hasattr(p, 'score') else p for p in predictions])
        true_labels = np.array(labels)
        
        best_threshold = 0.5
        best_metric_value = 0.0
        
        for threshold in np.linspace(search_range[0], search_range[1], n_steps):
            pred_labels = (scores >= threshold).astype(int)
            
            if metric == "f1":
                value = f1_score(true_labels, pred_labels, zero_division=0)
            elif metric == "accuracy":
                value = accuracy_score(true_labels, pred_labels)
            elif metric == "precision":
                value = precision_score(true_labels, pred_labels, zero_division=0)
            elif metric == "recall":
                value = recall_score(true_labels, pred_labels, zero_division=0)
            elif metric == "youden":
                tn = ((1 - true_labels) * (1 - pred_labels)).sum()
                fp = ((1 - true_labels) * pred_labels).sum()
                fn = (true_labels * (1 - pred_labels)).sum()
                tp = (true_labels * pred_labels).sum()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                value = sensitivity + specificity - 1
            else:
                value = f1_score(true_labels, pred_labels, zero_division=0)
            
            if value > best_metric_value:
                best_metric_value = value
                best_threshold = threshold
        
        return float(best_threshold), float(best_metric_value)
    
    def compare_methods(
        self,
        method_predictions: Dict[str, List[Any]],
        labels: List[int]
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple methods."""
        results = {}
        
        for name, predictions in method_predictions.items():
            results[name] = self.evaluate(predictions, labels)
        
        return results


def evaluate_predictions(
    predictions: List[Any],
    labels: List[int],
    config: Optional[EvaluationConfig] = None
) -> EvaluationResult:
    """Convenience function for evaluation."""
    evaluator = Evaluator(config)
    return evaluator.evaluate(predictions, labels)


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Quick AUROC computation."""
    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_f1(predictions: List[int], labels: List[int]) -> float:
    """Quick F1 computation."""
    return float(f1_score(labels, predictions, zero_division=0))
