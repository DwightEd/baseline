"""Hallucination detection methods."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import pickle
import json

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    BaseComponent, METHOD_REGISTRY,
    ExtractedFeatures, TrainingConfig,
    MethodError, MethodNotFittedError,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class Prediction:
    """Detection prediction result."""
    score: float
    label: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class MethodMetrics:
    """Training/evaluation metrics for a method."""
    auroc: float = 0.0
    auprc: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    threshold: float = 0.5
    cv_scores: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        result = {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
        }
        if self.cv_scores:
            result["cv_mean"] = float(np.mean(self.cv_scores))
            result["cv_std"] = float(np.std(self.cv_scores))
        return result


class BaseMethod(BaseComponent, ABC):
    """Abstract base class for detection methods."""
    
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._is_fitted = False
        self._feature_dim: Optional[int] = None
    
    @classmethod
    @abstractmethod
    def component_name(cls) -> str:
        pass
    
    @abstractmethod
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        """Train the detection method."""
        pass
    
    @abstractmethod
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """Predict for a single sample."""
        pass
    
    def predict_batch(self, features: List[ExtractedFeatures]) -> List[Prediction]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]
    
    def predict_proba(self, features: List[ExtractedFeatures]) -> np.ndarray:
        """Get probability predictions."""
        predictions = self.predict_batch(features)
        return np.array([p.score for p in predictions])
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise MethodNotFittedError(f"{self.component_name()} not fitted. Call fit() first.")
    
    def _extract_feature_vector(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract flat feature vector from ExtractedFeatures."""
        vectors = []
        
        if features.attention_features:
            attn = features.attention_features
            if attn.get("eigenvalues") is not None:
                eigs = attn["eigenvalues"]
                if isinstance(eigs, torch.Tensor):
                    eigs = eigs.numpy()
                vectors.append(eigs.flatten())
            
            if attn.get("laplacian_eigenvalues") is not None:
                lap_eigs = attn["laplacian_eigenvalues"]
                if isinstance(lap_eigs, torch.Tensor):
                    lap_eigs = lap_eigs.numpy()
                vectors.append(lap_eigs.flatten())
            
            if attn.get("row_entropy") is not None:
                entropy = attn["row_entropy"]
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.numpy()
                vectors.append(entropy.flatten())
        
        if features.token_prob_features:
            probs = features.token_prob_features
            prob_features = []
            
            if probs.get("mean_entropy") is not None:
                prob_features.append(probs["mean_entropy"])
            if probs.get("max_entropy") is not None:
                prob_features.append(probs["max_entropy"])
            if probs.get("perplexity") is not None:
                prob_features.append(probs["perplexity"])
            
            if probs.get("entropy") is not None:
                entropy = probs["entropy"]
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.numpy()
                prob_features.extend([entropy.mean(), entropy.std(), entropy.max(), entropy.min()])
            
            if prob_features:
                vectors.append(np.array(prob_features))
        
        if features.hidden_state_features:
            hidden = features.hidden_state_features
            if hidden.get("layer_norms") is not None:
                norms = hidden["layer_norms"]
                if isinstance(norms, torch.Tensor):
                    norms = norms.numpy()
                vectors.append(norms.flatten()[:32])
            
            if hidden.get("pooled") is not None:
                pooled = hidden["pooled"]
                if isinstance(pooled, torch.Tensor):
                    pooled = pooled.numpy()
                vectors.append(pooled.flatten()[:64])
        
        if not vectors:
            return np.zeros(1)
        
        return np.concatenate(vectors).astype(np.float32)
    
    def _compute_metrics(
        self,
        predictions: List[Prediction],
        labels: List[int]
    ) -> MethodMetrics:
        """Compute evaluation metrics."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            accuracy_score, f1_score, precision_score, recall_score
        )
        
        scores = np.array([p.score for p in predictions])
        pred_labels = np.array([p.label for p in predictions])
        true_labels = np.array(labels)
        
        metrics = MethodMetrics()
        
        if len(set(true_labels)) > 1:
            metrics.auroc = float(roc_auc_score(true_labels, scores))
            metrics.auprc = float(average_precision_score(true_labels, scores))
        
        metrics.accuracy = float(accuracy_score(true_labels, pred_labels))
        metrics.f1 = float(f1_score(true_labels, pred_labels, zero_division=0))
        metrics.precision = float(precision_score(true_labels, pred_labels, zero_division=0))
        metrics.recall = float(recall_score(true_labels, pred_labels, zero_division=0))
        
        return metrics


@METHOD_REGISTRY.register("lapeigvals", aliases=["laplacian_eigenvalues", "spectral"])
class LapEigvalsMethod(BaseMethod):
    """Detection using Laplacian eigenvalues of attention graphs."""
    
    @classmethod
    def component_name(cls) -> str:
        return "lapeigvals"
    
    def __init__(
        self,
        n_eigenvalues: int = 10,
        use_cv: bool = True,
        cv_folds: int = 5,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.n_eigenvalues = n_eigenvalues
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.classifier = None
    
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        X = np.array([self._extract_laplacian_features(f) for f in features])
        y = np.array(labels)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        self._feature_dim = X.shape[1]
        
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        
        metrics = MethodMetrics()
        if self.use_cv and len(set(y)) > 1:
            try:
                cv_scores = cross_val_score(
                    self.classifier, X, y,
                    cv=min(self.cv_folds, len(y) // 2),
                    scoring='roc_auc'
                )
                metrics.cv_scores = cv_scores.tolist()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        self.classifier.fit(X, y)
        self._is_fitted = True
        
        if val_features and val_labels:
            val_preds = self.predict_batch(val_features)
            metrics = self._compute_metrics(val_preds, val_labels)
        
        logger.info(f"LapEigvals fitted with {X.shape[1]} features")
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        self._check_fitted()
        
        X = self._extract_laplacian_features(features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        proba = self.classifier.predict_proba(X)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        label = int(score >= 0.5)
        
        return Prediction(
            score=score,
            label=label,
            confidence=float(max(proba)),
        )
    
    def _extract_laplacian_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract Laplacian eigenvalue features."""
        vectors = []
        
        if features.attention_features:
            attn = features.attention_features
            
            if attn.get("laplacian_eigenvalues") is not None:
                lap_eigs = attn["laplacian_eigenvalues"]
                if isinstance(lap_eigs, torch.Tensor):
                    lap_eigs = lap_eigs.numpy()
                vectors.append(lap_eigs.flatten())
            
            if attn.get("eigenvalues") is not None:
                eigs = attn["eigenvalues"]
                if isinstance(eigs, torch.Tensor):
                    eigs = eigs.numpy()
                vectors.append(eigs.flatten())
        
        if not vectors:
            return np.zeros(self.n_eigenvalues * 32)
        
        return np.concatenate(vectors).astype(np.float32)
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump({
                "scaler": self.scaler,
                "classifier": self.classifier,
                "n_eigenvalues": self.n_eigenvalues,
                "feature_dim": self._feature_dim,
            }, f)
    
    def load(self, path: Path) -> None:
        with open(Path(path) / "model.pkl", 'rb') as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.classifier = data["classifier"]
        self.n_eigenvalues = data["n_eigenvalues"]
        self._feature_dim = data.get("feature_dim")
        self._is_fitted = True


@METHOD_REGISTRY.register("entropy", aliases=["token_entropy"])
class EntropyMethod(BaseMethod):
    """Detection using token entropy."""
    
    @classmethod
    def component_name(cls) -> str:
        return "entropy"
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()
        self.classifier = None
    
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        X = np.array([self._extract_entropy(f) for f in features])
        y = np.array(labels)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X, y)
        self._is_fitted = True
        
        metrics = MethodMetrics()
        if val_features and val_labels:
            metrics = self._compute_metrics(self.predict_batch(val_features), val_labels)
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        self._check_fitted()
        
        X = self._extract_entropy(features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        proba = self.classifier.predict_proba(X)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            score=score,
            label=int(score >= 0.5),
            confidence=float(max(proba))
        )
    
    def _extract_entropy(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract entropy-based features."""
        feature_list = []
        
        if features.token_prob_features:
            probs = features.token_prob_features
            
            if probs.get("mean_entropy") is not None:
                feature_list.append(probs["mean_entropy"])
            if probs.get("max_entropy") is not None:
                feature_list.append(probs["max_entropy"])
            if probs.get("perplexity") is not None:
                feature_list.append(probs["perplexity"])
            
            if probs.get("entropy") is not None:
                entropy = probs["entropy"]
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.numpy()
                feature_list.extend([
                    float(entropy.mean()),
                    float(entropy.std()),
                    float(entropy.max()),
                    float(entropy.min()),
                    float(np.percentile(entropy, 75)),
                    float(np.percentile(entropy, 25)),
                ])
        
        return np.array(feature_list) if feature_list else np.zeros(6)
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump({"scaler": self.scaler, "classifier": self.classifier}, f)
    
    def load(self, path: Path) -> None:
        with open(Path(path) / "model.pkl", 'rb') as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.classifier = data["classifier"]
        self._is_fitted = True


@METHOD_REGISTRY.register("perplexity")
class PerplexityMethod(BaseMethod):
    """Detection using sequence perplexity thresholding."""
    
    @classmethod
    def component_name(cls) -> str:
        return "perplexity"
    
    def __init__(self, threshold: Optional[float] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.pos_mean = 0.0
        self.neg_mean = 0.0
    
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        perplexities = [self._get_perplexity(f) for f in features]
        
        pos_perp = [p for p, l in zip(perplexities, labels) if l == 1]
        neg_perp = [p for p, l in zip(perplexities, labels) if l == 0]
        
        self.pos_mean = np.mean(pos_perp) if pos_perp else 10.0
        self.neg_mean = np.mean(neg_perp) if neg_perp else 5.0
        
        if self.threshold is None:
            self.threshold = (self.pos_mean + self.neg_mean) / 2
        
        self._is_fitted = True
        
        metrics = MethodMetrics(threshold=self.threshold)
        if val_features and val_labels:
            metrics = self._compute_metrics(self.predict_batch(val_features), val_labels)
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        self._check_fitted()
        
        perplexity = self._get_perplexity(features)
        
        diff_range = max(abs(self.pos_mean - self.neg_mean), 1.0)
        normalized = (perplexity - self.neg_mean) / diff_range
        score = float(np.clip(normalized, 0, 1))
        
        label = int(perplexity > self.threshold)
        confidence = abs(perplexity - self.threshold) / diff_range
        
        return Prediction(
            score=score,
            label=label,
            confidence=float(np.clip(confidence, 0, 1)),
            metadata={"perplexity": perplexity}
        )
    
    def _get_perplexity(self, features: ExtractedFeatures) -> float:
        if features.token_prob_features and features.token_prob_features.get("perplexity"):
            return float(features.token_prob_features["perplexity"])
        return 0.0
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", 'w') as f:
            json.dump({
                "threshold": self.threshold,
                "pos_mean": self.pos_mean,
                "neg_mean": self.neg_mean,
            }, f)
    
    def load(self, path: Path) -> None:
        with open(Path(path) / "config.json", 'r') as f:
            data = json.load(f)
        self.threshold = data["threshold"]
        self.pos_mean = data["pos_mean"]
        self.neg_mean = data["neg_mean"]
        self._is_fitted = True


@METHOD_REGISTRY.register("random_forest", aliases=["rf"])
class RandomForestMethod(BaseMethod):
    """Detection using Random Forest classifier."""
    
    @classmethod
    def component_name(cls) -> str:
        return "random_forest"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.classifier = None
    
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        X = np.array([self._extract_feature_vector(f) for f in features])
        y = np.array(labels)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X, y)
        self._is_fitted = True
        
        metrics = MethodMetrics()
        if val_features and val_labels:
            metrics = self._compute_metrics(self.predict_batch(val_features), val_labels)
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        self._check_fitted()
        
        X = self._extract_feature_vector(features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        proba = self.classifier.predict_proba(X)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            score=score,
            label=int(score >= 0.5),
            confidence=float(max(proba))
        )
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump({
                "scaler": self.scaler,
                "classifier": self.classifier,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
            }, f)
    
    def load(self, path: Path) -> None:
        with open(Path(path) / "model.pkl", 'rb') as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.classifier = data["classifier"]
        self.n_estimators = data["n_estimators"]
        self.max_depth = data["max_depth"]
        self._is_fitted = True


@METHOD_REGISTRY.register("ensemble")
class EnsembleMethod(BaseMethod):
    """Ensemble of multiple detection methods."""
    
    @classmethod
    def component_name(cls) -> str:
        return "ensemble"
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        voting: str = "soft",
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.method_names = methods or ["lapeigvals", "entropy", "perplexity"]
        self.weights = weights
        self.voting = voting
        self.methods: List[BaseMethod] = []
    
    def fit(
        self,
        features: List[ExtractedFeatures],
        labels: List[int],
        val_features: Optional[List[ExtractedFeatures]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> MethodMetrics:
        self.methods = []
        method_metrics = []
        
        for name in self.method_names:
            logger.info(f"Training ensemble member: {name}")
            method = METHOD_REGISTRY.create(name)
            metrics = method.fit(features, labels, val_features, val_labels)
            self.methods.append(method)
            method_metrics.append(metrics.auroc if metrics.auroc > 0 else 0.5)
        
        if self.weights is None:
            total = sum(method_metrics)
            self.weights = [m / total for m in method_metrics] if total > 0 else [1.0 / len(self.methods)] * len(self.methods)
        
        self._is_fitted = True
        
        metrics = MethodMetrics()
        if val_features and val_labels:
            metrics = self._compute_metrics(self.predict_batch(val_features), val_labels)
        
        logger.info(f"Ensemble fitted with weights: {self.weights}")
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        self._check_fitted()
        
        predictions = [m.predict(features) for m in self.methods]
        
        if self.voting == "soft":
            weighted_score = sum(p.score * w for p, w in zip(predictions, self.weights))
        else:
            votes = [p.label * w for p, w in zip(predictions, self.weights)]
            weighted_score = sum(votes) / sum(self.weights)
        
        return Prediction(
            score=float(weighted_score),
            label=int(weighted_score >= 0.5),
            confidence=float(1.0 - abs(weighted_score - 0.5) * 2),
            metadata={
                "sub_predictions": [p.score for p in predictions],
                "weights": self.weights
            }
        )
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, method in enumerate(self.methods):
            method.save(path / f"method_{i}")
        
        with open(path / "ensemble_config.json", 'w') as f:
            json.dump({
                "method_names": self.method_names,
                "weights": self.weights,
                "voting": self.voting,
            }, f)
    
    def load(self, path: Path) -> None:
        path = Path(path)
        
        with open(path / "ensemble_config.json", 'r') as f:
            data = json.load(f)
        
        self.method_names = data["method_names"]
        self.weights = data["weights"]
        self.voting = data["voting"]
        
        self.methods = []
        for i, name in enumerate(self.method_names):
            method = METHOD_REGISTRY.create(name)
            method.load(path / f"method_{i}")
            self.methods.append(method)
        
        self._is_fitted = True


def create_method(name: str, **kwargs: Any) -> BaseMethod:
    """Create detection method by name."""
    return METHOD_REGISTRY.create(name, **kwargs)


def list_methods() -> List[str]:
    """List all available methods."""
    return METHOD_REGISTRY.list_registered()
