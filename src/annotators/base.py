"""Base classes for hallucination annotation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import TokenAnnotation, LabeledSample, get_logger

logger = get_logger(__name__)


class AnnotationMethod(str, Enum):
    """Available hallucination annotation methods."""
    LOOKBACK_LENS = "lookback_lens"
    API_BASED = "api_based"
    RULE_BASED = "rule_based"
    GOLD_LABELS = "gold_labels"


@dataclass
class AnnotationResult:
    """Result of hallucination annotation.
    
    Contains both token-level annotations and summary statistics.
    """
    sample_id: str
    token_annotations: List[TokenAnnotation] = field(default_factory=list)
    has_hallucination: bool = False
    hallucination_ratio: float = 0.0  # Fraction of tokens marked as hallucination
    confidence: float = 1.0
    method: AnnotationMethod = AnnotationMethod.LOOKBACK_LENS
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "token_annotations": [
                {
                    "token_id": ann.token_id,
                    "token_text": ann.token_text,
                    "position": ann.position,
                    "is_hallucination": ann.is_hallucination,
                    "confidence": ann.confidence,
                    "lookback_ratio": ann.lookback_ratio,
                    "attention_to_context": ann.attention_to_context,
                    "attention_to_recent": ann.attention_to_recent,
                }
                for ann in self.token_annotations
            ],
            "has_hallucination": self.has_hallucination,
            "hallucination_ratio": self.hallucination_ratio,
            "confidence": self.confidence,
            "method": self.method.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_gold_labels(cls, sample: LabeledSample) -> "AnnotationResult":
        """Create annotation result from gold labels in dataset."""
        has_hallucination = sample.has_hallucination
        return cls(
            sample_id=sample.id,
            token_annotations=[],  # Span-based, not token-level
            has_hallucination=has_hallucination,
            hallucination_ratio=1.0 if has_hallucination else 0.0,
            confidence=1.0,
            method=AnnotationMethod.GOLD_LABELS,
            metadata={
                "hallucination_spans": [
                    {
                        "text": span.text,
                        "start": span.start,
                        "end": span.end,
                        "label_type": span.label_type.value,
                    }
                    for span in sample.hallucination_spans
                ],
                "source": "gold_labels",
            }
        )


class AnnotationConfig(BaseModel):
    """Base configuration for annotation methods."""
    method: AnnotationMethod = Field(
        default=AnnotationMethod.LOOKBACK_LENS,
        description="Annotation method to use"
    )
    use_gold_labels: bool = Field(
        default=True,
        description="Use gold labels from dataset when available"
    )
    fallback_to_gold: bool = Field(
        default=True,
        description="Fall back to gold labels if annotation fails"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for binary hallucination classification"
    )


class BaseAnnotator(ABC):
    """Abstract base class for hallucination annotators.
    
    Subclasses must implement:
    - annotate(): Annotate a single sample
    - annotate_batch(): Annotate multiple samples (optional, default uses annotate)
    """
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        self.config = config or AnnotationConfig()
    
    @property
    @abstractmethod
    def method(self) -> AnnotationMethod:
        """Return the annotation method used by this annotator."""
        pass
    
    @abstractmethod
    def annotate(
        self,
        sample: LabeledSample,
        attention_weights: Optional[Any] = None,
        prompt_length: int = 0,
        **kwargs
    ) -> AnnotationResult:
        """Annotate a single sample for hallucinations.
        
        Args:
            sample: The labeled sample to annotate
            attention_weights: Optional attention weights from model
            prompt_length: Length of prompt in tokens
            **kwargs: Additional arguments
            
        Returns:
            AnnotationResult with token-level annotations
        """
        pass
    
    def annotate_batch(
        self,
        samples: List[LabeledSample],
        attention_weights_list: Optional[List[Any]] = None,
        prompt_lengths: Optional[List[int]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[AnnotationResult]:
        """Annotate multiple samples.
        
        Default implementation calls annotate() for each sample.
        Subclasses can override for batch-optimized implementations.
        
        Args:
            samples: List of samples to annotate
            attention_weights_list: Optional list of attention weights
            prompt_lengths: List of prompt lengths
            show_progress: Show progress bar
            **kwargs: Additional arguments
            
        Returns:
            List of AnnotationResults
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Annotating", disable=not show_progress)
        except ImportError:
            iterator = samples
            if show_progress:
                logger.warning("tqdm not installed, progress bar disabled")
        
        results = []
        for i, sample in enumerate(iterator):
            attn = attention_weights_list[i] if attention_weights_list else None
            prompt_len = prompt_lengths[i] if prompt_lengths else 0
            
            try:
                result = self.annotate(
                    sample=sample,
                    attention_weights=attn,
                    prompt_length=prompt_len,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Annotation failed for {sample.id}: {e}")
                # Fall back to gold labels if available
                if self.config.fallback_to_gold:
                    result = AnnotationResult.from_gold_labels(sample)
                else:
                    result = AnnotationResult(
                        sample_id=sample.id,
                        has_hallucination=False,
                        method=self.method,
                        metadata={"error": str(e)}
                    )
            
            results.append(result)
        
        return results
