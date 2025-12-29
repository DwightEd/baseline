"""Lookback Lens hallucination detection.

Implementation based on:
"Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models"
https://github.com/voidism/Lookback-Lens

Key insight: Hallucinations occur when the model attends more to recently generated
tokens rather than the context/prompt. The "lookback ratio" measures this.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Literal, Tuple
import numpy as np
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import TokenAnnotation, LabeledSample, get_logger
from .base import BaseAnnotator, AnnotationResult, AnnotationConfig, AnnotationMethod

logger = get_logger(__name__)


class LookbackLensConfig(BaseModel):
    """Configuration for Lookback Lens annotator.
    
    Based on the Lookback Lens paper's methodology:
    - Compute attention ratio between context tokens and recent tokens
    - High ratio to recent tokens indicates potential hallucination
    """
    enabled: bool = Field(default=True)
    
    # Window size for "recent" tokens (tokens generated before current)
    context_window: int = Field(
        default=10,
        ge=1,
        description="Number of recent tokens to consider as 'recent context'"
    )
    
    # Threshold for classifying as hallucination
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Lookback ratio threshold (higher = more likely hallucination)"
    )
    
    # Which layer(s) to use for attention analysis
    layer_to_use: Union[Literal["first", "last", "all", "mean"], int] = Field(
        default="first",
        description="Which layer's attention to use ('first', 'last', 'all', 'mean', or int)"
    )
    
    # Head aggregation method
    aggregate_heads: Literal["mean", "max", "median", "min"] = Field(
        default="mean",
        description="How to aggregate attention across heads"
    )
    
    # Whether to normalize attention weights
    normalize_attention: bool = Field(
        default=True,
        description="Normalize attention weights before computing ratio"
    )
    
    # Minimum confidence to report
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for annotation"
    )
    
    # Use gold labels for training data
    use_gold_for_train: bool = Field(
        default=True,
        description="Use gold labels for training split instead of prediction"
    )


class LookbackLensAnnotator(BaseAnnotator):
    """Lookback Lens hallucination annotator.
    
    Detects hallucinations by analyzing attention patterns. When the model
    attends more to its own recent outputs rather than the context/prompt,
    it's more likely to be hallucinating.
    
    Usage:
        annotator = LookbackLensAnnotator(config)
        result = annotator.annotate(
            sample=labeled_sample,
            attention_weights=attention_tensor,  # [n_layers, batch, n_heads, seq_len, seq_len]
            prompt_length=prompt_len
        )
    """
    
    def __init__(self, config: Optional[LookbackLensConfig] = None):
        self.lookback_config = config or LookbackLensConfig()
        super().__init__(AnnotationConfig(
            method=AnnotationMethod.LOOKBACK_LENS,
            threshold=self.lookback_config.threshold
        ))
    
    @property
    def method(self) -> AnnotationMethod:
        return AnnotationMethod.LOOKBACK_LENS
    
    def annotate(
        self,
        sample: LabeledSample,
        attention_weights: Optional[Any] = None,
        prompt_length: int = 0,
        token_ids: Optional[List[int]] = None,
        token_texts: Optional[List[str]] = None,
        **kwargs
    ) -> AnnotationResult:
        """Annotate a sample using Lookback Lens method.
        
        Args:
            sample: The labeled sample
            attention_weights: Attention tensor [n_layers, batch, n_heads, seq_len, seq_len]
                              or [n_layers, n_heads, seq_len, seq_len]
            prompt_length: Number of tokens in the prompt
            token_ids: Token IDs for the response
            token_texts: Decoded token texts for the response
            
        Returns:
            AnnotationResult with token-level hallucination annotations
        """
        # If no attention weights, fall back to gold labels
        if attention_weights is None:
            logger.debug(f"No attention weights for {sample.id}, using gold labels")
            return AnnotationResult.from_gold_labels(sample)
        
        try:
            import torch
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
        except ImportError:
            pass
        
        # Convert to numpy if needed
        attention_weights = np.array(attention_weights)
        
        # Handle different tensor shapes
        # Expected: [n_layers, batch, n_heads, seq_len, seq_len] or [n_layers, n_heads, seq_len, seq_len]
        if attention_weights.ndim == 5:
            attention_weights = attention_weights[:, 0, :, :, :]  # Remove batch dim
        elif attention_weights.ndim != 4:
            logger.warning(f"Unexpected attention shape: {attention_weights.shape}")
            return AnnotationResult.from_gold_labels(sample)
        
        n_layers, n_heads, seq_len, _ = attention_weights.shape
        
        # Select layer(s) to use
        layer_attention = self._select_layer(attention_weights, n_layers)
        
        # Compute lookback ratios for each response token
        response_length = seq_len - prompt_length
        if response_length <= 0:
            logger.warning(f"No response tokens for {sample.id}")
            return AnnotationResult.from_gold_labels(sample)
        
        token_annotations = []
        hallucination_count = 0
        
        for i in range(response_length):
            token_pos = prompt_length + i
            
            # Compute lookback ratio for this token
            lookback_ratio, attn_to_context, attn_to_recent = self._compute_lookback_ratio(
                layer_attention,
                token_pos,
                prompt_length,
                self.lookback_config.context_window
            )
            
            # Classify as hallucination if ratio exceeds threshold
            is_hallucination = lookback_ratio > self.lookback_config.threshold
            confidence = abs(lookback_ratio - self.lookback_config.threshold) / max(
                self.lookback_config.threshold, 1 - self.lookback_config.threshold
            )
            confidence = min(1.0, confidence)
            
            if is_hallucination:
                hallucination_count += 1
            
            # Get token info if available
            tok_id = token_ids[i] if token_ids and i < len(token_ids) else -1
            tok_text = token_texts[i] if token_texts and i < len(token_texts) else f"[token_{i}]"
            
            annotation = TokenAnnotation(
                token_id=tok_id,
                token_text=tok_text,
                position=i,
                is_hallucination=is_hallucination,
                confidence=confidence,
                lookback_ratio=float(lookback_ratio),
                attention_to_context=float(attn_to_context),
                attention_to_recent=float(attn_to_recent),
            )
            token_annotations.append(annotation)
        
        # Compute summary statistics
        hallucination_ratio = hallucination_count / response_length if response_length > 0 else 0.0
        has_hallucination = hallucination_ratio > 0
        
        # Overall confidence based on consistency of predictions
        lookback_ratios = [ann.lookback_ratio for ann in token_annotations if ann.lookback_ratio is not None]
        if lookback_ratios:
            ratio_std = np.std(lookback_ratios)
            overall_confidence = 1.0 - min(1.0, ratio_std)
        else:
            overall_confidence = 0.5
        
        return AnnotationResult(
            sample_id=sample.id,
            token_annotations=token_annotations,
            has_hallucination=has_hallucination,
            hallucination_ratio=hallucination_ratio,
            confidence=overall_confidence,
            method=AnnotationMethod.LOOKBACK_LENS,
            metadata={
                "threshold": self.lookback_config.threshold,
                "layer_used": str(self.lookback_config.layer_to_use),
                "context_window": self.lookback_config.context_window,
                "response_length": response_length,
                "prompt_length": prompt_length,
                "mean_lookback_ratio": float(np.mean(lookback_ratios)) if lookback_ratios else 0.0,
            }
        )
    
    def _select_layer(self, attention: np.ndarray, n_layers: int) -> np.ndarray:
        """Select which layer(s) to use for analysis.
        
        Args:
            attention: [n_layers, n_heads, seq_len, seq_len]
            n_layers: Number of layers
            
        Returns:
            Selected attention [n_heads, seq_len, seq_len] or aggregated
        """
        layer_spec = self.lookback_config.layer_to_use
        
        if isinstance(layer_spec, int):
            idx = layer_spec if layer_spec >= 0 else n_layers + layer_spec
            return attention[idx]
        elif layer_spec == "first":
            return attention[0]
        elif layer_spec == "last":
            return attention[-1]
        elif layer_spec == "mean":
            return np.mean(attention, axis=0)
        elif layer_spec == "all":
            # Return mean across layers
            return np.mean(attention, axis=0)
        else:
            logger.warning(f"Unknown layer_to_use: {layer_spec}, using first")
            return attention[0]
    
    def _compute_lookback_ratio(
        self,
        attention: np.ndarray,
        token_pos: int,
        prompt_length: int,
        context_window: int
    ) -> Tuple[float, float, float]:
        """Compute the lookback ratio for a single token.
        
        The lookback ratio measures how much attention goes to recently generated
        tokens vs. the original context (prompt).
        
        Args:
            attention: [n_heads, seq_len, seq_len]
            token_pos: Position of current token
            prompt_length: Length of prompt
            context_window: Size of recent context window
            
        Returns:
            (lookback_ratio, attention_to_context, attention_to_recent)
            - lookback_ratio: ratio of attention to recent vs total (higher = more hallucination risk)
            - attention_to_context: sum of attention to prompt tokens
            - attention_to_recent: sum of attention to recent generated tokens
        """
        # Get attention weights for this token position
        # attention[:, token_pos, :] gives attention FROM token_pos TO all other positions
        attn_weights = attention[:, token_pos, :token_pos + 1]  # Only look at past tokens
        
        # Aggregate across heads
        agg_method = self.lookback_config.aggregate_heads
        if agg_method == "mean":
            attn_weights = np.mean(attn_weights, axis=0)
        elif agg_method == "max":
            attn_weights = np.max(attn_weights, axis=0)
        elif agg_method == "median":
            attn_weights = np.median(attn_weights, axis=0)
        elif agg_method == "min":
            attn_weights = np.min(attn_weights, axis=0)
        else:
            attn_weights = np.mean(attn_weights, axis=0)
        
        # Normalize if requested
        if self.lookback_config.normalize_attention:
            total = np.sum(attn_weights)
            if total > 0:
                attn_weights = attn_weights / total
        
        # Split attention into context (prompt) and recent (generated)
        # Context: tokens 0 to prompt_length-1
        # Recent: tokens from (token_pos - context_window) to token_pos-1
        
        attention_to_context = np.sum(attn_weights[:prompt_length])
        
        # Recent tokens: last `context_window` tokens before current (but after prompt)
        recent_start = max(prompt_length, token_pos - context_window)
        recent_end = token_pos
        
        if recent_start < recent_end:
            attention_to_recent = np.sum(attn_weights[recent_start:recent_end])
        else:
            attention_to_recent = 0.0
        
        # Lookback ratio: proportion of attention going to recent tokens
        # Higher ratio = model is "looking back" at its own outputs more = hallucination risk
        total_attention = attention_to_context + attention_to_recent
        if total_attention > 0:
            lookback_ratio = attention_to_recent / total_attention
        else:
            lookback_ratio = 0.5  # Neutral if no attention
        
        return lookback_ratio, attention_to_context, attention_to_recent
    
    def compute_batch_ratios(
        self,
        attention_weights: np.ndarray,
        prompt_length: int,
        response_length: int
    ) -> np.ndarray:
        """Efficiently compute lookback ratios for all response tokens.
        
        Args:
            attention_weights: [n_layers, n_heads, seq_len, seq_len] or similar
            prompt_length: Length of prompt
            response_length: Length of response
            
        Returns:
            Array of lookback ratios for each response token
        """
        if attention_weights.ndim == 5:
            attention_weights = attention_weights[:, 0, :, :, :]
        
        n_layers = attention_weights.shape[0]
        layer_attention = self._select_layer(attention_weights, n_layers)
        
        ratios = []
        for i in range(response_length):
            token_pos = prompt_length + i
            ratio, _, _ = self._compute_lookback_ratio(
                layer_attention,
                token_pos,
                prompt_length,
                self.lookback_config.context_window
            )
            ratios.append(ratio)
        
        return np.array(ratios)


def create_lookback_annotator(
    threshold: float = 0.5,
    layer: Union[str, int] = "first",
    context_window: int = 10,
    **kwargs
) -> LookbackLensAnnotator:
    """Factory function to create a Lookback Lens annotator.
    
    Args:
        threshold: Hallucination threshold (0-1)
        layer: Which layer to use ("first", "last", "mean", or int)
        context_window: Recent context window size
        **kwargs: Additional config options
        
    Returns:
        Configured LookbackLensAnnotator
    """
    config = LookbackLensConfig(
        threshold=threshold,
        layer_to_use=layer,
        context_window=context_window,
        **kwargs
    )
    return LookbackLensAnnotator(config)
