"""Feature extraction from LLM intermediate states."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Iterator, Tuple
from dataclasses import dataclass, field
import json

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    BaseComponent, FEATURE_EXTRACTOR_REGISTRY,
    DataSample, ExtractedFeatures,
    FeatureExtractionConfig, ExtractionMode,
    AttentionConfig, HiddenStateConfig, TokenProbConfig,
    FeatureExtractionError,
    FeatureCache, MetadataCache, compute_hash,
    get_logger, ProgressLogger
)

logger = get_logger(__name__)


@dataclass
class AttentionFeatures:
    """Extracted attention features."""
    weights: Optional[torch.Tensor] = None
    eigenvalues: Optional[torch.Tensor] = None
    laplacian_eigenvalues: Optional[torch.Tensor] = None
    row_entropy: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"metadata": self.metadata}
        if self.weights is not None:
            result["weights_shape"] = list(self.weights.shape)
        if self.eigenvalues is not None:
            result["eigenvalues"] = self.eigenvalues
        if self.laplacian_eigenvalues is not None:
            result["laplacian_eigenvalues"] = self.laplacian_eigenvalues
        if self.row_entropy is not None:
            result["row_entropy"] = self.row_entropy
        return result


@dataclass
class HiddenStateFeatures:
    """Extracted hidden state features."""
    states: Optional[torch.Tensor] = None
    pca_reduced: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None
    layer_norms: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"metadata": self.metadata}
        if self.states is not None:
            result["states_shape"] = list(self.states.shape)
        if self.pca_reduced is not None:
            result["pca_reduced"] = self.pca_reduced
        if self.pooled is not None:
            result["pooled"] = self.pooled
        if self.layer_norms is not None:
            result["layer_norms"] = self.layer_norms
        return result


@dataclass
class TokenProbFeatures:
    """Extracted token probability features."""
    probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    perplexity: Optional[float] = None
    top_k_probs: Optional[torch.Tensor] = None
    mean_entropy: Optional[float] = None
    max_entropy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"metadata": self.metadata}
        if self.probs is not None:
            result["probs"] = self.probs
        if self.entropy is not None:
            result["entropy"] = self.entropy
        if self.perplexity is not None:
            result["perplexity"] = self.perplexity
        if self.top_k_probs is not None:
            result["top_k_probs"] = self.top_k_probs
        if self.mean_entropy is not None:
            result["mean_entropy"] = self.mean_entropy
        if self.max_entropy is not None:
            result["max_entropy"] = self.max_entropy
        return result


class BaseFeatureExtractor(BaseComponent, ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
    
    @classmethod
    @abstractmethod
    def component_name(cls) -> str:
        pass
    
    @abstractmethod
    def extract(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        """Extract features from a single sample."""
        pass
    
    def extract_batch(
        self,
        model: Any,
        samples: List[DataSample],
        cache: Optional[FeatureCache] = None,
        show_progress: bool = True,
    ) -> Iterator[ExtractedFeatures]:
        """Extract features from batch of samples with optional caching."""
        progress = ProgressLogger(logger, len(samples), "Feature extraction") if show_progress else None
        
        for sample in samples:
            cache_key = None
            if cache:
                cache_key = f"{sample.id}_{self.config.mode.value}"
                cached = cache.get(cache_key)
                if cached:
                    if progress:
                        progress.update()
                    yield cached
                    continue
            
            try:
                features = self.extract(model, sample)
                if cache and cache_key:
                    cache.set(cache_key, features)
                yield features
            except Exception as e:
                logger.error(f"Failed to extract features for {sample.id}: {e}")
                yield ExtractedFeatures(
                    sample_id=sample.id,
                    prompt_length=0,
                    response_length=0,
                    total_length=0,
                    layers_extracted=[],
                    metadata={"error": str(e)}
                )
            
            if progress:
                progress.update()
        
        if progress:
            progress.finish()


@FEATURE_EXTRACTOR_REGISTRY.register("default", aliases=["standard", "full"])
class StandardFeatureExtractor(BaseFeatureExtractor):
    """Standard feature extractor for attention, hidden states, and token probabilities."""
    
    @classmethod
    def component_name(cls) -> str:
        return "default"
    
    def extract(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        """Extract features based on extraction mode."""
        if self.config.mode == ExtractionMode.TEACHER_FORCING:
            return self._extract_teacher_forcing(model, sample)
        else:
            return self._extract_generation(model, sample)
    
    def _extract_teacher_forcing(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        """Extract features using teacher forcing (prompt + answer concatenated)."""
        prompt = sample.question
        if sample.context:
            prompt = f"{sample.context}\n\n{prompt}"
        
        response = sample.answer or sample.gold_answer
        full_text = prompt + response
        
        prompt_tokens = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        full_tokens = model.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        total_length = full_tokens.input_ids.shape[1]
        response_length = total_length - prompt_length
        
        if response_length <= 0:
            return ExtractedFeatures(
                sample_id=sample.id,
                prompt_length=prompt_length,
                response_length=0,
                total_length=total_length,
                layers_extracted=[],
                metadata={"mode": self.config.mode.value, "warning": "no_response_tokens"}
            )
        
        outputs = model.forward(
            input_ids=full_tokens.input_ids,
            attention_mask=full_tokens.attention_mask,
            output_attentions=self.config.attention.enabled,
            output_hidden_states=self.config.hidden_states.enabled,
        )
        
        attention_features = None
        if self.config.attention.enabled and outputs.get("attentions"):
            attention_features = self._process_attention(
                outputs["attentions"],
                prompt_length,
                total_length,
                self.config.attention
            )
        
        hidden_features = None
        if self.config.hidden_states.enabled and outputs.get("hidden_states"):
            hidden_features = self._process_hidden_states(
                outputs["hidden_states"],
                prompt_length,
                total_length,
                self.config.hidden_states
            )
        
        prob_features = None
        if self.config.token_probs.enabled:
            prob_features = self._process_token_probs(
                outputs["logits"],
                full_tokens.input_ids,
                prompt_length,
                self.config.token_probs
            )
        
        # Get layers extracted from attention features metadata
        layers_extracted = []
        if attention_features and attention_features.metadata:
            layers_extracted = attention_features.metadata.get("layers", [])
        
        return ExtractedFeatures(
            sample_id=sample.id,
            prompt_length=prompt_length,
            response_length=response_length,
            total_length=total_length,
            attention_features=attention_features.to_dict() if attention_features else None,
            hidden_state_features=hidden_features.to_dict() if hidden_features else None,
            token_prob_features=prob_features.to_dict() if prob_features else None,
            layers_extracted=layers_extracted,
            metadata={
                "mode": self.config.mode.value,
                "model": model.config.name,
                "label": sample.label,
                "attention_layer_config": str(self.config.attention.layers),
            }
        )
    
    def _extract_generation(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        """Extract features during generation (not teacher forcing)."""
        prompt = sample.question
        if sample.context:
            prompt = f"{sample.context}\n\n{prompt}"
        
        prompt_tokens = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.model.generate(
                **prompt_tokens.to(model.device),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        total_length = outputs.sequences.shape[1]
        response_length = total_length - prompt_length
        
        generated_text = model.tokenizer.decode(
            outputs.sequences[0, prompt_length:],
            skip_special_tokens=True
        )
        
        return ExtractedFeatures(
            sample_id=sample.id,
            prompt_length=prompt_length,
            response_length=response_length,
            total_length=total_length,
            layers_extracted=[],  # Generation mode doesn't extract specific layers
            metadata={
                "mode": self.config.mode.value,
                "model": model.config.name,
                "generated_text": generated_text,
                "label": sample.label,
            }
        )
    
    def _process_attention(
        self,
        attentions: Tuple[torch.Tensor, ...],
        prompt_length: int,
        total_length: int,
        config: AttentionConfig
    ) -> AttentionFeatures:
        """Process attention weights to extract features.
        
        Uses AttentionConfig.get_layer_indices() for flexible layer selection.
        """
        n_layers = len(attentions)
        # Use the new get_layer_indices method for flexible layer selection
        layer_indices = config.get_layer_indices(n_layers)
        
        response_attentions = []
        for layer_idx in layer_indices:
            layer_attn = attentions[layer_idx]
            response_attn = layer_attn[:, :, prompt_length:, :].cpu()
            response_attentions.append(response_attn)
        
        if not response_attentions:
            return AttentionFeatures(metadata={"warning": "no_attention_layers"})
        
        stacked = torch.stack(response_attentions)
        
        eigenvalues = None
        if config.compute_eigenvalues:
            eigenvalues = self._compute_eigenvalues(stacked, config.n_eigenvalues)
        
        laplacian_eigenvalues = None
        if config.compute_laplacian:
            laplacian_eigenvalues = self._compute_laplacian_eigenvalues(
                stacked, config.n_eigenvalues, config.normalization
            )
        
        row_entropy = self._compute_attention_entropy(stacked)
        
        # Store raw weights if requested
        weights_to_store = stacked if config.save_raw_weights else None
        
        return AttentionFeatures(
            weights=weights_to_store,
            eigenvalues=eigenvalues,
            laplacian_eigenvalues=laplacian_eigenvalues,
            row_entropy=row_entropy,
            metadata={
                "layers": layer_indices,
                "layer_selection_mode": str(config.layers),
                "response_tokens": total_length - prompt_length,
                "n_heads": stacked.shape[2] if stacked.dim() >= 3 else 0,
                "save_raw_weights": config.save_raw_weights,
            }
        )
    
    def _compute_eigenvalues(
        self,
        attention: torch.Tensor,
        n_eigenvalues: int
    ) -> torch.Tensor:
        """Compute eigenvalues of attention matrices."""
        n_layers = attention.shape[0]
        n_heads = attention.shape[2]
        
        all_eigenvalues = []
        
        for layer in range(n_layers):
            layer_eigs = []
            for head in range(n_heads):
                attn_matrix = attention[layer, 0, head]
                if attn_matrix.shape[0] > 0 and attn_matrix.shape[1] > 0:
                    try:
                        gram = attn_matrix @ attn_matrix.T
                        eigs = torch.linalg.eigvalsh(gram)
                        eigs = torch.sort(eigs, descending=True)[0][:n_eigenvalues]
                        
                        if len(eigs) < n_eigenvalues:
                            padding = torch.zeros(n_eigenvalues - len(eigs))
                            eigs = torch.cat([eigs, padding])
                        
                        layer_eigs.append(eigs)
                    except Exception:
                        layer_eigs.append(torch.zeros(n_eigenvalues))
                else:
                    layer_eigs.append(torch.zeros(n_eigenvalues))
            
            all_eigenvalues.append(torch.stack(layer_eigs))
        
        return torch.stack(all_eigenvalues)
    
    def _compute_laplacian_eigenvalues(
        self,
        attention: torch.Tensor,
        n_eigenvalues: int,
        normalization: str
    ) -> torch.Tensor:
        """Compute Laplacian eigenvalues for spectral analysis."""
        n_layers = attention.shape[0]
        n_heads = attention.shape[2]
        
        all_laplacian_eigs = []
        
        for layer in range(n_layers):
            layer_eigs = []
            for head in range(n_heads):
                attn_matrix = attention[layer, 0, head]
                if attn_matrix.shape[0] > 1:
                    try:
                        A = (attn_matrix + attn_matrix.T) / 2
                        D = torch.diag(A.sum(dim=1))
                        L = D - A
                        
                        if normalization == "symmetric":
                            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-10))
                            L = D_inv_sqrt @ L @ D_inv_sqrt
                        elif normalization == "random_walk":
                            D_inv = torch.diag(1.0 / (D.diag() + 1e-10))
                            L = D_inv @ L
                        
                        eigs = torch.linalg.eigvalsh(L)
                        eigs = torch.sort(eigs)[0][:n_eigenvalues]
                        
                        if len(eigs) < n_eigenvalues:
                            padding = torch.zeros(n_eigenvalues - len(eigs))
                            eigs = torch.cat([eigs, padding])
                        
                        layer_eigs.append(eigs)
                    except Exception:
                        layer_eigs.append(torch.zeros(n_eigenvalues))
                else:
                    layer_eigs.append(torch.zeros(n_eigenvalues))
            
            all_laplacian_eigs.append(torch.stack(layer_eigs))
        
        return torch.stack(all_laplacian_eigs)
    
    def _compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions."""
        attn_clamped = torch.clamp(attention, min=1e-10)
        entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)
        return entropy.mean(dim=(0, 1))
    
    def _process_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        prompt_length: int,
        total_length: int,
        config: HiddenStateConfig
    ) -> HiddenStateFeatures:
        """Process hidden states to extract features."""
        if config.layers == "all":
            states = list(hidden_states)
        elif config.layers == "last_n":
            states = list(hidden_states[-config.last_n:])
        else:
            states = [hidden_states[i] for i in config.layers if i < len(hidden_states)]
        
        if not states:
            return HiddenStateFeatures(metadata={"warning": "no_hidden_states"})
        
        response_states = [s[:, prompt_length:, :].cpu() for s in states]
        stacked = torch.stack(response_states)
        
        if config.pooling == "mean":
            pooled = stacked.mean(dim=2)
        elif config.pooling == "max":
            pooled = stacked.max(dim=2)[0]
        elif config.pooling == "last":
            pooled = stacked[:, :, -1, :]
        else:
            pooled = stacked[:, :, 0, :]
        
        layer_norms = torch.norm(pooled, dim=-1)
        
        pca_reduced = None
        if config.compute_pca and pooled.shape[-1] > config.pca_components:
            try:
                flat = pooled.reshape(-1, pooled.shape[-1])
                U, S, V = torch.pca_lowrank(flat, q=min(config.pca_components, flat.shape[0], flat.shape[1]))
                pca_reduced = (flat @ V).reshape(*pooled.shape[:-1], V.shape[1])
            except Exception as e:
                logger.warning(f"PCA failed: {e}")
        
        return HiddenStateFeatures(
            states=stacked,
            pca_reduced=pca_reduced,
            pooled=pooled,
            layer_norms=layer_norms,
            metadata={
                "n_layers": len(states),
                "response_tokens": total_length - prompt_length,
                "hidden_size": stacked.shape[-1] if stacked.dim() >= 4 else 0,
                "pooling": config.pooling
            }
        )
    
    def _process_token_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_length: int,
        config: TokenProbConfig
    ) -> TokenProbFeatures:
        """Process logits to extract token probability features."""
        response_logits = logits[:, prompt_length-1:-1, :].cpu()
        response_ids = input_ids[:, prompt_length:].cpu()
        
        if response_logits.shape[1] == 0:
            return TokenProbFeatures(metadata={"warning": "no_response_tokens"})
        
        probs = torch.softmax(response_logits, dim=-1)
        
        token_probs = probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        
        entropy = None
        mean_entropy = None
        max_entropy = None
        if config.compute_entropy:
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            mean_entropy = float(entropy.mean())
            max_entropy = float(entropy.max())
        
        perplexity = None
        if config.compute_perplexity:
            log_probs = torch.log(token_probs + 1e-10)
            perplexity = float(torch.exp(-log_probs.mean()))
        
        top_k_probs = None
        if config.top_k_probs > 0:
            top_k_probs = torch.topk(probs, min(config.top_k_probs, probs.shape[-1]), dim=-1)[0]
        
        return TokenProbFeatures(
            probs=token_probs,
            entropy=entropy,
            perplexity=perplexity,
            top_k_probs=top_k_probs,
            mean_entropy=mean_entropy,
            max_entropy=max_entropy,
            metadata={"response_tokens": response_ids.shape[1]}
        )


@FEATURE_EXTRACTOR_REGISTRY.register("attention_only")
class AttentionOnlyExtractor(BaseFeatureExtractor):
    """Extractor for attention features only (faster)."""
    
    @classmethod
    def component_name(cls) -> str:
        return "attention_only"
    
    def extract(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        modified_config = FeatureExtractionConfig(
            mode=self.config.mode,
            max_length=self.config.max_length,
            attention=self.config.attention,
            hidden_states=HiddenStateConfig(enabled=False),
            token_probs=TokenProbConfig(enabled=False),
        )
        extractor = StandardFeatureExtractor(modified_config)
        return extractor.extract(model, sample)


@FEATURE_EXTRACTOR_REGISTRY.register("probability_only")
class ProbabilityOnlyExtractor(BaseFeatureExtractor):
    """Extractor for token probability features only (faster)."""
    
    @classmethod
    def component_name(cls) -> str:
        return "probability_only"
    
    def extract(self, model: Any, sample: DataSample) -> ExtractedFeatures:
        modified_config = FeatureExtractionConfig(
            mode=self.config.mode,
            max_length=self.config.max_length,
            attention=AttentionConfig(enabled=False),
            hidden_states=HiddenStateConfig(enabled=False),
            token_probs=self.config.token_probs,
        )
        extractor = StandardFeatureExtractor(modified_config)
        return extractor.extract(model, sample)


def create_extractor(
    config: Optional[FeatureExtractionConfig] = None,
    name: str = "default"
) -> BaseFeatureExtractor:
    """Create feature extractor by name."""
    if config is None:
        config = FeatureExtractionConfig()
    return FEATURE_EXTRACTOR_REGISTRY.create(name, config=config)
