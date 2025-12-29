"""Model loading and management with registry pattern."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import gc

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    BaseComponent, MODEL_REGISTRY,
    ModelConfig, GenerationConfig,
    ModelError, ModelLoadError, ModelInferenceError,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """Container for loaded model and tokenizer with utility methods."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: ModelConfig
    device: torch.device
    
    def generate(
        self,
        texts: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate text from input prompts."""
        if isinstance(texts, str):
            texts = [texts]
        
        gen_cfg = generation_config or GenerationConfig()
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature if gen_cfg.do_sample else 1.0,
                top_p=gen_cfg.top_p if gen_cfg.do_sample else 1.0,
                top_k=gen_cfg.top_k if gen_cfg.do_sample else 0,
                do_sample=gen_cfg.do_sample,
                repetition_penalty=gen_cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[:, input_length:]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Forward pass with optional attention and hidden state outputs."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs
            )
        
        return {
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any
    ) -> torch.Tensor:
        """Encode text to token ids."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            add_special_tokens=add_special_tokens,
            **kwargs
        ).input_ids.to(self.device)
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs: Any
    ) -> str:
        """Decode token ids to text."""
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
    
    def get_token_probabilities(
        self,
        text: str,
        target_text: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get token probabilities for text."""
        if target_text:
            full_text = text + target_text
        else:
            full_text = text
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs.gather(-1, inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        return token_probs, inputs.input_ids
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        return 0
    
    @property
    def num_heads(self) -> int:
        """Get number of attention heads."""
        if hasattr(self.model.config, 'num_attention_heads'):
            return self.model.config.num_attention_heads
        return 0
    
    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        return 0


class BaseModelLoader(BaseComponent, ABC):
    """Abstract base class for model loaders."""
    
    @classmethod
    @abstractmethod
    def component_name(cls) -> str:
        pass
    
    @abstractmethod
    def load(self, config: ModelConfig) -> LoadedModel:
        """Load model with given configuration."""
        pass
    
    @abstractmethod
    def supports(self, model_name: str) -> bool:
        """Check if loader supports given model."""
        pass


@MODEL_REGISTRY.register("huggingface", aliases=["hf", "transformers"])
class HuggingFaceModelLoader(BaseModelLoader):
    """Loader for HuggingFace transformers models."""
    
    @classmethod
    def component_name(cls) -> str:
        return "huggingface"
    
    def supports(self, model_name: str) -> bool:
        return True
    
    def load(self, config: ModelConfig) -> LoadedModel:
        """Load model from HuggingFace."""
        model_path = config.path or config.name
        logger.info(f"Loading model: {model_path}")
        
        device = self._get_device(config.device_map)
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.dtype, torch.float16)
        
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=config.device_map if device.type == "cuda" else None,
                trust_remote_code=config.trust_remote_code,
                attn_implementation=config.attn_implementation,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
            
            if config.device_map != "auto" and device.type != "cuda":
                model = model.to(device)
            
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=config.trust_remote_code,
                padding_side="left",
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            actual_device = next(model.parameters()).device
            logger.info(f"Model loaded successfully on {actual_device}")
            logger.info(f"Model config: {model.config.num_hidden_layers} layers, "
                       f"{model.config.num_attention_heads} heads, "
                       f"{model.config.hidden_size} hidden size")
            
            return LoadedModel(
                model=model,
                tokenizer=tokenizer,
                config=config,
                device=actual_device,
            )
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model {model_path}: {e}",
                details={"model": model_path, "error": str(e)}
            )
    
    def _get_device(self, device_map: str) -> torch.device:
        """Determine the device to use."""
        if device_map == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_map)


class ModelManager:
    """Manages model loading, caching, and lifecycle."""
    
    def __init__(self):
        self._loaded_models: Dict[str, LoadedModel] = {}
        self._default_loader = HuggingFaceModelLoader()
    
    def load(
        self,
        name_or_config: Union[str, ModelConfig],
        cache: bool = True,
        **kwargs: Any
    ) -> LoadedModel:
        """Load model by name or config."""
        if isinstance(name_or_config, str):
            config = ModelConfig(name=name_or_config, **kwargs)
        else:
            config = name_or_config
        
        cache_key = self._make_cache_key(config)
        
        if cache and cache_key in self._loaded_models:
            logger.info(f"Using cached model: {config.name}")
            return self._loaded_models[cache_key]
        
        loader = self._get_loader(config.name)
        loaded = loader.load(config)
        
        if cache:
            self._loaded_models[cache_key] = loaded
        
        return loaded
    
    def _make_cache_key(self, config: ModelConfig) -> str:
        """Create cache key from config."""
        return f"{config.name}_{config.dtype}_{config.load_in_8bit}_{config.load_in_4bit}"
    
    def _get_loader(self, model_name: str) -> BaseModelLoader:
        """Get appropriate loader for model."""
        for loader_name in MODEL_REGISTRY.list_registered():
            try:
                loader = MODEL_REGISTRY.create(loader_name)
                if loader.supports(model_name):
                    return loader
            except Exception:
                continue
        return self._default_loader
    
    def unload(self, name: str) -> bool:
        """Unload model from cache."""
        keys_to_remove = [k for k in self._loaded_models if name in k]
        
        for key in keys_to_remove:
            model_container = self._loaded_models.pop(key)
            del model_container.model
            del model_container.tokenizer
            logger.info(f"Unloaded model: {key}")
        
        if keys_to_remove:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
    
    def unload_all(self) -> None:
        """Unload all cached models."""
        for key in list(self._loaded_models.keys()):
            model_container = self._loaded_models.pop(key)
            del model_container.model
            del model_container.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded all models")
    
    def list_loaded(self) -> List[str]:
        """List currently loaded models."""
        return list(self._loaded_models.keys())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        info = {"models": {}}
        
        for key, model in self._loaded_models.items():
            param_count = sum(p.numel() for p in model.model.parameters())
            param_bytes = sum(p.numel() * p.element_size() for p in model.model.parameters())
            info["models"][key] = {
                "parameters": param_count,
                "memory_mb": param_bytes / (1024 * 1024),
                "device": str(model.device),
            }
        
        if torch.cuda.is_available():
            info["cuda"] = {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            }
        
        return info


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_model(
    name_or_config: Union[str, ModelConfig],
    cache: bool = True,
    **kwargs: Any
) -> LoadedModel:
    """Convenience function to load a model."""
    return get_model_manager().load(name_or_config, cache=cache, **kwargs)


def unload_model(name: str) -> bool:
    """Convenience function to unload a model."""
    return get_model_manager().unload(name)


def unload_all_models() -> None:
    """Convenience function to unload all models."""
    get_model_manager().unload_all()
