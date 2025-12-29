"""Core module - Registry, types, configuration, logging, caching."""
from .registry import (
    Registry, RegistryError, BaseComponent,
    DATASET_REGISTRY, MODEL_REGISTRY, FEATURE_EXTRACTOR_REGISTRY,
    METHOD_REGISTRY, EVALUATOR_REGISTRY, PROMPT_REGISTRY,
    get_registry,
)

from .types import (
    TaskType, SplitType, ExtractionMode, HallucinationType,
    DataSample, LabeledSample, HallucinationSpan, ExtractedFeatures,
    TokenAnnotation,  # NEW
    AttentionConfig, HiddenStateConfig, TokenProbConfig,
    FeatureExtractionConfig, ModelConfig, GenerationConfig,
    TrainingConfig, EvaluationConfig, DatasetConfig, Prediction,
    OutputConfig, MetadataEntry, BatchProcessConfig,  # NEW
)

from .exceptions import (
    HallucinationDetectionError, ConfigurationError,
    DatasetError, DatasetNotFoundError, DatasetParseError,
    ModelError, ModelLoadError, ModelInferenceError,
    FeatureExtractionError, MethodError, MethodNotFittedError,
    EvaluationError, CacheError, ValidationError, PipelineError, PromptError,
)

from .logging import (
    setup_logging, get_logger, ProgressLogger, LogContext,
    JsonFormatter, ColoredFormatter,
)

from .config import (
    ConfigManager, ConfigLoader, init_config, get_config, get_value,
    expand_env_vars, deep_merge, load_yaml,
)

from .cache import (
    FeatureCache, MetadataCache, CacheManager,
    compute_hash, cached,
)

__all__ = [
    # Registry
    "Registry", "RegistryError", "BaseComponent",
    "DATASET_REGISTRY", "MODEL_REGISTRY", "FEATURE_EXTRACTOR_REGISTRY",
    "METHOD_REGISTRY", "EVALUATOR_REGISTRY", "PROMPT_REGISTRY", "get_registry",
    
    # Types
    "TaskType", "SplitType", "ExtractionMode", "HallucinationType",
    "DataSample", "LabeledSample", "HallucinationSpan", "ExtractedFeatures",
    "TokenAnnotation",  # NEW
    "AttentionConfig", "HiddenStateConfig", "TokenProbConfig",
    "FeatureExtractionConfig", "ModelConfig", "GenerationConfig",
    "TrainingConfig", "EvaluationConfig", "DatasetConfig", "Prediction",
    "OutputConfig", "MetadataEntry", "BatchProcessConfig",  # NEW
    
    # Exceptions
    "HallucinationDetectionError", "ConfigurationError",
    "DatasetError", "DatasetNotFoundError", "DatasetParseError",
    "ModelError", "ModelLoadError", "ModelInferenceError",
    "FeatureExtractionError", "MethodError", "MethodNotFittedError",
    "EvaluationError", "CacheError", "ValidationError", "PipelineError", "PromptError",
    
    # Logging
    "setup_logging", "get_logger", "ProgressLogger", "LogContext",
    "JsonFormatter", "ColoredFormatter",
    
    # Config
    "ConfigManager", "ConfigLoader", "init_config", "get_config", "get_value",
    "expand_env_vars", "deep_merge", "load_yaml",
    
    # Cache
    "FeatureCache", "MetadataCache", "CacheManager", "compute_hash", "cached",
]
