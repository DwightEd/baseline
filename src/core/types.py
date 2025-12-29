"""Type definitions and Pydantic models."""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    DATA2TXT = "data2txt"
    DIALOGUE = "dialogue"
    MATH = "math"
    GENERATION = "generation"


class SplitType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ExtractionMode(str, Enum):
    GENERATION = "generation"
    TEACHER_FORCING = "teacher_forcing"


class HallucinationType(str, Enum):
    EVIDENT_CONFLICT = "evident_conflict"
    EVIDENT_BASELESS = "evident_baseless"
    SUBTLE_CONFLICT = "subtle_conflict"
    SUBTLE_BASELESS = "subtle_baseless"
    UNKNOWN = "unknown"


class DataSample(BaseModel):
    """Standardized data sample."""
    id: str = Field(..., description="Unique identifier")
    question: str = Field(..., description="Input question/prompt")
    answer: str = Field(default="", description="Generated answer")
    gold_answer: str = Field(default="", description="Reference answer")
    context: Optional[str] = Field(default=None, description="Context/passages")
    task_type: TaskType = Field(default=TaskType.QA)
    split: Optional[SplitType] = Field(default=None)
    label: Optional[int] = Field(default=None, description="0/1 label")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"frozen": False, "extra": "allow"}
    
    @field_validator('question', 'answer', 'gold_answer', mode='before')
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        return str(v) if v is not None else ""


class HallucinationSpan(BaseModel):
    """Hallucination span annotation."""
    text: str
    start: int
    end: int
    label_type: HallucinationType = HallucinationType.UNKNOWN
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LabeledSample(DataSample):
    """Sample with hallucination labels."""
    has_hallucination: bool = Field(default=False)
    hallucination_spans: List[HallucinationSpan] = Field(default_factory=list)
    source_model: Optional[str] = Field(default=None)


class TokenAnnotation(BaseModel):
    """Token-level hallucination annotation (Lookback Lens compatible).
    
    This format is compatible with the Lookback Lens paper's approach
    of detecting hallucinations via attention pattern analysis.
    """
    token_id: int = Field(..., description="Token ID from tokenizer")
    token_text: str = Field(..., description="Decoded token text")
    position: int = Field(..., description="Position in response sequence")
    is_hallucination: bool = Field(default=False)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Lookback Lens specific fields
    lookback_ratio: Optional[float] = Field(default=None, description="Attention ratio to context vs recent tokens")
    attention_to_context: Optional[float] = Field(default=None, description="Attention weight to context/prompt")
    attention_to_recent: Optional[float] = Field(default=None, description="Attention weight to recent generated tokens")
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AttentionConfig(BaseModel):
    """Attention extraction config with flexible layer selection.
    
    Layer selection modes:
    - "all": Extract from all layers
    - "first": Extract from first layer only (layer 0)
    - "last": Extract from last layer only
    - "first_n": Extract from first N layers (use first_n parameter)
    - "last_n": Extract from last N layers (use last_n parameter)
    - List[int]: Extract from specific layer indices, e.g., [0, 4, 8]
    """
    enabled: bool = True
    layers: Union[Literal["all", "first", "last", "first_n", "last_n"], List[int]] = "all"
    first_n: int = Field(default=1, ge=1, description="Number of layers for 'first_n' mode")
    last_n: int = Field(default=4, ge=1, description="Number of layers for 'last_n' mode")
    heads: Union[Literal["all"], List[int]] = "all"
    compute_eigenvalues: bool = True
    n_eigenvalues: int = Field(default=10, ge=1)
    compute_laplacian: bool = True
    normalization: Literal["symmetric", "random_walk"] = "symmetric"
    save_raw_weights: bool = Field(default=False, description="Save full attention matrices")

    def get_layer_indices(self, n_layers: int) -> List[int]:
        """Resolve layer specification to actual indices.
        
        Args:
            n_layers: Total number of layers in the model
            
        Returns:
            List of layer indices to extract
        """
        if self.layers == "all":
            return list(range(n_layers))
        elif self.layers == "first":
            return [0]
        elif self.layers == "last":
            return [n_layers - 1]
        elif self.layers == "first_n":
            return list(range(min(self.first_n, n_layers)))
        elif self.layers == "last_n":
            start = max(0, n_layers - self.last_n)
            return list(range(start, n_layers))
        elif isinstance(self.layers, list):
            # Filter valid indices
            return [i for i in self.layers if 0 <= i < n_layers]
        return list(range(n_layers))


class HiddenStateConfig(BaseModel):
    """Hidden state extraction config."""
    enabled: bool = True
    layers: Union[Literal["all"], Literal["last_n"], List[int]] = "last_n"
    last_n: int = Field(default=4, ge=1)
    compute_pca: bool = True
    pca_components: int = Field(default=64, ge=1)
    pooling: Literal["mean", "max", "last", "first"] = "mean"


class TokenProbConfig(BaseModel):
    """Token probability extraction config."""
    enabled: bool = True
    compute_entropy: bool = True
    compute_perplexity: bool = True
    top_k_probs: int = Field(default=10, ge=1)


class FeatureExtractionConfig(BaseModel):
    """Complete feature extraction config."""
    mode: ExtractionMode = ExtractionMode.TEACHER_FORCING
    batch_size: int = Field(default=1, ge=1)
    max_length: int = Field(default=4096, ge=1)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    hidden_states: HiddenStateConfig = Field(default_factory=HiddenStateConfig)
    token_probs: TokenProbConfig = Field(default_factory=TokenProbConfig)
    cache_features: bool = True
    cache_dir: Optional[Path] = None


class ExtractedFeatures(BaseModel):
    """Container for extracted features."""
    sample_id: str
    prompt_length: int
    response_length: int
    total_length: int
    attention_features: Optional[Dict[str, Any]] = None
    hidden_state_features: Optional[Dict[str, Any]] = None
    token_prob_features: Optional[Dict[str, Any]] = None
    token_annotations: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Token-level hallucination annotations (Lookback Lens format)"
    )
    layers_extracted: List[int] = Field(
        default_factory=list,
        description="Which attention layers were extracted"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}


class ModelConfig(BaseModel):
    """Model loading config."""
    name: str
    path: Optional[str] = None
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    max_length: int = Field(default=4096, ge=1)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    @model_validator(mode='after')
    def validate_quantization(self) -> 'ModelConfig':
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        return self


class GenerationConfig(BaseModel):
    """Generation config."""
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = True
    repetition_penalty: float = Field(default=1.0, ge=1.0)


class TrainingConfig(BaseModel):
    """Training config."""
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    scheduler: Optional[Literal["cosine", "linear", "step"]] = None
    early_stopping_patience: int = Field(default=10, ge=1)
    gradient_clip_norm: Optional[float] = Field(default=1.0, ge=0)


class EvaluationConfig(BaseModel):
    """Evaluation config."""
    metrics: List[str] = Field(default_factory=lambda: ["auroc", "auprc", "f1", "accuracy"])
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    bootstrap_samples: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    """Dataset config."""
    name: str
    path: Path
    format: Literal["json", "jsonl", "csv", "parquet", "huggingface"] = "jsonl"
    split: Optional[SplitType] = None
    max_samples: Optional[int] = Field(default=None, ge=1)
    task_type: TaskType = TaskType.QA
    field_mapping: Dict[str, str] = Field(default_factory=dict)


class Prediction(BaseModel):
    """Detection prediction."""
    sample_id: str
    score: float = Field(ge=0.0, le=1.0)
    label: int = Field(ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Output Configuration
# ============================================================================
class OutputConfig(BaseModel):
    """Output configuration for organized file structure.
    
    Supports path templates with the following variables:
    - {dataset}: Dataset name (e.g., "ragtruth")
    - {model}: Model name (sanitized for filesystem)
    - {mode}: Extraction mode ("teacher_forcing" or "generation")
    - {split}: Data split ("train", "test", "validation")
    - {task_type}: Task type (optional, e.g., "QA", "Summary")
    """
    base_dir: Path = Field(default=Path("./outputs"))
    path_template: str = Field(
        default="{dataset}_{model}_{mode}_{split}",
        description="Template for output directory names"
    )
    save_features: bool = Field(default=True, description="Save extracted features")
    save_metadata: bool = Field(default=True, description="Save metadata.jsonl (lapeigvals format)")
    save_attention_maps: bool = Field(default=False, description="Save raw attention matrices")
    compress: bool = Field(default=True, description="Compress feature files")
    
    def get_output_path(
        self,
        dataset: str,
        model: str,
        mode: str,
        split: str,
        task_type: Optional[str] = None
    ) -> Path:
        """Generate organized output path.
        
        Args:
            dataset: Dataset name
            model: Model name
            mode: Extraction mode
            split: Data split
            task_type: Optional task type for further organization
            
        Returns:
            Path to output directory
        """
        # Sanitize model name for filesystem
        model_safe = model.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        path_name = self.path_template.format(
            dataset=dataset,
            model=model_safe,
            mode=mode,
            split=split
        )
        
        if task_type:
            path_name = f"{path_name}_{task_type}"
        
        return self.base_dir / path_name


# ============================================================================
# Metadata Entry (lapeigvals compatible format)
# ============================================================================
class MetadataEntry(BaseModel):
    """Metadata entry for lapeigvals-compatible output.
    
    This format ensures compatibility with the lapeigvals project's
    expected data structure for training and evaluation.
    """
    sample_id: str
    question: str
    gold_answer: str
    model_answer: str
    context: Optional[str] = None
    task_type: str
    split: str
    label: int = Field(description="0 = no hallucination, 1 = has hallucination")
    has_hallucination: bool
    hallucination_spans: List[Dict[str, Any]] = Field(default_factory=list)
    token_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    source_model: Optional[str] = None
    feature_file: str = Field(description="Path to corresponding feature file")
    extraction_mode: str
    attention_layers_used: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Batch Processing Configuration
# ============================================================================
class BatchProcessConfig(BaseModel):
    """Batch processing configuration for one-click processing.
    
    Supports processing all task_types and splits in a single command.
    """
    dataset: str = Field(default="ragtruth", description="Dataset name")
    data_path: Path = Field(default=Path("./data/ragtruth"), description="Path to dataset")
    model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", description="Model to use")
    mode: ExtractionMode = Field(default=ExtractionMode.TEACHER_FORCING)
    # Process all task_types: None or "all" means all available
    task_types: Optional[Union[Literal["all"], List[str]]] = Field(
        default="all",
        description="Task types to process ('all' or list like ['QA', 'Summary'])"
    )
    # Process all splits: None or "all" means all available  
    splits: Optional[Union[Literal["all"], List[str]]] = Field(
        default="all",
        description="Splits to process ('all' or list like ['train', 'test'])"
    )
    max_samples_per_task: Optional[int] = Field(
        default=None,
        description="Max samples per task type (None for all)"
    )
    output_dir: Path = Field(default=Path("./outputs"))
    # Attention layer config for batch
    attention_layers: Union[Literal["all", "first", "last", "first_n", "last_n"], List[int]] = Field(
        default="first",
        description="Which attention layers to extract"
    )
    enable_annotation: bool = Field(default=True, description="Enable hallucination annotation")
    show_progress: bool = Field(default=True, description="Show tqdm progress bars")
