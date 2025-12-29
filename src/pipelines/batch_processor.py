"""Batch processor for one-click processing of all RAGTruth task_types.

Features:
- Process all task_types (QA, Summary, Data2txt) in one command
- Process all splits (train, test) automatically
- Integrated tqdm progress bars at all stages
- Organized output paths: {dataset}_{model}_{mode}_{split}_{task_type}
- Automatic hallucination annotation via Lookback Lens
- lapeigvals-compatible metadata.jsonl output
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Literal, Iterator
from dataclasses import dataclass, field
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    get_logger, ProgressLogger,
    DataSample, LabeledSample, ExtractedFeatures,
    FeatureExtractionConfig, ExtractionMode, AttentionConfig,
    BatchProcessConfig, OutputConfig, MetadataEntry,
    TaskType, SplitType,
)
from src.datasets import RAGTruthParser
from src.models import load_model, ModelConfig
from src.features import create_extractor
from src.annotators import (
    LookbackLensAnnotator, LookbackLensConfig,
    AnnotationResult, AnnotationMethod
)

logger = get_logger(__name__)

# RAGTruth available task types
RAGTRUTH_TASK_TYPES = ["QA", "Summary", "Data2txt"]
RAGTRUTH_SPLITS = ["train", "test"]


@dataclass
class BatchProcessResult:
    """Result of batch processing."""
    task_type: str
    split: str
    total_samples: int
    processed_samples: int
    error_samples: int
    output_dir: Path
    feature_files: List[str] = field(default_factory=list)
    metadata_file: Optional[str] = None
    processing_time: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "split": self.split,
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "error_samples": self.error_samples,
            "output_dir": str(self.output_dir),
            "feature_files_count": len(self.feature_files),
            "metadata_file": self.metadata_file,
            "processing_time": self.processing_time,
            "statistics": self.statistics,
        }


class BatchProcessor:
    """Batch processor for one-click RAGTruth processing.
    
    Supports processing all task_types and splits with:
    - Flexible attention layer selection (first, last, all, specific indices)
    - Automatic Lookback Lens hallucination annotation
    - Teacher forcing mode with gold labels
    - Organized output directory structure
    - Progress bars at all processing stages
    
    Usage:
        processor = BatchProcessor(
            data_path="./data/ragtruth",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="./outputs",
            attention_layers="first",  # Only first layer for efficiency
        )
        results = processor.process_all()
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir: Union[str, Path] = "./outputs",
        mode: ExtractionMode = ExtractionMode.TEACHER_FORCING,
        attention_layers: Union[str, List[int]] = "first",
        task_types: Optional[Union[str, List[str]]] = "all",
        splits: Optional[Union[str, List[str]]] = "all",
        max_samples_per_task: Optional[int] = None,
        enable_annotation: bool = True,
        annotation_threshold: float = 0.5,
        show_progress: bool = True,
        model_config: Optional[ModelConfig] = None,
    ):
        """Initialize batch processor.
        
        Args:
            data_path: Path to RAGTruth dataset
            model_name: Model to use for feature extraction
            output_dir: Base output directory
            mode: Extraction mode (teacher_forcing or generation)
            attention_layers: Which layers to extract ("first", "last", "all", or list)
            task_types: Task types to process ("all" or list)
            splits: Splits to process ("all" or list)
            max_samples_per_task: Max samples per task (None = all)
            enable_annotation: Enable Lookback Lens annotation
            annotation_threshold: Threshold for hallucination detection
            show_progress: Show tqdm progress bars
            model_config: Optional custom model configuration
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.attention_layers = attention_layers
        self.max_samples_per_task = max_samples_per_task
        self.enable_annotation = enable_annotation
        self.annotation_threshold = annotation_threshold
        self.show_progress = show_progress
        
        # Resolve task types
        if task_types == "all" or task_types is None:
            self.task_types = RAGTRUTH_TASK_TYPES
        else:
            self.task_types = task_types if isinstance(task_types, list) else [task_types]
        
        # Resolve splits
        if splits == "all" or splits is None:
            self.splits = RAGTRUTH_SPLITS
        else:
            self.splits = splits if isinstance(splits, list) else [splits]
        
        # Model config
        self.model_config = model_config or ModelConfig(
            name=model_name,
            dtype="float16",
            device_map="auto",
            attn_implementation="eager",  # Required for attention output
        )
        
        # Output config
        self.output_config = OutputConfig(
            base_dir=self.output_dir,
            path_template="{dataset}_{model}_{mode}_{split}",
            save_features=True,
            save_metadata=True,
        )
        
        # Initialize annotator if enabled
        self.annotator = None
        if enable_annotation:
            self.annotator = LookbackLensAnnotator(LookbackLensConfig(
                threshold=annotation_threshold,
                layer_to_use="first",
            ))
        
        # Model will be loaded lazily
        self._model = None
        self._extractor = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = load_model(self.model_config)
        return self._model
    
    @property
    def extractor(self):
        """Lazy create extractor."""
        if self._extractor is None:
            # Build attention config with layer selection
            attn_config = AttentionConfig(
                enabled=True,
                layers=self.attention_layers,
                compute_eigenvalues=True,
                compute_laplacian=True,
                n_eigenvalues=10,
            )
            
            feat_config = FeatureExtractionConfig(
                mode=self.mode,
                attention=attn_config,
            )
            
            self._extractor = create_extractor(feat_config)
        return self._extractor
    
    def process_all(self) -> List[BatchProcessResult]:
        """Process all task types and splits.
        
        Returns:
            List of BatchProcessResult for each (task_type, split) combination
        """
        all_results = []
        
        total_combinations = len(self.task_types) * len(self.splits)
        
        logger.info(f"Starting batch processing:")
        logger.info(f"  - Task types: {self.task_types}")
        logger.info(f"  - Splits: {self.splits}")
        logger.info(f"  - Mode: {self.mode.value}")
        logger.info(f"  - Attention layers: {self.attention_layers}")
        logger.info(f"  - Total combinations: {total_combinations}")
        
        try:
            from tqdm import tqdm
            combinations = list(self._get_combinations())
            iterator = tqdm(
                combinations,
                desc="Processing task/split combinations",
                disable=not self.show_progress
            )
        except ImportError:
            iterator = list(self._get_combinations())
            if self.show_progress:
                logger.warning("tqdm not installed, progress bar disabled")
        
        for task_type, split in iterator:
            if hasattr(iterator, 'set_postfix'):
                iterator.set_postfix(task=task_type, split=split)
            
            try:
                result = self.process_single(task_type, split)
                all_results.append(result)
                logger.info(
                    f"Completed {task_type}/{split}: "
                    f"{result.processed_samples}/{result.total_samples} samples"
                )
            except Exception as e:
                logger.error(f"Failed to process {task_type}/{split}: {e}")
                all_results.append(BatchProcessResult(
                    task_type=task_type,
                    split=split,
                    total_samples=0,
                    processed_samples=0,
                    error_samples=0,
                    output_dir=self.output_dir,
                    statistics={"error": str(e)}
                ))
        
        # Save overall summary
        self._save_summary(all_results)
        
        return all_results
    
    def _get_combinations(self) -> Iterator[tuple[str, str]]:
        """Generate all (task_type, split) combinations."""
        for task_type in self.task_types:
            for split in self.splits:
                yield task_type, split
    
    def process_single(
        self,
        task_type: str,
        split: str
    ) -> BatchProcessResult:
        """Process a single task_type + split combination.
        
        Args:
            task_type: Task type (QA, Summary, Data2txt)
            split: Data split (train, test)
            
        Returns:
            BatchProcessResult with processing statistics
        """
        start_time = time.time()
        
        # Create output directory
        output_path = self._get_output_path(task_type, split)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load samples
        samples = self._load_samples(task_type, split)
        total_samples = len(samples)
        
        if total_samples == 0:
            return BatchProcessResult(
                task_type=task_type,
                split=split,
                total_samples=0,
                processed_samples=0,
                error_samples=0,
                output_dir=output_path,
            )
        
        logger.info(f"Processing {task_type}/{split}: {total_samples} samples")
        
        # Process samples with progress bar
        processed_count = 0
        error_count = 0
        metadata_entries = []
        feature_files = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(
                samples,
                desc=f"{task_type}/{split}",
                disable=not self.show_progress,
                leave=False
            )
        except ImportError:
            iterator = samples
        
        for sample in iterator:
            try:
                # Extract features
                features = self.extractor.extract(self.model, sample)
                
                # Annotate if enabled
                annotation = None
                if self.enable_annotation and self.annotator:
                    attn_weights = features.attention_features.get("weights") if features.attention_features else None
                    annotation = self.annotator.annotate(
                        sample=sample,
                        attention_weights=attn_weights,
                        prompt_length=features.prompt_length,
                    )
                
                # Save feature file
                feature_file = self._save_features(features, output_path)
                feature_files.append(feature_file)
                
                # Create metadata entry
                metadata_entry = self._create_metadata_entry(
                    sample, features, annotation, feature_file
                )
                metadata_entries.append(metadata_entry)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                error_count += 1
        
        # Save metadata.jsonl
        metadata_file = self._save_metadata(metadata_entries, output_path)
        
        # Compute statistics
        statistics = self._compute_statistics(metadata_entries)
        
        processing_time = time.time() - start_time
        
        return BatchProcessResult(
            task_type=task_type,
            split=split,
            total_samples=total_samples,
            processed_samples=processed_count,
            error_samples=error_count,
            output_dir=output_path,
            feature_files=feature_files,
            metadata_file=metadata_file,
            processing_time=processing_time,
            statistics=statistics,
        )
    
    def _get_output_path(self, task_type: str, split: str) -> Path:
        """Generate output path for a task/split combination."""
        model_safe = self.model_name.replace("/", "_").replace("\\", "_")
        
        path_name = f"ragtruth_{model_safe}_{self.mode.value}_{split}_{task_type}"
        return self.output_dir / path_name
    
    def _load_samples(self, task_type: str, split: str) -> List[LabeledSample]:
        """Load samples for a specific task type and split."""
        parser = RAGTruthParser(
            path=self.data_path,
            split=split,
            task_types=[task_type],
        )
        
        samples = parser.load_all(max_samples=self.max_samples_per_task)
        return samples
    
    def _save_features(self, features: ExtractedFeatures, output_path: Path) -> str:
        """Save extracted features to file."""
        feature_file = f"{features.sample_id}.pt"
        feature_path = output_path / "features" / feature_file
        feature_path.parent.mkdir(exist_ok=True)
        
        try:
            import torch
            torch.save(features.model_dump(), feature_path)
        except Exception:
            # Fallback to JSON
            feature_file = f"{features.sample_id}.json"
            feature_path = output_path / "features" / feature_file
            with open(feature_path, 'w') as f:
                json.dump(features.model_dump(), f, default=str)
        
        return feature_file
    
    def _create_metadata_entry(
        self,
        sample: LabeledSample,
        features: ExtractedFeatures,
        annotation: Optional[AnnotationResult],
        feature_file: str
    ) -> MetadataEntry:
        """Create metadata entry in lapeigvals format."""
        # Get layers used
        layers_used = features.layers_extracted if features.layers_extracted else []
        if not layers_used and features.attention_features:
            layers_used = features.attention_features.get("metadata", {}).get("layers", [])
        
        # Get token annotations if available
        token_annotations = []
        if annotation:
            token_annotations = annotation.to_dict().get("token_annotations", [])
        
        return MetadataEntry(
            sample_id=sample.id,
            question=sample.question,
            gold_answer=sample.gold_answer,
            model_answer=sample.answer,
            context=sample.context,
            task_type=sample.task_type.value if isinstance(sample.task_type, TaskType) else str(sample.task_type),
            split=sample.split.value if isinstance(sample.split, SplitType) else str(sample.split or "unknown"),
            label=sample.label or (1 if sample.has_hallucination else 0),
            has_hallucination=sample.has_hallucination,
            hallucination_spans=[
                {
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "label_type": span.label_type.value,
                }
                for span in sample.hallucination_spans
            ],
            token_annotations=token_annotations,
            source_model=sample.source_model,
            feature_file=feature_file,
            extraction_mode=self.mode.value,
            attention_layers_used=layers_used,
        )
    
    def _save_metadata(
        self,
        entries: List[MetadataEntry],
        output_path: Path
    ) -> str:
        """Save metadata entries to jsonl file."""
        metadata_file = "metadata.jsonl"
        metadata_path = output_path / metadata_file
        
        with open(metadata_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry.model_dump(), default=str) + '\n')
        
        return metadata_file
    
    def _compute_statistics(self, entries: List[MetadataEntry]) -> Dict[str, Any]:
        """Compute statistics from processed entries."""
        if not entries:
            return {}
        
        total = len(entries)
        hallucinated = sum(1 for e in entries if e.has_hallucination)
        
        return {
            "total_samples": total,
            "hallucinated_samples": hallucinated,
            "clean_samples": total - hallucinated,
            "hallucination_rate": hallucinated / total if total > 0 else 0,
        }
    
    def _save_summary(self, results: List[BatchProcessResult]) -> None:
        """Save overall processing summary."""
        summary = {
            "model": self.model_name,
            "mode": self.mode.value,
            "attention_layers": str(self.attention_layers),
            "enable_annotation": self.enable_annotation,
            "task_types": self.task_types,
            "splits": self.splits,
            "results": [r.to_dict() for r in results],
            "total_processed": sum(r.processed_samples for r in results),
            "total_errors": sum(r.error_samples for r in results),
            "total_time": sum(r.processing_time for r in results),
        }
        
        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved batch summary to {summary_path}")


def process_all_tasks(
    data_path: Union[str, Path],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: Union[str, Path] = "./outputs",
    attention_layers: Union[str, List[int]] = "first",
    **kwargs
) -> List[BatchProcessResult]:
    """Convenience function for one-click batch processing.
    
    Args:
        data_path: Path to RAGTruth dataset
        model_name: Model name to use
        output_dir: Output directory
        attention_layers: Which layers to extract
        **kwargs: Additional arguments for BatchProcessor
        
    Returns:
        List of BatchProcessResult
    """
    processor = BatchProcessor(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        attention_layers=attention_layers,
        **kwargs
    )
    return processor.process_all()


def create_batch_processor(config: BatchProcessConfig) -> BatchProcessor:
    """Create BatchProcessor from config object.
    
    Args:
        config: BatchProcessConfig with all settings
        
    Returns:
        Configured BatchProcessor
    """
    return BatchProcessor(
        data_path=config.data_path,
        model_name=config.model,
        output_dir=config.output_dir,
        mode=config.mode,
        attention_layers=config.attention_layers,
        task_types=config.task_types,
        splits=config.splits,
        max_samples_per_task=config.max_samples_per_task,
        enable_annotation=config.enable_annotation,
        show_progress=config.show_progress,
    )
