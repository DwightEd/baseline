"""Output writer for lapeigvals-compatible format.

This module provides utilities for writing features and metadata
in a format compatible with the lapeigvals project:
https://github.com/graphml-lab-pwr/lapeigvals

Output structure:
    {output_dir}/
    ├── metadata.jsonl          # Sample metadata with questions, answers, labels
    ├── features/
    │   ├── {sample_id}.pt      # PyTorch tensor features
    │   └── {sample_id}.json    # JSON fallback
    ├── index.json              # Feature file index
    └── extraction_summary.json # Processing summary
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator
from dataclasses import dataclass
import json
import gzip
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    get_logger,
    ExtractedFeatures, MetadataEntry, OutputConfig,
    LabeledSample,
)

logger = get_logger(__name__)


@dataclass
class WriteResult:
    """Result of write operation."""
    success: bool
    file_path: str
    file_size: int = 0
    error: Optional[str] = None


class OutputWriter:
    """Writer for organized output structure.
    
    Creates output directories and files in lapeigvals-compatible format:
    - metadata.jsonl: Line-delimited JSON with sample metadata
    - features/: Directory containing extracted features
    - index.json: Index mapping sample IDs to feature files
    
    Usage:
        writer = OutputWriter(output_dir="./outputs/ragtruth_llama_tf_test")
        writer.initialize()
        
        for sample, features in zip(samples, features_list):
            writer.write_sample(sample, features)
        
        writer.finalize()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[OutputConfig] = None,
        compress: bool = False,
    ):
        """Initialize output writer.
        
        Args:
            output_dir: Base output directory
            config: Optional output configuration
            compress: Whether to compress feature files
        """
        self.output_dir = Path(output_dir)
        self.config = config or OutputConfig()
        self.compress = compress
        
        # Internal state
        self._metadata_entries: List[Dict[str, Any]] = []
        self._feature_index: List[Dict[str, Any]] = []
        self._initialized = False
        self._finalized = False
    
    def initialize(self) -> None:
        """Initialize output directory structure."""
        if self._initialized:
            return
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        
        self._initialized = True
        logger.info(f"Initialized output directory: {self.output_dir}")
    
    def write_sample(
        self,
        sample: LabeledSample,
        features: ExtractedFeatures,
        annotation_data: Optional[Dict[str, Any]] = None,
    ) -> WriteResult:
        """Write a single sample's features and metadata.
        
        Args:
            sample: The labeled sample
            features: Extracted features
            annotation_data: Optional annotation results
            
        Returns:
            WriteResult with file path and status
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Write features
            feature_result = self._write_features(features)
            
            # Create metadata entry
            metadata = self._create_metadata(
                sample, features, feature_result.file_path, annotation_data
            )
            self._metadata_entries.append(metadata)
            
            # Add to index
            self._feature_index.append({
                "sample_id": sample.id,
                "feature_file": feature_result.file_path,
                "label": sample.label,
            })
            
            return feature_result
            
        except Exception as e:
            logger.error(f"Failed to write sample {sample.id}: {e}")
            return WriteResult(
                success=False,
                file_path="",
                error=str(e)
            )
    
    def _write_features(self, features: ExtractedFeatures) -> WriteResult:
        """Write features to file."""
        feature_dir = self.output_dir / "features"
        
        try:
            import torch
            
            # Prepare data for saving
            data = {
                "sample_id": features.sample_id,
                "prompt_length": features.prompt_length,
                "response_length": features.response_length,
                "total_length": features.total_length,
                "layers_extracted": features.layers_extracted,
                "metadata": features.metadata,
            }
            
            # Add feature tensors
            if features.attention_features:
                data["attention_features"] = features.attention_features
            if features.hidden_state_features:
                data["hidden_state_features"] = features.hidden_state_features
            if features.token_prob_features:
                data["token_prob_features"] = features.token_prob_features
            if features.token_annotations:
                data["token_annotations"] = features.token_annotations
            
            # Save as PyTorch file
            file_name = f"{features.sample_id}.pt"
            file_path = feature_dir / file_name
            torch.save(data, file_path)
            
            # Optionally compress
            if self.compress:
                self._compress_file(file_path)
                file_name = f"{features.sample_id}.pt.gz"
            
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            return WriteResult(
                success=True,
                file_path=file_name,
                file_size=file_size
            )
            
        except ImportError:
            # Fallback to JSON if PyTorch not available
            return self._write_features_json(features)
    
    def _write_features_json(self, features: ExtractedFeatures) -> WriteResult:
        """Fallback: Write features as JSON."""
        feature_dir = self.output_dir / "features"
        file_name = f"{features.sample_id}.json"
        file_path = feature_dir / file_name
        
        # Convert to JSON-serializable format
        data = features.model_dump()
        
        # Handle non-serializable types
        def convert_value(v):
            if hasattr(v, 'tolist'):
                return v.tolist()
            if hasattr(v, 'item'):
                return v.item()
            return v
        
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(convert_value(v)) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(convert_value(x)) for x in d]
            return convert_value(d)
        
        data = convert_dict(data)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, default=str)
        
        if self.compress:
            self._compress_file(file_path)
            file_name = f"{features.sample_id}.json.gz"
        
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        return WriteResult(
            success=True,
            file_path=file_name,
            file_size=file_size
        )
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress a file using gzip."""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        file_path.unlink()
    
    def _create_metadata(
        self,
        sample: LabeledSample,
        features: ExtractedFeatures,
        feature_file: str,
        annotation_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create metadata entry for a sample."""
        metadata = {
            "sample_id": sample.id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "model_answer": sample.answer,
            "context": sample.context,
            "task_type": sample.task_type.value if hasattr(sample.task_type, 'value') else str(sample.task_type),
            "split": sample.split.value if sample.split and hasattr(sample.split, 'value') else str(sample.split or ""),
            "label": sample.label,
            "has_hallucination": sample.has_hallucination,
            "hallucination_spans": [
                {
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "label_type": span.label_type.value if hasattr(span.label_type, 'value') else str(span.label_type),
                }
                for span in sample.hallucination_spans
            ],
            "source_model": sample.source_model,
            "feature_file": feature_file,
            "extraction_mode": features.metadata.get("mode", "unknown"),
            "attention_layers_used": features.layers_extracted or [],
            "prompt_length": features.prompt_length,
            "response_length": features.response_length,
        }
        
        # Add annotation data if available
        if annotation_data:
            metadata["token_annotations"] = annotation_data.get("token_annotations", [])
            metadata["annotation_method"] = annotation_data.get("method", "unknown")
            metadata["annotation_hallucination_ratio"] = annotation_data.get("hallucination_ratio", 0)
        
        return metadata
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize output and write summary files.
        
        Returns:
            Summary statistics
        """
        if self._finalized:
            return {}
        
        # Write metadata.jsonl
        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for entry in self._metadata_entries:
                f.write(json.dumps(entry, default=str) + '\n')
        
        # Write index.json
        index_path = self.output_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(self._feature_index, f, indent=2)
        
        # Compute and write summary
        summary = self._compute_summary()
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._finalized = True
        logger.info(f"Finalized output: {len(self._metadata_entries)} samples")
        
        return summary
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute extraction summary statistics."""
        total = len(self._metadata_entries)
        
        if total == 0:
            return {"total_samples": 0}
        
        hallucinated = sum(1 for m in self._metadata_entries if m.get("has_hallucination"))
        
        return {
            "total_samples": total,
            "hallucinated_samples": hallucinated,
            "clean_samples": total - hallucinated,
            "hallucination_rate": hallucinated / total,
            "output_dir": str(self.output_dir),
            "metadata_file": "metadata.jsonl",
            "index_file": "index.json",
        }


def write_metadata_jsonl(
    entries: List[MetadataEntry],
    output_path: Union[str, Path],
) -> str:
    """Write metadata entries to JSONL file.
    
    Args:
        entries: List of MetadataEntry objects
        output_path: Output file path
        
    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for entry in entries:
            data = entry.model_dump() if hasattr(entry, 'model_dump') else entry
            f.write(json.dumps(data, default=str) + '\n')
    
    return str(output_path)


def write_features_batch(
    features_list: List[ExtractedFeatures],
    output_dir: Union[str, Path],
    compress: bool = False,
) -> List[str]:
    """Write batch of features to files.
    
    Args:
        features_list: List of ExtractedFeatures
        output_dir: Output directory
        compress: Whether to compress files
        
    Returns:
        List of written file names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_names = []
    
    try:
        from tqdm import tqdm
        iterator = tqdm(features_list, desc="Writing features")
    except ImportError:
        iterator = features_list
    
    for features in iterator:
        writer = OutputWriter(output_dir, compress=compress)
        writer.initialize()
        
        # Create minimal sample for writing
        from src.core import LabeledSample
        sample = LabeledSample(
            id=features.sample_id,
            question="",
            answer="",
            label=features.metadata.get("label", 0),
            has_hallucination=features.metadata.get("label", 0) == 1,
        )
        
        result = writer.write_sample(sample, features)
        if result.success:
            file_names.append(result.file_path)
    
    return file_names


def create_output_structure(
    base_dir: Union[str, Path],
    dataset: str,
    model: str,
    mode: str,
    split: str,
    task_type: Optional[str] = None,
) -> Path:
    """Create organized output directory structure.
    
    Args:
        base_dir: Base output directory
        dataset: Dataset name
        model: Model name (will be sanitized)
        mode: Extraction mode
        split: Data split
        task_type: Optional task type
        
    Returns:
        Created output directory path
    """
    base_dir = Path(base_dir)
    
    # Sanitize model name
    model_safe = model.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    # Build path
    dir_name = f"{dataset}_{model_safe}_{mode}_{split}"
    if task_type:
        dir_name = f"{dir_name}_{task_type}"
    
    output_path = base_dir / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "features").mkdir(exist_ok=True)
    
    return output_path
