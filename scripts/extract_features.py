#!/usr/bin/env python
"""Feature extraction script for hallucination detection.

Enhanced with:
- Flexible attention layer selection (--attention_layers)
- tqdm progress bars
- Organized output path generation
- layers_extracted tracking
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    init_config, setup_logging, get_logger,
    FeatureExtractionConfig, ExtractionMode, AttentionConfig,
    DataSample, LabeledSample, FeatureCache, OutputConfig, MetadataEntry,
)
from src.datasets import RAGTruthParser, create_parser
from src.models import load_model, ModelConfig
from src.features import create_extractor
from src.outputs import OutputWriter

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from LLM for hallucination detection"
    )
    parser.add_argument("--config", "-c", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--task_types", type=str, nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("./cache"))
    parser.add_argument("--mode", type=str, default="teacher_forcing",
                        choices=["teacher_forcing", "generation"])
    parser.add_argument("--model", type=str, default=None)
    
    # NEW: Attention layer selection
    parser.add_argument("--attention_layers", type=str, default="all",
                        help="Attention layers to extract: 'all', 'first', 'last', 'first_n', 'last_n', or comma-separated indices (e.g., '0,1,2')")
    parser.add_argument("--first_n", type=int, default=1,
                        help="Number of layers for 'first_n' mode")
    parser.add_argument("--last_n", type=int, default=4,
                        help="Number of layers for 'last_n' mode")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    return parser.parse_args()


def parse_attention_layers(layers_str: str) -> any:
    """Parse attention layers argument.
    
    Args:
        layers_str: String like 'all', 'first', '0,1,2', etc.
        
    Returns:
        Layer specification for AttentionConfig
    """
    layers_str = layers_str.strip().lower()
    
    if layers_str in ['all', 'first', 'last', 'first_n', 'last_n']:
        return layers_str
    
    # Try to parse as comma-separated indices
    try:
        indices = [int(x.strip()) for x in layers_str.split(',')]
        return indices
    except ValueError:
        logger.warning(f"Invalid attention_layers '{layers_str}', using 'all'")
        return "all"


def load_samples(args: argparse.Namespace) -> List[DataSample]:
    """Load dataset samples."""
    logger.info(f"Loading dataset: {args.dataset} from {args.data_path}")
    
    if args.dataset.lower() == "ragtruth":
        parser = RAGTruthParser(
            path=Path(args.data_path),
            split=args.split,
            task_types=args.task_types,
        )
    else:
        parser = create_parser(
            Path(args.data_path),
        )
    
    samples = parser.load_all(max_samples=args.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    stats = parser.get_statistics()
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2, default=str)}")
    
    return samples


def main() -> None:
    args = parse_args()
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    config = init_config(args.config)
    
    samples = load_samples(args)
    
    if not samples:
        logger.error("No samples loaded. Exiting.")
        return
    
    model_name = args.model or config.get("model.name", "meta-llama/Llama-3.1-8B-Instruct")
    logger.info(f"Loading model: {model_name}")
    
    model_config = ModelConfig(
        name=model_name,
        dtype=config.get("model.dtype", "float16"),
        device_map=config.get("model.device_map", "auto"),
        attn_implementation="eager",
        load_in_4bit=config.get("model.load_in_4bit", False),
    )
    model = load_model(model_config)
    
    # Parse attention layers
    attention_layers = parse_attention_layers(args.attention_layers)
    logger.info(f"Attention layers: {attention_layers}")
    
    # Build attention config with layer selection
    attn_config = AttentionConfig(
        enabled=True,
        layers=attention_layers,
        first_n=args.first_n,
        last_n=args.last_n,
        compute_eigenvalues=True,
        compute_laplacian=True,
    )
    
    feat_config_dict = config.get("feature_extraction", {})
    feat_config_dict["mode"] = ExtractionMode(args.mode)
    feat_config_dict["attention"] = attn_config
    feat_config = FeatureExtractionConfig(**feat_config_dict)
    
    extractor = create_extractor(feat_config)
    
    # Generate organized output path
    output_config = OutputConfig(base_dir=args.output_dir)
    output_path = output_config.get_output_path(
        dataset=args.dataset,
        model=model_name,
        mode=args.mode,
        split=args.split or "all",
        task_type=args.task_types[0] if args.task_types and len(args.task_types) == 1 else None
    )
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache = FeatureCache(
        output_path,
        enabled=not args.no_cache
    )
    
    logger.info(f"Extracting features (mode: {args.mode})...")
    logger.info(f"Output directory: {output_path}")
    
    # Use tqdm for progress if available and enabled
    show_progress = not args.no_progress
    
    index_data = []
    error_count = 0
    layers_used = set()
    
    for features in extractor.extract_batch(model, samples, cache=cache, show_progress=show_progress):
        if features.metadata.get("error"):
            error_count += 1
            continue
        
        # Track layers used
        if features.layers_extracted:
            layers_used.update(features.layers_extracted)
        
        index_data.append({
            "sample_id": features.sample_id,
            "prompt_length": features.prompt_length,
            "response_length": features.response_length,
            "total_length": features.total_length,
            "label": features.metadata.get("label"),
            "layers_extracted": features.layers_extracted,
        })
    
    with open(output_path / "index.json", "w") as f:
        json.dump(index_data, f, indent=2)
    
    # Write metadata.jsonl (lapeigvals format)
    logger.info("Writing metadata.jsonl...")
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for i, sample in enumerate(samples):
            if i >= len(index_data):
                break
            idx_entry = index_data[i]
            
            # Create metadata entry
            metadata = {
                "sample_id": sample.id,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "model_answer": sample.answer,
                "context": sample.context,
                "task_type": sample.task_type.value if hasattr(sample.task_type, 'value') else str(sample.task_type),
                "split": sample.split.value if sample.split and hasattr(sample.split, 'value') else str(sample.split or args.split or ""),
                "label": sample.label or 0,
                "has_hallucination": getattr(sample, 'has_hallucination', sample.label == 1),
                "hallucination_spans": [
                    {"text": s.text, "start": s.start, "end": s.end, "label_type": s.label_type.value}
                    for s in getattr(sample, 'hallucination_spans', [])
                ],
                "source_model": getattr(sample, 'source_model', None),
                "feature_file": f"{sample.id}.pt",
                "extraction_mode": args.mode,
                "attention_layers_used": idx_entry.get("layers_extracted", []),
                "prompt_length": idx_entry.get("prompt_length", 0),
                "response_length": idx_entry.get("response_length", 0),
            }
            f.write(json.dumps(metadata, default=str) + '\n')
    
    logger.info(f"Wrote metadata to {metadata_path}")
    
    summary = {
        "total_samples": len(samples),
        "extracted": len(index_data),
        "errors": error_count,
        "mode": args.mode,
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "task_types": args.task_types,
        "attention_layers_config": str(attention_layers),
        "layers_used": sorted(list(layers_used)),
    }
    
    with open(output_path / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Feature extraction complete:")
    logger.info(f"  - Extracted: {len(index_data)}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"  - Layers used: {sorted(list(layers_used))}")
    logger.info(f"  - Output: {output_path}")
    logger.info(f"  - Cache stats: {cache.stats}")


if __name__ == "__main__":
    main()
