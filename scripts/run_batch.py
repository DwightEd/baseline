#!/usr/bin/env python
"""One-click batch processing for RAGTruth hallucination detection.

This script processes all RAGTruth task_types (QA, Summary, Data2txt)
and splits (train, test) in a single command with:
- Flexible attention layer selection
- Lookback Lens hallucination annotation
- lapeigvals-compatible metadata.jsonl output
- tqdm progress bars at all stages

Usage:
    # Process all task types with first layer attention
    python scripts/run_batch.py --data_path ./data/ragtruth --attention_layers first
    
    # Process only QA task with specific layers
    python scripts/run_batch.py --data_path ./data/ragtruth --task_types QA --attention_layers 0,1,2
    
    # Quick test with limited samples
    python scripts/run_batch.py --data_path ./data/ragtruth --max_samples 10
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import setup_logging, get_logger, ExtractionMode
from src.pipelines import BatchProcessor, process_all_tasks

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click batch processing for RAGTruth hallucination detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all task types with first layer attention
  python scripts/run_batch.py --data_path ./data/ragtruth --attention_layers first
  
  # Process only QA task type
  python scripts/run_batch.py --data_path ./data/ragtruth --task_types QA
  
  # Process with specific attention layers
  python scripts/run_batch.py --data_path ./data/ragtruth --attention_layers 0,4,8
  
  # Quick test with limited samples
  python scripts/run_batch.py --data_path ./data/ragtruth --max_samples 10
  
  # Disable annotation for faster processing
  python scripts/run_batch.py --data_path ./data/ragtruth --no_annotation
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        required=True,
        help="Path to RAGTruth dataset directory"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name/path to use for feature extraction"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Base output directory"
    )
    
    # Processing mode
    parser.add_argument(
        "--mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "generation"],
        help="Feature extraction mode"
    )
    
    # Task type filtering
    parser.add_argument(
        "--task_types", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Task types to process (default: all). Options: QA, Summary, Data2txt"
    )
    
    # Split filtering
    parser.add_argument(
        "--splits", "-s",
        type=str,
        nargs="+",
        default=None,
        help="Splits to process (default: all). Options: train, test"
    )
    
    # Sample limiting
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per task/split combination (for testing)"
    )
    
    # Attention layer selection
    parser.add_argument(
        "--attention_layers", "-a",
        type=str,
        default="first",
        help="Attention layers: 'all', 'first', 'last', 'first_n', 'last_n', or indices like '0,1,2'"
    )
    
    # Annotation settings
    parser.add_argument(
        "--no_annotation",
        action="store_true",
        help="Disable Lookback Lens hallucination annotation"
    )
    parser.add_argument(
        "--annotation_threshold",
        type=float,
        default=0.5,
        help="Threshold for hallucination detection (0-1)"
    )
    
    # Progress and logging
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def parse_attention_layers(layers_str: str):
    """Parse attention layers argument."""
    layers_str = layers_str.strip().lower()
    
    if layers_str in ['all', 'first', 'last', 'first_n', 'last_n']:
        return layers_str
    
    # Try to parse as comma-separated indices
    try:
        indices = [int(x.strip()) for x in layers_str.split(',')]
        return indices
    except ValueError:
        logger.warning(f"Invalid attention_layers '{layers_str}', using 'first'")
        return "first"


def main() -> None:
    args = parse_args()
    
    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    logger.info("=" * 60)
    logger.info("RAGTruth Batch Processing")
    logger.info("=" * 60)
    
    # Parse attention layers
    attention_layers = parse_attention_layers(args.attention_layers)
    
    # Determine task types and splits
    task_types = args.task_types if args.task_types else "all"
    splits = args.splits if args.splits else "all"
    
    logger.info(f"Configuration:")
    logger.info(f"  - Data path: {args.data_path}")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Mode: {args.mode}")
    logger.info(f"  - Task types: {task_types}")
    logger.info(f"  - Splits: {splits}")
    logger.info(f"  - Attention layers: {attention_layers}")
    logger.info(f"  - Annotation: {'disabled' if args.no_annotation else 'enabled'}")
    logger.info(f"  - Output: {args.output_dir}")
    
    if args.max_samples:
        logger.info(f"  - Max samples per task: {args.max_samples}")
    
    logger.info("")
    
    # Create batch processor
    processor = BatchProcessor(
        data_path=args.data_path,
        model_name=args.model,
        output_dir=args.output_dir,
        mode=ExtractionMode(args.mode),
        attention_layers=attention_layers,
        task_types=task_types,
        splits=splits,
        max_samples_per_task=args.max_samples,
        enable_annotation=not args.no_annotation,
        annotation_threshold=args.annotation_threshold,
        show_progress=not args.no_progress,
    )
    
    # Run batch processing
    try:
        results = processor.process_all()
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Processing Complete!")
        logger.info("=" * 60)
        
        total_processed = sum(r.processed_samples for r in results)
        total_errors = sum(r.error_samples for r in results)
        total_time = sum(r.processing_time for r in results)
        
        logger.info(f"Summary:")
        logger.info(f"  - Total processed: {total_processed}")
        logger.info(f"  - Total errors: {total_errors}")
        logger.info(f"  - Total time: {total_time:.1f}s")
        logger.info("")
        
        for r in results:
            status = "✓" if r.error_samples == 0 else "⚠"
            logger.info(
                f"  {status} {r.task_type}/{r.split}: "
                f"{r.processed_samples}/{r.total_samples} samples, "
                f"{r.processing_time:.1f}s"
            )
        
        logger.info("")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Batch summary: {args.output_dir}/batch_summary.json")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
