#!/usr/bin/env python
"""Training script for hallucination detection methods."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    init_config, setup_logging, get_logger,
    FeatureCache, ExtractedFeatures,
)
from src.methods import create_method, list_methods

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hallucination detection methods"
    )
    parser.add_argument("--config", "-c", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=["lapeigvals"])
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_features_and_labels(
    features_dir: Path
) -> Tuple[List[ExtractedFeatures], List[int]]:
    """Load extracted features and labels from cache."""
    cache = FeatureCache(features_dir, enabled=True)
    
    index_path = features_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    features = []
    labels = []
    
    for item in index:
        sample_id = item["sample_id"]
        cache_key = f"{sample_id}_teacher_forcing"
        
        cached = cache.get(cache_key)
        if cached is None:
            cache_key = f"{sample_id}_generation"
            cached = cache.get(cache_key)
        
        if cached:
            features.append(cached)
            label = item.get("label")
            if label is None:
                label = cached.metadata.get("label", 0)
            labels.append(int(label) if label is not None else 0)
    
    return features, labels


def split_data(
    features: List[ExtractedFeatures],
    labels: List[int],
    val_split: float,
    seed: int
) -> Tuple[List[ExtractedFeatures], List[int], List[ExtractedFeatures], List[int]]:
    """Split data into train and validation sets."""
    import numpy as np
    
    np.random.seed(seed)
    n = len(features)
    indices = np.random.permutation(n)
    
    split_idx = int(n * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_features = [features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_features = [features[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    return train_features, train_labels, val_features, val_labels


def main() -> None:
    args = parse_args()
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    config = init_config(args.config)
    
    logger.info(f"Loading features from {args.features_dir}")
    features, labels = load_features_and_labels(args.features_dir)
    logger.info(f"Loaded {len(features)} samples")
    
    label_dist = {str(l): labels.count(l) for l in set(labels)}
    logger.info(f"Label distribution: {label_dist}")
    
    train_features, train_labels, val_features, val_labels = split_data(
        features, labels, args.val_split, args.random_seed
    )
    logger.info(f"Train: {len(train_features)}, Val: {len(val_features)}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    available_methods = list_methods()
    logger.info(f"Available methods: {available_methods}")
    
    results = {}
    
    for method_name in args.methods:
        if method_name not in available_methods:
            logger.warning(f"Unknown method: {method_name}, skipping")
            continue
        
        logger.info(f"Training {method_name}...")
        
        try:
            method = create_method(method_name)
            
            metrics = method.fit(
                train_features,
                train_labels,
                val_features,
                val_labels
            )
            
            method_dir = output_path / method_name
            method.save(method_dir)
            
            results[method_name] = metrics.to_dict()
            
            logger.info(f"{method_name} results:")
            logger.info(f"  AUROC: {metrics.auroc:.4f}")
            logger.info(f"  AUPRC: {metrics.auprc:.4f}")
            logger.info(f"  F1: {metrics.f1:.4f}")
            logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {method_name}: {e}")
            results[method_name] = {"error": str(e)}
    
    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    training_config = {
        "features_dir": str(args.features_dir),
        "methods": args.methods,
        "val_split": args.val_split,
        "random_seed": args.random_seed,
        "n_train": len(train_features),
        "n_val": len(val_features),
        "label_distribution": label_dist,
    }
    
    with open(output_path / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"Training complete. Models saved to {output_path}")
    
    if results:
        best_method = max(
            [(k, v.get("auroc", 0)) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1],
            default=(None, 0)
        )
        if best_method[0]:
            logger.info(f"Best method: {best_method[0]} (AUROC={best_method[1]:.4f})")


if __name__ == "__main__":
    main()
