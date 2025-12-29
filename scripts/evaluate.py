#!/usr/bin/env python
"""Evaluation script for hallucination detection methods."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    init_config, setup_logging, get_logger,
    FeatureCache, ExtractedFeatures, EvaluationConfig,
)
from src.methods import create_method
from src.evaluation import Evaluator, EvaluationResult

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection methods"
    )
    parser.add_argument("--config", "-c", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--models_dir", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_test_data(features_dir: Path) -> tuple:
    """Load test features and labels."""
    cache = FeatureCache(features_dir, enabled=True)
    
    with open(features_dir / "index.json") as f:
        index = json.load(f)
    
    features = []
    labels = []
    
    for item in index:
        sample_id = item["sample_id"]
        
        for mode in ["teacher_forcing", "generation"]:
            cached = cache.get(f"{sample_id}_{mode}")
            if cached:
                features.append(cached)
                label = item.get("label")
                if label is None:
                    label = cached.metadata.get("label", 0)
                labels.append(int(label) if label is not None else 0)
                break
    
    return features, labels


def main() -> None:
    args = parse_args()
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    config = init_config(args.config)
    
    logger.info(f"Loading test data from {args.features_dir}")
    test_features, test_labels = load_test_data(args.features_dir)
    logger.info(f"Loaded {len(test_features)} test samples")
    
    label_dist = {str(l): test_labels.count(l) for l in set(test_labels)}
    logger.info(f"Label distribution: {label_dist}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    eval_config = EvaluationConfig(
        threshold=args.threshold or config.get("evaluation.threshold", 0.5),
        bootstrap_samples=args.bootstrap,
        confidence_level=config.get("evaluation.confidence_level", 0.95),
    )
    evaluator = Evaluator(eval_config)
    
    all_results: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, List] = {}
    
    models_dir = Path(args.models_dir)
    
    for method_dir in sorted(models_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        
        method_name = method_dir.name
        logger.info(f"Evaluating {method_name}...")
        
        try:
            method = create_method(method_name)
            method.load(method_dir)
            
            predictions = method.predict_batch(test_features)
            all_predictions[method_name] = predictions
            
            result = evaluator.evaluate(predictions, test_labels)
            
            optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
                predictions, test_labels, metric="f1"
            )
            
            result_dict = result.to_dict()
            result_dict["optimal_threshold"] = optimal_threshold
            result_dict["optimal_f1"] = optimal_f1
            
            all_results[method_name] = result_dict
            
            logger.info(f"{method_name} results:")
            logger.info(f"  AUROC: {result.auroc:.4f}")
            logger.info(f"  AUPRC: {result.auprc:.4f}")
            logger.info(f"  F1: {result.f1:.4f}")
            logger.info(f"  Accuracy: {result.accuracy:.4f}")
            
            if result.confidence_intervals:
                ci = result.confidence_intervals.get("auroc")
                if ci:
                    logger.info(f"  AUROC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {method_name}: {e}")
            all_results[method_name] = {"error": str(e)}
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    roc_curves = {}
    pr_curves = {}
    
    for method_name, result in all_results.items():
        if "error" in result:
            continue
        
        if "roc_curve" in result and result["roc_curve"]:
            roc_curves[method_name] = result["roc_curve"]
        
        if "pr_curve" in result and result["pr_curve"]:
            pr_curves[method_name] = result["pr_curve"]
    
    if roc_curves:
        with open(output_path / "roc_curves.json", "w") as f:
            json.dump(roc_curves, f, indent=2)
    
    if pr_curves:
        with open(output_path / "pr_curves.json", "w") as f:
            json.dump(pr_curves, f, indent=2)
    
    comparison_data = []
    for method_name, result in all_results.items():
        if "error" in result:
            continue
        comparison_data.append({
            "method": method_name,
            "auroc": result.get("auroc", 0),
            "auprc": result.get("auprc", 0),
            "f1": result.get("f1", 0),
            "accuracy": result.get("accuracy", 0),
            "precision": result.get("precision", 0),
            "recall": result.get("recall", 0),
        })
    
    if comparison_data:
        import csv
        with open(output_path / "comparison.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
            writer.writeheader()
            writer.writerows(comparison_data)
    
    eval_summary = {
        "n_test_samples": len(test_features),
        "label_distribution": label_dist,
        "n_methods_evaluated": len([r for r in all_results.values() if "error" not in r]),
        "threshold": eval_config.threshold,
        "bootstrap_samples": eval_config.bootstrap_samples,
    }
    
    with open(output_path / "evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1].get("auroc", 0))
        logger.info(f"Best method: {best[0]} (AUROC={best[1]['auroc']:.4f})")
        
        logger.info("\nMethod Comparison:")
        logger.info("-" * 60)
        logger.info(f"{'Method':<20} {'AUROC':>10} {'AUPRC':>10} {'F1':>10}")
        logger.info("-" * 60)
        for method, result in sorted(
            valid_results.items(),
            key=lambda x: x[1].get("auroc", 0),
            reverse=True
        ):
            logger.info(
                f"{method:<20} {result.get('auroc', 0):>10.4f} "
                f"{result.get('auprc', 0):>10.4f} {result.get('f1', 0):>10.4f}"
            )


if __name__ == "__main__":
    main()
