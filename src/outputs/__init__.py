"""Output management module.

This module provides utilities for:
- Writing features and metadata in lapeigvals-compatible format
- Organizing output directories by dataset/model/mode/split
- Managing feature file serialization
"""
from .writer import (
    OutputWriter,
    write_metadata_jsonl,
    write_features_batch,
    create_output_structure,
)

__all__ = [
    "OutputWriter",
    "write_metadata_jsonl",
    "write_features_batch",
    "create_output_structure",
]
