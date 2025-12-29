"""Dataset parsing module with unified interface for multiple formats."""
from .base import (
    BaseDatasetParser,
    JsonDatasetParser,
    JsonlDatasetParser,
    CsvDatasetParser,
    ParquetDatasetParser,
    create_parser,
    load_dataset,
)

from .ragtruth import RAGTruthParser

from .parsers import (
    GSM8KParser,
    TriviaQAParser,
    TruthfulQAParser,
    HaluEvalParser,
    CoQAParser,
    NQOpenParser,
    SQuADv2Parser,
    CustomParser,
)

__all__ = [
    # Base classes and utilities
    "BaseDatasetParser",
    "JsonDatasetParser",
    "JsonlDatasetParser",
    "CsvDatasetParser",
    "ParquetDatasetParser",
    "create_parser",
    "load_dataset",
    
    # Specialized parsers
    "RAGTruthParser",
    "GSM8KParser",
    "TriviaQAParser",
    "TruthfulQAParser",
    "HaluEvalParser",
    "CoQAParser",
    "NQOpenParser",
    "SQuADv2Parser",
    "CustomParser",
]
