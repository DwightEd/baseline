"""Pipeline module for batch processing and orchestration.

This module provides:
- BatchProcessor: One-click batch processing of all RAGTruth task_types
- Progress tracking with tqdm integration
- Organized output path generation
"""
from .batch_processor import (
    BatchProcessor,
    BatchProcessResult,
    process_all_tasks,
    create_batch_processor,
)

__all__ = [
    "BatchProcessor",
    "BatchProcessResult",
    "process_all_tasks",
    "create_batch_processor",
]
