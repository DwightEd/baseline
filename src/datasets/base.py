"""Dataset parsing interfaces and base implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any, Union, Callable
import json
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    BaseComponent, DATASET_REGISTRY,
    DataSample, LabeledSample, TaskType, SplitType,
    DatasetError, DatasetNotFoundError, DatasetParseError,
    get_logger, ProgressLogger
)

logger = get_logger(__name__)


class BaseDatasetParser(BaseComponent, ABC):
    """Abstract base class for dataset parsers."""
    
    def __init__(self, path: Path, **kwargs: Any):
        self.path = Path(path)
        self.kwargs = kwargs
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that the dataset path exists."""
        if not self.path.exists():
            raise DatasetNotFoundError(
                f"Dataset path not found: {self.path}",
                details={"path": str(self.path), "parser": self.component_name()}
            )
    
    @classmethod
    @abstractmethod
    def component_name(cls) -> str:
        """Return component name for registration."""
        pass
    
    @abstractmethod
    def parse(self) -> Iterator[DataSample]:
        """Parse dataset and yield standardized samples."""
        pass
    
    def load_all(self, max_samples: Optional[int] = None) -> List[DataSample]:
        """Load all samples into memory."""
        samples = []
        for i, sample in enumerate(self.parse()):
            if max_samples and i >= max_samples:
                break
            samples.append(sample)
        return samples
    
    def filter(
        self,
        split: Optional[SplitType] = None,
        task_type: Optional[TaskType] = None,
        predicate: Optional[Callable[[DataSample], bool]] = None,
    ) -> Iterator[DataSample]:
        """Filter samples by criteria."""
        for sample in self.parse():
            if split and sample.split != split:
                continue
            if task_type and sample.task_type != task_type:
                continue
            if predicate and not predicate(sample):
                continue
            yield sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "total": 0,
            "by_split": {},
            "by_task_type": {},
            "by_label": {},
            "avg_question_length": 0,
            "avg_answer_length": 0,
        }
        
        total_q_len = 0
        total_a_len = 0
        
        for sample in self.parse():
            stats["total"] += 1
            total_q_len += len(sample.question)
            total_a_len += len(sample.answer)
            
            if sample.split:
                key = sample.split.value
                stats["by_split"][key] = stats["by_split"].get(key, 0) + 1
            
            key = sample.task_type.value
            stats["by_task_type"][key] = stats["by_task_type"].get(key, 0) + 1
            
            if sample.label is not None:
                label_key = str(sample.label)
                stats["by_label"][label_key] = stats["by_label"].get(label_key, 0) + 1
        
        if stats["total"] > 0:
            stats["avg_question_length"] = total_q_len / stats["total"]
            stats["avg_answer_length"] = total_a_len / stats["total"]
        
        return stats


@DATASET_REGISTRY.register("json")
class JsonDatasetParser(BaseDatasetParser):
    """Parser for JSON format datasets."""
    
    @classmethod
    def component_name(cls) -> str:
        return "json"
    
    def __init__(
        self,
        path: Path,
        field_mapping: Optional[Dict[str, str]] = None,
        data_key: Optional[str] = None,
        task_type: TaskType = TaskType.QA,
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.field_mapping = field_mapping or {}
        self.data_key = data_key
        self.task_type = task_type
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if self.data_key:
            data = data.get(self.data_key, [])
        
        if isinstance(data, dict):
            data = [data]
        
        for idx, item in enumerate(data):
            yield self._parse_item(item, idx)
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> DataSample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped_key = self.field_mapping.get(key, key)
            return item.get(mapped_key, default)
        
        return DataSample(
            id=str(get_field("id", idx)),
            question=str(get_field("question", "")),
            answer=str(get_field("answer", "")),
            gold_answer=str(get_field("gold_answer", get_field("reference", ""))),
            context=get_field("context") or None,
            task_type=self.task_type,
            label=get_field("label") if "label" in item or self.field_mapping.get("label") in item else None,
            metadata={"raw": item},
        )


@DATASET_REGISTRY.register("jsonl", aliases=["jsonlines"])
class JsonlDatasetParser(BaseDatasetParser):
    """Parser for JSON Lines format datasets."""
    
    @classmethod
    def component_name(cls) -> str:
        return "jsonl"
    
    def __init__(
        self,
        path: Path,
        field_mapping: Optional[Dict[str, str]] = None,
        task_type: TaskType = TaskType.QA,
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.field_mapping = field_mapping or {}
        self.task_type = task_type
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield self._parse_item(item, idx)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {idx}: {e}")
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> DataSample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped_key = self.field_mapping.get(key, key)
            return item.get(mapped_key, default)
        
        return DataSample(
            id=str(get_field("id", idx)),
            question=str(get_field("question", "")),
            answer=str(get_field("answer", "")),
            gold_answer=str(get_field("gold_answer", "")),
            context=get_field("context") or None,
            task_type=self.task_type,
            label=get_field("label") if "label" in item else None,
            metadata={"raw": item},
        )


@DATASET_REGISTRY.register("csv", aliases=["tsv"])
class CsvDatasetParser(BaseDatasetParser):
    """Parser for CSV/TSV format datasets."""
    
    @classmethod
    def component_name(cls) -> str:
        return "csv"
    
    def __init__(
        self,
        path: Path,
        field_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = ",",
        task_type: TaskType = TaskType.QA,
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.field_mapping = field_mapping or {}
        self.delimiter = delimiter
        self.task_type = task_type
        
        if str(path).endswith('.tsv'):
            self.delimiter = "\t"
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for idx, row in enumerate(reader):
                yield self._parse_item(row, idx)
    
    def _parse_item(self, row: Dict[str, str], idx: int) -> DataSample:
        def get_field(key: str, default: str = "") -> str:
            mapped_key = self.field_mapping.get(key, key)
            return row.get(mapped_key, default)
        
        label = None
        label_str = get_field("label")
        if label_str and label_str.isdigit():
            label = int(label_str)
        
        return DataSample(
            id=get_field("id", str(idx)),
            question=get_field("question"),
            answer=get_field("answer"),
            gold_answer=get_field("gold_answer"),
            context=get_field("context") or None,
            task_type=self.task_type,
            label=label,
            metadata={"raw": dict(row)},
        )


@DATASET_REGISTRY.register("parquet")
class ParquetDatasetParser(BaseDatasetParser):
    """Parser for Parquet format datasets."""
    
    @classmethod
    def component_name(cls) -> str:
        return "parquet"
    
    def __init__(
        self,
        path: Path,
        field_mapping: Optional[Dict[str, str]] = None,
        task_type: TaskType = TaskType.QA,
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.field_mapping = field_mapping or {}
        self.task_type = task_type
    
    def parse(self) -> Iterator[DataSample]:
        try:
            import pandas as pd
        except ImportError:
            raise DatasetError("pandas is required for Parquet support")
        
        df = pd.read_parquet(self.path)
        
        for idx, row in df.iterrows():
            yield self._parse_item(row.to_dict(), idx)
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> DataSample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped_key = self.field_mapping.get(key, key)
            value = item.get(mapped_key, default)
            if hasattr(value, 'item'):
                value = value.item()
            return value
        
        return DataSample(
            id=str(get_field("id", idx)),
            question=str(get_field("question", "")),
            answer=str(get_field("answer", "")),
            gold_answer=str(get_field("gold_answer", "")),
            context=get_field("context") or None,
            task_type=self.task_type,
            metadata={"raw": item},
        )


def create_parser(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs: Any
) -> BaseDatasetParser:
    """Factory function to create appropriate parser based on file format."""
    path = Path(path)
    
    if format is None:
        suffix = path.suffix.lower()
        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }
        format = format_map.get(suffix)
        if format is None:
            raise DatasetError(f"Unknown file format: {suffix}")
        
        if suffix == ".tsv":
            kwargs.setdefault("delimiter", "\t")
    
    return DATASET_REGISTRY.create(format, path=path, **kwargs)


def load_dataset(
    path: Union[str, Path],
    format: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs: Any
) -> List[DataSample]:
    """Convenience function to load dataset into memory."""
    parser = create_parser(path, format, **kwargs)
    return parser.load_all(max_samples=max_samples)
