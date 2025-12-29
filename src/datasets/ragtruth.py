"""RAGTruth dataset parser implementation."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import (
    DATASET_REGISTRY,
    DataSample, LabeledSample, HallucinationSpan,
    TaskType, SplitType, HallucinationType,
    DatasetError, DatasetNotFoundError,
    get_logger
)
from .base import BaseDatasetParser

logger = get_logger(__name__)


HALLUCINATION_TYPE_MAP = {
    "Evident Conflict": HallucinationType.EVIDENT_CONFLICT,
    "Evident Baseless Info": HallucinationType.EVIDENT_BASELESS,
    "Subtle Conflict": HallucinationType.SUBTLE_CONFLICT,
    "Subtle Baseless Info": HallucinationType.SUBTLE_BASELESS,
}

TASK_TYPE_MAP = {
    "QA": TaskType.QA,
    "Summary": TaskType.SUMMARY,
    "Data2txt": TaskType.DATA2TXT,
}


@DATASET_REGISTRY.register("ragtruth", aliases=["rag_truth", "RAGTruth"])
class RAGTruthParser(BaseDatasetParser):
    """Parser for RAGTruth hallucination detection dataset.
    
    RAGTruth dataset structure:
    - response.jsonl: Model responses with hallucination labels
    - source_info.jsonl: Source information for each response
    """
    
    @classmethod
    def component_name(cls) -> str:
        return "ragtruth"
    
    def __init__(
        self,
        path: Path,
        split: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        exclude_quality: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """Initialize RAGTruth parser.
        
        Args:
            path: Path to RAGTruth dataset directory
            split: Filter by split ('train' or 'test')
            task_types: Filter by task types (e.g., ['QA', 'Summary'])
            models: Filter by source models
            exclude_quality: Quality labels to exclude (default: ['incorrect_refusal', 'truncated'])
        """
        self.data_dir = Path(path)
        super().__init__(self.data_dir, **kwargs)
        
        self.split_filter = split
        self.task_type_filter = set(task_types) if task_types else None
        self.model_filter = set(models) if models else None
        self.exclude_quality = set(exclude_quality or ["incorrect_refusal", "truncated"])
        
        self._source_map: Optional[Dict[str, Dict[str, Any]]] = None
    
    def _validate_path(self) -> None:
        """Validate RAGTruth directory structure."""
        response_file = self.data_dir / "response.jsonl"
        source_file = self.data_dir / "source_info.jsonl"
        
        if not response_file.exists():
            raise DatasetNotFoundError(
                f"RAGTruth response file not found: {response_file}",
                details={"expected_file": str(response_file)}
            )
        if not source_file.exists():
            raise DatasetNotFoundError(
                f"RAGTruth source file not found: {source_file}",
                details={"expected_file": str(source_file)}
            )
    
    def _load_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Load and cache source information."""
        if self._source_map is not None:
            return self._source_map
        
        self._source_map = {}
        source_file = self.data_dir / "source_info.jsonl"
        
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    source_id = item.get("source_id")
                    if source_id:
                        self._source_map[source_id] = item
        
        logger.debug(f"Loaded {len(self._source_map)} source entries")
        return self._source_map
    
    def parse(self) -> Iterator[LabeledSample]:
        """Parse RAGTruth dataset and yield labeled samples."""
        source_map = self._load_source_info()
        response_file = self.data_dir / "response.jsonl"
        
        with open(response_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                sample = self._parse_response(item, source_map)
                
                if sample and self._should_include(sample, item):
                    yield sample
    
    def _parse_response(
        self,
        item: Dict[str, Any],
        source_map: Dict[str, Dict[str, Any]]
    ) -> Optional[LabeledSample]:
        """Parse a single response entry."""
        sample_id = item.get("id", "")
        source_id = item.get("source_id", "")
        source_info = source_map.get(source_id, {})
        
        if not source_info:
            logger.warning(f"No source info for sample {sample_id}")
            return None
        
        task_type_str = source_info.get("task_type", "QA")
        task_type = TASK_TYPE_MAP.get(task_type_str, TaskType.QA)
        
        question, context = self._extract_question_context(source_info, task_type)
        
        labels = item.get("labels", [])
        hallucination_spans = [
            HallucinationSpan(
                text=lbl.get("text", ""),
                start=lbl.get("start", 0),
                end=lbl.get("end", 0),
                label_type=HALLUCINATION_TYPE_MAP.get(
                    lbl.get("label_type", ""),
                    HallucinationType.UNKNOWN
                ),
                metadata={
                    k: v for k, v in lbl.items()
                    if k not in ["text", "start", "end", "label_type"]
                }
            )
            for lbl in labels
        ]
        
        split_str = item.get("split", "")
        split = None
        if split_str == "train":
            split = SplitType.TRAIN
        elif split_str == "test":
            split = SplitType.TEST
        
        return LabeledSample(
            id=sample_id,
            question=question,
            answer=item.get("response", ""),
            gold_answer="",
            context=context,
            task_type=task_type,
            split=split,
            label=1 if labels else 0,
            has_hallucination=bool(labels),
            hallucination_spans=hallucination_spans,
            source_model=item.get("model"),
            metadata={
                "source_id": source_id,
                "temperature": item.get("temperature"),
                "quality": item.get("quality"),
                "prompt": source_info.get("prompt", ""),
            }
        )
    
    def _extract_question_context(
        self,
        source_info: Dict[str, Any],
        task_type: TaskType
    ) -> tuple[str, Optional[str]]:
        """Extract question and context based on task type."""
        source_data = source_info.get("source_info", {})
        prompt = source_info.get("prompt", "")
        
        if task_type == TaskType.QA:
            if isinstance(source_data, dict):
                question = source_data.get("question", "")
                passages = source_data.get("passages", "")
                if isinstance(passages, list):
                    passages = "\n\n".join(str(p) for p in passages)
                return question, str(passages) if passages else None
        
        elif task_type == TaskType.SUMMARY:
            context = str(source_data) if source_data else None
            return prompt, context
        
        elif task_type == TaskType.DATA2TXT:
            if isinstance(source_data, dict):
                context = json.dumps(source_data, ensure_ascii=False, indent=2)
            else:
                context = str(source_data) if source_data else None
            return prompt, context
        
        return prompt, None
    
    def _should_include(self, sample: LabeledSample, raw_item: Dict[str, Any]) -> bool:
        """Check if sample passes all filters."""
        if self.split_filter:
            if sample.split and sample.split.value != self.split_filter:
                return False
        
        if self.task_type_filter:
            task_val = sample.task_type.value
            if task_val not in self.task_type_filter and task_val.upper() not in self.task_type_filter:
                return False
        
        if self.model_filter:
            if sample.source_model and sample.source_model not in self.model_filter:
                return False
        
        quality = raw_item.get("quality", "good")
        if quality in self.exclude_quality:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive RAGTruth statistics."""
        stats = {
            "total": 0,
            "by_split": {},
            "by_task_type": {},
            "by_model": {},
            "by_label": {"0": 0, "1": 0},
            "hallucination_types": {},
            "avg_response_length": 0,
            "avg_spans_per_hallucinated": 0,
        }
        
        total_response_len = 0
        total_spans = 0
        hallucinated_count = 0
        
        for sample in self.parse():
            stats["total"] += 1
            total_response_len += len(sample.answer)
            
            if sample.split:
                key = sample.split.value
                stats["by_split"][key] = stats["by_split"].get(key, 0) + 1
            
            key = sample.task_type.value
            stats["by_task_type"][key] = stats["by_task_type"].get(key, 0) + 1
            
            if sample.source_model:
                stats["by_model"][sample.source_model] = stats["by_model"].get(sample.source_model, 0) + 1
            
            stats["by_label"][str(sample.label)] += 1
            
            if sample.has_hallucination:
                hallucinated_count += 1
                total_spans += len(sample.hallucination_spans)
                
                for span in sample.hallucination_spans:
                    key = span.label_type.value
                    stats["hallucination_types"][key] = stats["hallucination_types"].get(key, 0) + 1
        
        if stats["total"] > 0:
            stats["avg_response_length"] = total_response_len / stats["total"]
            stats["hallucination_rate"] = stats["by_label"]["1"] / stats["total"]
        
        if hallucinated_count > 0:
            stats["avg_spans_per_hallucinated"] = total_spans / hallucinated_count
        
        return stats
    
    def get_by_model(self, model_name: str) -> Iterator[LabeledSample]:
        """Get samples from a specific model."""
        for sample in self.parse():
            if sample.source_model == model_name:
                yield sample
    
    def get_hallucinated_only(self) -> Iterator[LabeledSample]:
        """Get only samples with hallucinations."""
        for sample in self.parse():
            if sample.has_hallucination:
                yield sample
    
    def get_clean_only(self) -> Iterator[LabeledSample]:
        """Get only samples without hallucinations."""
        for sample in self.parse():
            if not sample.has_hallucination:
                yield sample
