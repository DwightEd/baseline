"""Additional dataset parsers for common benchmarks."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List
import json
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import DATASET_REGISTRY, DataSample, TaskType, SplitType, get_logger
from .base import BaseDatasetParser

logger = get_logger(__name__)


@DATASET_REGISTRY.register("gsm8k", aliases=["math", "gsm"])
class GSM8KParser(BaseDatasetParser):
    """Parser for GSM8K math reasoning dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "gsm8k"
    
    def __init__(
        self,
        path: Path,
        split: str = "test",
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.split = split
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                answer_text = item.get("answer", "")
                
                match = re.search(r'####\s*(.+)$', answer_text)
                gold_answer = match.group(1).strip() if match else answer_text
                
                yield DataSample(
                    id=str(idx),
                    question=item.get("question", ""),
                    answer="",
                    gold_answer=gold_answer,
                    context=None,
                    task_type=TaskType.MATH,
                    split=SplitType.TEST if self.split == "test" else SplitType.TRAIN,
                    metadata={
                        "full_answer": answer_text,
                        "raw": item
                    }
                )


@DATASET_REGISTRY.register("triviaqa")
class TriviaQAParser(BaseDatasetParser):
    """Parser for TriviaQA dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "triviaqa"
    
    def __init__(self, path: Path, **kwargs: Any):
        super().__init__(path, **kwargs)
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("data", data) if isinstance(data, dict) else data
        
        for idx, item in enumerate(items):
            answer_info = item.get("answer", {})
            if isinstance(answer_info, dict):
                gold_answer = answer_info.get("value", answer_info.get("normalized_value", ""))
                aliases = answer_info.get("aliases", [])
            else:
                gold_answer = str(answer_info)
                aliases = []
            
            yield DataSample(
                id=item.get("question_id", str(idx)),
                question=item.get("question", ""),
                answer="",
                gold_answer=gold_answer,
                context=None,
                task_type=TaskType.QA,
                metadata={
                    "aliases": aliases,
                    "raw": item
                }
            )


@DATASET_REGISTRY.register("truthfulqa")
class TruthfulQAParser(BaseDatasetParser):
    """Parser for TruthfulQA dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "truthfulqa"
    
    def __init__(self, path: Path, **kwargs: Any):
        super().__init__(path, **kwargs)
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data if isinstance(data, list) else data.get("data", [])
        
        for idx, item in enumerate(items):
            correct_answers = item.get("correct_answers", [])
            gold_answer = correct_answers[0] if correct_answers else item.get("best_answer", "")
            
            yield DataSample(
                id=str(idx),
                question=item.get("question", ""),
                answer="",
                gold_answer=gold_answer,
                context=None,
                task_type=TaskType.QA,
                metadata={
                    "category": item.get("category"),
                    "source": item.get("source"),
                    "correct_answers": correct_answers,
                    "incorrect_answers": item.get("incorrect_answers", []),
                    "raw": item
                }
            )


@DATASET_REGISTRY.register("halueval", aliases=["halu_eval"])
class HaluEvalParser(BaseDatasetParser):
    """Parser for HaluEval hallucination evaluation dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "halueval"
    
    def __init__(self, path: Path, **kwargs: Any):
        super().__init__(path, **kwargs)
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                
                hallucination_val = item.get("hallucination", "")
                if isinstance(hallucination_val, str):
                    label = 1 if hallucination_val.lower() == "yes" else 0
                else:
                    label = int(hallucination_val) if hallucination_val else 0
                
                yield DataSample(
                    id=str(idx),
                    question=item.get("question", item.get("user_query", "")),
                    answer=item.get("hallucinated_answer", item.get("response", "")),
                    gold_answer=item.get("right_answer", item.get("ground_truth", "")),
                    context=item.get("knowledge", item.get("context")),
                    task_type=TaskType.QA,
                    label=label,
                    metadata={"raw": item}
                )


@DATASET_REGISTRY.register("coqa")
class CoQAParser(BaseDatasetParser):
    """Parser for CoQA conversational QA dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "coqa"
    
    def __init__(
        self,
        path: Path,
        flatten: bool = True,
        **kwargs: Any
    ):
        """
        Args:
            path: Path to CoQA JSON file
            flatten: If True, yield one sample per QA turn. If False, yield one per story.
        """
        super().__init__(path, **kwargs)
        self.flatten = flatten
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("data", data)
        
        for story_item in items:
            story = story_item.get("story", "")
            story_id = story_item.get("id", "")
            questions = story_item.get("questions", [])
            answers = story_item.get("answers", [])
            
            if self.flatten:
                for q, a in zip(questions, answers):
                    yield DataSample(
                        id=f"{story_id}_{q.get('turn_id', '')}",
                        question=q.get("input_text", ""),
                        answer="",
                        gold_answer=a.get("input_text", ""),
                        context=story,
                        task_type=TaskType.DIALOGUE,
                        metadata={
                            "story_id": story_id,
                            "turn_id": q.get("turn_id"),
                            "span_start": a.get("span_start"),
                            "span_end": a.get("span_end"),
                        }
                    )
            else:
                all_qa = [
                    (q.get("input_text", ""), a.get("input_text", ""))
                    for q, a in zip(questions, answers)
                ]
                yield DataSample(
                    id=story_id,
                    question=json.dumps(all_qa, ensure_ascii=False),
                    answer="",
                    gold_answer="",
                    context=story,
                    task_type=TaskType.DIALOGUE,
                    metadata={"turns": len(questions)}
                )


@DATASET_REGISTRY.register("nqopen", aliases=["natural_questions", "nq"])
class NQOpenParser(BaseDatasetParser):
    """Parser for Natural Questions Open dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "nqopen"
    
    def __init__(self, path: Path, **kwargs: Any):
        super().__init__(path, **kwargs)
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                answers = item.get("answer", [])
                gold_answer = answers[0] if answers else ""
                
                yield DataSample(
                    id=str(idx),
                    question=item.get("question", ""),
                    answer="",
                    gold_answer=gold_answer,
                    context=None,
                    task_type=TaskType.QA,
                    metadata={
                        "all_answers": answers,
                        "raw": item
                    }
                )


@DATASET_REGISTRY.register("squadv2", aliases=["squad", "squad2"])
class SQuADv2Parser(BaseDatasetParser):
    """Parser for SQuAD v2.0 dataset."""
    
    @classmethod
    def component_name(cls) -> str:
        return "squadv2"
    
    def __init__(self, path: Path, **kwargs: Any):
        super().__init__(path, **kwargs)
    
    def parse(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get("data", [])
        
        for article in articles:
            title = article.get("title", "")
            
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                
                for qa in para.get("qas", []):
                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", False)
                    
                    if answers:
                        gold_answer = answers[0].get("text", "")
                        all_answers = [a.get("text", "") for a in answers]
                    else:
                        gold_answer = ""
                        all_answers = []
                    
                    yield DataSample(
                        id=qa.get("id", ""),
                        question=qa.get("question", ""),
                        answer="",
                        gold_answer=gold_answer,
                        context=context,
                        task_type=TaskType.QA,
                        label=1 if is_impossible else 0,
                        metadata={
                            "title": title,
                            "is_impossible": is_impossible,
                            "all_answers": all_answers,
                        }
                    )


@DATASET_REGISTRY.register("custom")
class CustomParser(BaseDatasetParser):
    """Flexible parser for custom datasets with configurable field mapping."""
    
    @classmethod
    def component_name(cls) -> str:
        return "custom"
    
    def __init__(
        self,
        path: Path,
        format: str = "jsonl",
        field_mapping: Optional[Dict[str, str]] = None,
        task_type: TaskType = TaskType.QA,
        **kwargs: Any
    ):
        super().__init__(path, **kwargs)
        self.format = format
        self.field_mapping = field_mapping or {}
        self.task_type = task_type
    
    def parse(self) -> Iterator[DataSample]:
        if self.format == "json":
            yield from self._parse_json()
        elif self.format == "jsonl":
            yield from self._parse_jsonl()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _parse_json(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = data.get("data", [data])
        
        for idx, item in enumerate(data):
            yield self._parse_item(item, idx)
    
    def _parse_jsonl(self) -> Iterator[DataSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    yield self._parse_item(item, idx)
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> DataSample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped = self.field_mapping.get(key, key)
            return item.get(mapped, default)
        
        return DataSample(
            id=str(get_field("id", idx)),
            question=str(get_field("question", "")),
            answer=str(get_field("answer", "")),
            gold_answer=str(get_field("gold_answer", "")),
            context=get_field("context") or None,
            task_type=self.task_type,
            label=get_field("label"),
            metadata={"raw": item}
        )
