"""Integration tests for end-to-end workflows."""
import pytest
import tempfile
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatasetParsers:
    """Integration tests for dataset parsers."""
    
    def test_jsonl_parser(self):
        from src.datasets import JsonlDatasetParser
        from src.core import TaskType
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "question": "Q1", "answer": "A1", "gold_answer": "G1"}\n')
            f.write('{"id": "2", "question": "Q2", "answer": "A2", "gold_answer": "G2"}\n')
            f.flush()
            
            parser = JsonlDatasetParser(Path(f.name))
            samples = parser.load_all()
            
            assert len(samples) == 2
            assert samples[0].id == "1"
            assert samples[0].question == "Q1"
    
    def test_json_parser(self):
        from src.datasets import JsonDatasetParser
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {"id": "1", "question": "Q1", "answer": "A1"},
                {"id": "2", "question": "Q2", "answer": "A2"},
            ]
            json.dump(data, f)
            f.flush()
            
            parser = JsonDatasetParser(Path(f.name))
            samples = parser.load_all()
            
            assert len(samples) == 2
    
    def test_csv_parser(self):
        from src.datasets import CsvDatasetParser
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,question,answer,gold_answer\n")
            f.write("1,Q1,A1,G1\n")
            f.write("2,Q2,A2,G2\n")
            f.flush()
            
            parser = CsvDatasetParser(Path(f.name))
            samples = parser.load_all()
            
            assert len(samples) == 2
            assert samples[0].id == "1"
    
    def test_create_parser_auto_detect(self):
        from src.datasets import create_parser
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "question": "Q", "answer": "A"}\n')
            f.flush()
            
            parser = create_parser(f.name)
            samples = parser.load_all()
            
            assert len(samples) == 1
    
    def test_parser_with_field_mapping(self):
        from src.datasets import JsonlDatasetParser
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"sample_id": "1", "query": "Q1", "response": "A1"}\n')
            f.flush()
            
            parser = JsonlDatasetParser(
                Path(f.name),
                field_mapping={
                    "id": "sample_id",
                    "question": "query",
                    "answer": "response"
                }
            )
            samples = parser.load_all()
            
            assert samples[0].id == "1"
            assert samples[0].question == "Q1"
            assert samples[0].answer == "A1"
    
    def test_parser_statistics(self):
        from src.datasets import JsonlDatasetParser
        from src.core import TaskType
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(10):
                f.write(f'{{"id": "{i}", "question": "Q{i}", "answer": "A{i}"}}\n')
            f.flush()
            
            parser = JsonlDatasetParser(Path(f.name))
            stats = parser.get_statistics()
            
            assert stats["total"] == 10
            assert "by_task_type" in stats


class TestMethodsIntegration:
    """Integration tests for detection methods."""
    
    def test_method_registry(self):
        from src.methods import list_methods, create_method
        
        available = list_methods()
        assert "lapeigvals" in available
        assert "entropy" in available
        assert "perplexity" in available
    
    def test_prediction_dataclass(self):
        from src.methods import Prediction
        
        pred = Prediction(
            score=0.8,
            label=1,
            confidence=0.9,
            metadata={"method": "test"}
        )
        
        result = pred.to_dict()
        assert result["score"] == 0.8
        assert result["label"] == 1
    
    def test_method_metrics(self):
        from src.methods import MethodMetrics
        
        metrics = MethodMetrics(
            auroc=0.85,
            auprc=0.80,
            f1=0.75,
            accuracy=0.80
        )
        
        result = metrics.to_dict()
        assert result["auroc"] == 0.85
        assert result["f1"] == 0.75


class TestEvaluationIntegration:
    """Integration tests for evaluation module."""
    
    def test_evaluator_basic(self):
        from src.evaluation import Evaluator, EvaluationResult
        from src.methods import Prediction
        
        predictions = [
            Prediction(score=0.9, label=1, confidence=0.9),
            Prediction(score=0.8, label=1, confidence=0.8),
            Prediction(score=0.3, label=0, confidence=0.7),
            Prediction(score=0.2, label=0, confidence=0.8),
        ]
        labels = [1, 1, 0, 0]
        
        evaluator = Evaluator()
        result = evaluator.evaluate(predictions, labels)
        
        assert result.auroc > 0.5
        assert result.accuracy == 1.0
    
    def test_evaluation_result_save_load(self):
        from src.evaluation import EvaluationResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = EvaluationResult(
                auroc=0.85,
                auprc=0.80,
                f1=0.75
            )
            
            path = Path(tmpdir) / "result.json"
            result.save(path)
            
            loaded = EvaluationResult.load(path)
            assert loaded.auroc == 0.85
    
    def test_find_optimal_threshold(self):
        from src.evaluation import Evaluator
        from src.methods import Prediction
        
        predictions = [
            Prediction(score=0.9, label=1, confidence=0.9),
            Prediction(score=0.7, label=1, confidence=0.7),
            Prediction(score=0.4, label=0, confidence=0.6),
            Prediction(score=0.2, label=0, confidence=0.8),
        ]
        labels = [1, 1, 0, 0]
        
        evaluator = Evaluator()
        threshold, f1 = evaluator.find_optimal_threshold(predictions, labels)
        
        assert 0 < threshold < 1
        assert f1 > 0
    
    def test_evaluation_result_summary(self):
        from src.evaluation import EvaluationResult
        
        result = EvaluationResult(
            auroc=0.85,
            auprc=0.80,
            f1=0.75,
            accuracy=0.80,
            confusion_matrix=[[10, 2], [3, 15]]
        )
        
        summary = result.summary()
        assert "AUROC" in summary
        assert "0.85" in summary


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_load(self):
        from src.core import ConfigManager
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
default:
  dataset:
    name: test
    path: ./data
  model:
    name: test-model
    dtype: float16
  feature_extraction:
    mode: teacher_forcing
    batch_size: 1
""")
            f.flush()
            
            config = ConfigManager(Path(f.name))
            
            assert config.get("dataset.name") == "test"
            assert config.get("model.dtype") == "float16"
            assert config.get("feature_extraction.mode") == "teacher_forcing"
    
    def test_config_environment_override(self):
        from src.core import ConfigManager
        import os
        
        os.environ["TEST_DATA_PATH"] = "/custom/path"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
default:
  data_path: "${TEST_DATA_PATH:/default/path}"
""")
            f.flush()
            
            config = ConfigManager(Path(f.name))
            assert config.get("data_path") == "/custom/path"
        
        del os.environ["TEST_DATA_PATH"]
    
    def test_config_merge_environments(self):
        from src.core import ConfigManager
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
default:
  batch_size: 32
  learning_rate: 0.001

development:
  batch_size: 8
""")
            f.flush()
            
            config = ConfigManager(Path(f.name), env="development")
            
            assert config.get("batch_size") == 8
            assert config.get("learning_rate") == 0.001


class TestCacheIntegration:
    """Integration tests for caching system."""
    
    def test_cache_manager(self):
        from src.core import CacheManager
        import torch
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))
            
            feature_cache = manager.get_feature_cache("features")
            metadata_cache = manager.get_metadata_cache("metadata")
            
            feature_cache.set("key1", torch.tensor([1, 2, 3]))
            metadata_cache.set("key1", {"info": "test"})
            
            assert feature_cache.exists("key1")
            assert metadata_cache.exists("key1")
            
            stats = manager.get_stats()
            assert "features" in stats
    
    def test_cached_decorator(self):
        from src.core import cached
        import torch
        
        call_count = 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            @cached(Path(tmpdir))
            def expensive_function(x):
                nonlocal call_count
                call_count += 1
                return torch.tensor([x * 2])
            
            result1 = expensive_function(5)
            result2 = expensive_function(5)
            
            assert call_count == 1
            assert torch.equal(result1, result2)


class TestEndToEnd:
    """End-to-end workflow tests."""
    
    def test_data_to_prediction_pipeline(self):
        from src.core import DataSample, ExtractedFeatures
        from src.methods import create_method, Prediction
        import numpy as np
        
        samples = [
            DataSample(id=str(i), question=f"Q{i}", answer=f"A{i}")
            for i in range(20)
        ]
        
        features = [
            ExtractedFeatures(
                sample_id=s.id,
                prompt_length=10,
                response_length=5,
                total_length=15,
                token_prob_features={
                    "mean_entropy": np.random.random(),
                    "max_entropy": np.random.random(),
                    "perplexity": np.random.random() * 10,
                }
            )
            for s in samples
        ]
        
        labels = [1 if i % 2 == 0 else 0 for i in range(20)]
        
        method = create_method("entropy")
        method.fit(features[:15], labels[:15], features[15:], labels[15:])
        
        predictions = method.predict_batch(features[15:])
        
        assert len(predictions) == 5
        assert all(isinstance(p, Prediction) for p in predictions)
        assert all(0 <= p.score <= 1 for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
