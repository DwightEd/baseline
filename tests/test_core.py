"""Unit tests for core module components."""
import pytest
import tempfile
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRegistry:
    """Tests for Registry pattern implementation."""
    
    def test_register_and_get(self):
        from src.core import Registry
        
        registry = Registry("test")
        
        @registry.register("component")
        class TestComponent:
            pass
        
        assert "component" in registry
        assert registry.get("component") == TestComponent
    
    def test_register_with_aliases(self):
        from src.core import Registry
        
        registry = Registry("test")
        
        @registry.register("main", aliases=["alias1", "alias2"])
        class Component:
            pass
        
        assert registry.get("main") == Component
        assert registry.get("alias1") == Component
        assert registry.get("alias2") == Component
    
    def test_duplicate_registration_raises_error(self):
        from src.core import Registry, RegistryError
        
        registry = Registry("test")
        
        @registry.register("duplicate")
        class First:
            pass
        
        with pytest.raises(RegistryError):
            @registry.register("duplicate")
            class Second:
                pass
    
    def test_create_instance(self):
        from src.core import Registry
        
        registry = Registry("test")
        
        @registry.register("creatable")
        class Creatable:
            def __init__(self, value: int):
                self.value = value
        
        instance = registry.create("creatable", value=42)
        assert instance.value == 42
    
    def test_get_or_create_caches_instance(self):
        from src.core import Registry
        
        registry = Registry("test")
        
        @registry.register("singleton")
        class Singleton:
            pass
        
        instance1 = registry.get_or_create("singleton")
        instance2 = registry.get_or_create("singleton")
        assert instance1 is instance2


class TestDataSample:
    """Tests for DataSample type."""
    
    def test_basic_creation(self):
        from src.core import DataSample
        
        sample = DataSample(
            id="test_1",
            question="What is AI?",
            answer="Artificial Intelligence",
            gold_answer="AI is artificial intelligence"
        )
        
        assert sample.id == "test_1"
        assert sample.question == "What is AI?"
        assert sample.answer == "Artificial Intelligence"
    
    def test_with_metadata(self):
        from src.core import DataSample
        
        sample = DataSample(
            id="test_2",
            question="Test",
            metadata={"source": "test", "score": 0.9}
        )
        
        assert sample.metadata["source"] == "test"
        assert sample.metadata["score"] == 0.9
    
    def test_ensure_string_validator(self):
        from src.core import DataSample
        
        sample = DataSample(
            id="test",
            question=None,
            answer=123,
        )
        
        assert sample.question == ""
        assert sample.answer == "123"
    
    def test_labeled_sample(self):
        from src.core import LabeledSample, HallucinationSpan, HallucinationType
        
        sample = LabeledSample(
            id="labeled_1",
            question="Test question",
            has_hallucination=True,
            hallucination_spans=[
                HallucinationSpan(
                    text="fake info",
                    start=0,
                    end=9,
                    label_type=HallucinationType.EVIDENT_BASELESS
                )
            ]
        )
        
        assert sample.has_hallucination
        assert len(sample.hallucination_spans) == 1
        assert sample.hallucination_spans[0].label_type == HallucinationType.EVIDENT_BASELESS


class TestConfigManager:
    """Tests for configuration management."""
    
    def test_load_yaml_config(self):
        from src.core import ConfigManager
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
default:
  project: test
  value: 42
  nested:
    key: value
""")
            f.flush()
            
            config = ConfigManager(Path(f.name))
            assert config.get("project") == "test"
            assert config.get("value") == 42
            assert config.get("nested.key") == "value"
    
    def test_get_with_default(self):
        from src.core import ConfigManager
        
        config = ConfigManager()
        assert config.get("nonexistent", "default") == "default"
        assert config.get("deep.nested.key", 123) == 123
    
    def test_set_value(self):
        from src.core import ConfigManager
        
        config = ConfigManager()
        config.set("new.nested.key", "value")
        assert config.get("new.nested.key") == "value"
    
    def test_environment_variable_expansion(self):
        from src.core import expand_env_vars
        import os
        
        os.environ["TEST_VAR"] = "test_value"
        
        result = expand_env_vars("${TEST_VAR}")
        assert result == "test_value"
        
        result = expand_env_vars("${NONEXISTENT:default}")
        assert result == "default"
        
        del os.environ["TEST_VAR"]
    
    def test_deep_merge(self):
        from src.core import deep_merge
        
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}
        
        result = deep_merge(base, override)
        
        assert result["a"] == 1
        assert result["b"]["c"] == 10
        assert result["b"]["d"] == 3
        assert result["e"] == 5


class TestFeatureCache:
    """Tests for feature caching."""
    
    def test_cache_set_and_get(self):
        from src.core import FeatureCache
        import torch
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir), enabled=True)
            
            data = {"tensor": torch.tensor([1, 2, 3]), "value": 42}
            
            cache.set("test_key", data)
            assert cache.exists("test_key")
            
            retrieved = cache.get("test_key")
            assert retrieved is not None
            assert retrieved["value"] == 42
            assert torch.equal(retrieved["tensor"], data["tensor"])
    
    def test_cache_disabled(self):
        from src.core import FeatureCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir), enabled=False)
            
            cache.set("key", {"data": 1})
            assert cache.get("key") is None
            assert not cache.exists("key")
    
    def test_cache_stats(self):
        from src.core import FeatureCache
        import torch
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir), enabled=True)
            
            cache.set("exists", torch.tensor([1]))
            
            cache.get("exists")
            cache.get("missing")
            
            stats = cache.stats
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 0.5
    
    def test_cache_clear(self):
        from src.core import FeatureCache
        import torch
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir), enabled=True)
            
            cache.set("key1", torch.tensor([1]))
            cache.set("key2", torch.tensor([2]))
            
            count = cache.clear()
            assert count == 2
            assert not cache.exists("key1")
            assert not cache.exists("key2")


class TestComputeHash:
    """Tests for hash computation."""
    
    def test_deterministic_hash(self):
        from src.core import compute_hash
        
        h1 = compute_hash("a", "b", key=1)
        h2 = compute_hash("a", "b", key=1)
        assert h1 == h2
    
    def test_different_inputs_different_hash(self):
        from src.core import compute_hash
        
        h1 = compute_hash("a")
        h2 = compute_hash("b")
        assert h1 != h2
    
    def test_complex_inputs(self):
        from src.core import compute_hash
        
        h = compute_hash(
            {"nested": {"key": "value"}},
            [1, 2, 3],
            Path("/tmp/test")
        )
        
        assert len(h) == 16
        assert isinstance(h, str)


class TestTypeValidation:
    """Tests for Pydantic type validation."""
    
    def test_feature_extraction_config_defaults(self):
        from src.core import FeatureExtractionConfig
        
        config = FeatureExtractionConfig()
        
        assert config.batch_size == 1
        assert config.attention.enabled == True
        assert config.hidden_states.enabled == True
        assert config.token_probs.enabled == True
    
    def test_model_config_validation(self):
        from src.core import ModelConfig
        
        config = ModelConfig(name="test-model")
        assert config.dtype == "float16"
        assert config.device_map == "auto"
    
    def test_model_config_quantization_conflict(self):
        from src.core import ModelConfig
        
        with pytest.raises(ValueError):
            ModelConfig(name="test", load_in_8bit=True, load_in_4bit=True)
    
    def test_attention_config(self):
        from src.core import AttentionConfig
        
        config = AttentionConfig(
            layers="all",
            n_eigenvalues=20,
            normalization="symmetric"
        )
        
        assert config.n_eigenvalues == 20
        assert config.normalization == "symmetric"
    
    def test_evaluation_config(self):
        from src.core import EvaluationConfig
        
        config = EvaluationConfig(
            threshold=0.6,
            bootstrap_samples=500
        )
        
        assert config.threshold == 0.6
        assert config.bootstrap_samples == 500


class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_base_exception(self):
        from src.core import HallucinationDetectionError
        
        error = HallucinationDetectionError("Test error", details={"key": "value"})
        
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert "Test error" in str(error)
    
    def test_specific_exceptions(self):
        from src.core import (
            ConfigurationError,
            DatasetNotFoundError,
            ModelLoadError,
            MethodNotFittedError,
        )
        
        assert issubclass(ConfigurationError, Exception)
        assert issubclass(DatasetNotFoundError, Exception)
        assert issubclass(ModelLoadError, Exception)
        assert issubclass(MethodNotFittedError, Exception)


class TestLogging:
    """Tests for logging utilities."""
    
    def test_setup_logging(self):
        from src.core import setup_logging, get_logger
        import logging
        
        setup_logging("DEBUG")
        logger = get_logger("test")
        
        assert logger.level == logging.NOTSET
    
    def test_progress_logger(self):
        from src.core import ProgressLogger, get_logger
        
        logger = get_logger("test")
        progress = ProgressLogger(logger, total=100, prefix="Test")
        
        for _ in range(100):
            progress.update()
        
        assert progress.current == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
