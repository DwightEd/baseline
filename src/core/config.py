"""Modular configuration management with inheritance support."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, List
import yaml
from pydantic import BaseModel, ValidationError

from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)
T = TypeVar('T', bound=BaseModel)

ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables."""
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    elif isinstance(value, str):
        def replace(match: re.Match) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")
        return ENV_PATTERN.sub(replace, value)
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    if not path.exists():
        raise ConfigurationError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class ConfigLoader:
    """Loads and manages modular configurations."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_with_inheritance(self, path: Path) -> Dict[str, Any]:
        """Load config with _inherits support."""
        config = load_yaml(path)
        
        if "_inherits" in config:
            parent_name = config.pop("_inherits")
            parent_path = path.parent / f"{parent_name}.yaml"
            parent_config = self.load_with_inheritance(parent_path)
            config = deep_merge(parent_config, config)
        
        return config
    
    def load_module(self, category: str, name: str) -> Dict[str, Any]:
        """Load config from category/name.yaml."""
        cache_key = f"{category}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self.config_dir / category / f"{name}.yaml"
        config = self.load_with_inheritance(path)
        config = expand_env_vars(config)
        
        self._cache[cache_key] = config
        return config
    
    def load_dataset(self, name: str) -> Dict[str, Any]:
        return self.load_module("datasets", name)
    
    def load_model(self, name: str) -> Dict[str, Any]:
        return self.load_module("models", name)
    
    def load_prompts(self, name: str) -> Dict[str, Any]:
        return self.load_module("prompts", name)
    
    def load_method(self, name: str) -> Dict[str, Any]:
        return self.load_module("methods", name)
    
    def load_features(self, name: str) -> Dict[str, Any]:
        return self.load_module("features", name)
    
    def load_environment(self, name: str) -> Dict[str, Any]:
        return self.load_module("environments", name)
    
    def clear_cache(self) -> None:
        self._cache.clear()


class ConfigManager:
    """Main configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None, config_dir: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._validated: Optional[BaseModel] = None
        self.config_path = config_path
        self.config_dir = config_dir or (config_path.parent if config_path else Path("config"))
        self.loader = ConfigLoader(self.config_dir)
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Path) -> 'ConfigManager':
        """Load main configuration."""
        self._config = load_yaml(config_path)
        self._config = expand_env_vars(self._config)
        self._validated = None
        logger.info(f"Loaded config from {config_path}")
        return self
    
    def load_active_configs(self) -> 'ConfigManager':
        """Load all configs specified in 'active' section."""
        active = self._config.get("active", {})
        
        if dataset := active.get("dataset"):
            self._config["_dataset"] = self.loader.load_dataset(dataset)
        if model := active.get("model"):
            self._config["_model"] = self.loader.load_model(model)
        if prompt := active.get("prompt"):
            self._config["_prompts"] = self.loader.load_prompts(prompt)
        if method := active.get("method"):
            self._config["_method"] = self.loader.load_method(method)
        if env := active.get("environment"):
            env_config = self.loader.load_environment(env)
            self._config = deep_merge(self._config, env_config)
        
        return self
    
    def load_all_features(self) -> Dict[str, Dict[str, Any]]:
        """Load all feature configs."""
        return {
            "attention": self.loader.load_features("attention"),
            "hidden_states": self.loader.load_features("hidden_states"),
            "token_probs": self.loader.load_features("token_probs"),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set config value."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self._validated = None
    
    def get_dataset_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get dataset configuration."""
        if name:
            return self.loader.load_dataset(name)
        return self._config.get("_dataset", {})
    
    def get_model_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration."""
        if name:
            return self.loader.load_model(name)
        return self._config.get("_model", {})
    
    def get_method_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get method configuration."""
        if name:
            return self.loader.load_method(name)
        return self._config.get("_method", {})
    
    def get_prompts(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get prompt configuration."""
        if name:
            return self.loader.load_prompts(name)
        return self._config.get("_prompts", {})
    
    def validate(self, schema: Type[T]) -> T:
        """Validate against schema."""
        try:
            self._validated = schema.model_validate(self._config)
            return self._validated
        except ValidationError as e:
            raise ConfigurationError("Validation failed", details=e.errors())
    
    def override_from_dict(self, overrides: Dict[str, Any]) -> 'ConfigManager':
        """Override from dictionary."""
        for key, value in overrides.items():
            if value is not None:
                self.set(key, value)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return self._config.copy()
    
    def save(self, path: Path) -> None:
        """Save configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


_global_config: Optional[ConfigManager] = None


def init_config(
    config_path: Path,
    config_dir: Optional[Path] = None,
    load_active: bool = True
) -> ConfigManager:
    """Initialize global configuration."""
    global _global_config
    _global_config = ConfigManager(config_path, config_dir)
    if load_active:
        _global_config.load_active_configs()
    return _global_config


def get_config() -> ConfigManager:
    """Get global configuration."""
    if _global_config is None:
        raise ConfigurationError("Config not initialized. Call init_config() first.")
    return _global_config


def get_value(key: str, default: Any = None) -> Any:
    """Get value from global config."""
    return get_config().get(key, default)
