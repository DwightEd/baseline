"""Registry pattern for dynamic component registration."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Type, Callable, Optional, Any, List
import logging

logger = logging.getLogger(__name__)
T = TypeVar('T')


class RegistryError(Exception):
    """Registry operation error."""
    pass


class Registry(Generic[T]):
    """Generic registry for component registration."""
    
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[T]] = {}
        self._instances: Dict[str, T] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **metadata: Any
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator for registering a class."""
        def decorator(cls: Type[T]) -> Type[T]:
            key = name or cls.__name__.lower()
            if key in self._registry:
                raise RegistryError(f"'{key}' already registered in {self._name}")
            self._registry[key] = cls
            self._metadata[key] = metadata
            for alias in (aliases or []):
                if alias in self._registry:
                    raise RegistryError(f"Alias '{alias}' exists in {self._name}")
                self._registry[alias] = cls
            logger.debug(f"Registered {cls.__name__} as '{key}' in {self._name}")
            return cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """Get registered class."""
        if name not in self._registry:
            available = sorted(set(self._registry.keys()))
            raise RegistryError(f"'{name}' not in {self._name}. Available: {available}")
        return self._registry[name]
    
    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Create instance of registered class."""
        return self.get(name)(*args, **kwargs)
    
    def get_or_create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Get cached instance or create new."""
        if name not in self._instances:
            self._instances[name] = self.create(name, *args, **kwargs)
        return self._instances[name]
    
    def list_registered(self) -> List[str]:
        """List registered names (unique)."""
        return sorted(set(self._registry.keys()))
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get component metadata."""
        return self._metadata.get(name, {})
    
    def clear_cache(self) -> None:
        """Clear instance cache."""
        self._instances.clear()
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry
    
    def __len__(self) -> int:
        return len(set(self._registry.values()))


class BaseComponent(ABC):
    """Abstract base for registrable components."""
    
    @classmethod
    @abstractmethod
    def component_name(cls) -> str:
        """Component name for registration."""
        pass
    
    @classmethod
    def component_description(cls) -> str:
        """Component description."""
        return cls.__doc__ or ""


# Global registries
DATASET_REGISTRY: Registry = Registry("datasets")
MODEL_REGISTRY: Registry = Registry("models")
FEATURE_EXTRACTOR_REGISTRY: Registry = Registry("feature_extractors")
METHOD_REGISTRY: Registry = Registry("methods")
EVALUATOR_REGISTRY: Registry = Registry("evaluators")
PROMPT_REGISTRY: Registry = Registry("prompts")


def get_registry(name: str) -> Registry:
    """Get registry by name."""
    registries = {
        "datasets": DATASET_REGISTRY,
        "models": MODEL_REGISTRY,
        "feature_extractors": FEATURE_EXTRACTOR_REGISTRY,
        "methods": METHOD_REGISTRY,
        "evaluators": EVALUATOR_REGISTRY,
        "prompts": PROMPT_REGISTRY,
    }
    if name not in registries:
        raise RegistryError(f"Unknown registry: {name}")
    return registries[name]
