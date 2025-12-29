"""Caching utilities for features and computations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Dict
from functools import wraps

from .logging import get_logger
from .exceptions import CacheError

logger = get_logger(__name__)
T = TypeVar('T')


def compute_hash(*args: Any, **kwargs: Any) -> str:
    """Compute stable hash for arguments."""
    def serialize(obj: Any) -> str:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return f"[{','.join(serialize(x) for x in obj)}]"
        elif isinstance(obj, dict):
            items = sorted((str(k), serialize(v)) for k, v in obj.items())
            return f"{{{','.join(f'{k}:{v}' for k, v in items)}}}"
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return serialize(obj.__dict__)
        return str(type(obj).__name__)
    
    content = serialize(args) + serialize(kwargs)
    return hashlib.md5(content.encode()).hexdigest()[:16]


class FeatureCache:
    """Cache for extracted features (PyTorch format)."""
    
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._stats = {"hits": 0, "misses": 0}
        
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str, suffix: str = ".pt") -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}{suffix}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if not self.enabled:
            return None
        
        path = self._get_path(key)
        if path.exists():
            try:
                import torch
                self._stats["hits"] += 1
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(f"Cache load failed '{key}': {e}")
        
        self._stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        if not self.enabled:
            return
        
        try:
            import torch
            torch.save(value, self._get_path(key))
        except Exception as e:
            logger.warning(f"Cache save failed '{key}': {e}")
    
    def exists(self, key: str) -> bool:
        return self.enabled and self._get_path(key).exists()
    
    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cached values."""
        count = 0
        if self.cache_dir.exists():
            for path in self.cache_dir.glob("*.pt"):
                path.unlink()
                count += 1
        self._stats = {"hits": 0, "misses": 0}
        return count
    
    def list_keys(self) -> list:
        if not self.cache_dir.exists():
            return []
        return [p.stem for p in self.cache_dir.glob("*.pt")]
    
    @property
    def stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "total": total,
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
            "size": len(self.list_keys())
        }


class MetadataCache:
    """Cache for JSON metadata."""
    
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        path = self._get_path(key)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Metadata load failed '{key}': {e}")
        return None
    
    def set(self, key: str, value: Dict) -> None:
        if not self.enabled:
            return
        try:
            with open(self._get_path(key), 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Metadata save failed '{key}': {e}")
    
    def exists(self, key: str) -> bool:
        return self.enabled and self._get_path(key).exists()
    
    def clear(self) -> int:
        count = 0
        if self.cache_dir.exists():
            for path in self.cache_dir.glob("*.json"):
                path.unlink()
                count += 1
        return count


class CacheManager:
    """Manages multiple cache instances."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._feature_caches: Dict[str, FeatureCache] = {}
        self._metadata_caches: Dict[str, MetadataCache] = {}
    
    def get_feature_cache(self, name: str, enabled: bool = True) -> FeatureCache:
        if name not in self._feature_caches:
            self._feature_caches[name] = FeatureCache(self.base_dir / name, enabled)
        return self._feature_caches[name]
    
    def get_metadata_cache(self, name: str, enabled: bool = True) -> MetadataCache:
        if name not in self._metadata_caches:
            self._metadata_caches[name] = MetadataCache(self.base_dir / f"{name}_meta", enabled)
        return self._metadata_caches[name]
    
    def clear_all(self) -> Dict[str, int]:
        results = {}
        for name, cache in self._feature_caches.items():
            results[name] = cache.clear()
        for name, cache in self._metadata_caches.items():
            results[f"{name}_meta"] = cache.clear()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {name: cache.stats for name, cache in self._feature_caches.items()}


def cached(cache_dir: Path, key_fn: Optional[Callable[..., str]] = None, enabled: bool = True):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = FeatureCache(cache_dir, enabled=enabled)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not enabled:
                return func(*args, **kwargs)
            
            key = key_fn(*args, **kwargs) if key_fn else f"{func.__name__}_{compute_hash(*args, **kwargs)}"
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator
