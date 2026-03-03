"""
Cache layer for AutoOCR.

Provides caching functionality for OCR results and other expensive operations.
"""
import json
import hashlib
import time
from typing import Any, Optional, Dict, Callable
from functools import wraps
from threading import Lock

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheBackend:
    """Base cache backend."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        raise NotImplementedError
    
    def delete(self, key: str):
        """Delete value from cache."""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    def clear(self):
        """Clear all cache."""
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if expiry is None or time.time() < expiry:
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache."""
        with self._lock:
            expiry = None if ttl is None else time.time() + ttl
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None
    
    def clear(self):
        """Clear all memory cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'size': len(self._cache)
            }


class RedisCache(CacheBackend):
    """Redis cache implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not installed")
        
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = self._client.get(key)
            if value is not None:
                self._hits += 1
                # Use JSON instead of pickle for security
                return json.loads(value.decode('utf-8'))
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache."""
        try:
            serialized = json.dumps(value)
            if ttl:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
        except Exception as e:
            print(f"Redis cache set error: {e}")
    
    def delete(self, key: str):
        """Delete value from Redis cache."""
        try:
            self._client.delete(key)
        except Exception as e:
            print(f"Redis cache delete error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False
    
    def clear(self):
        """Clear all Redis cache."""
        try:
            self._client.flushdb()
            self._hits = 0
            self._misses = 0
        except Exception as e:
            print(f"Redis cache clear error: {e}")
    
    def get_stats(self) -> Dict:
        """Get Redis cache statistics."""
        try:
            info = self._client.info('stats')
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                'redis_keys': self._client.dbsize()
            }
        except Exception:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': 0,
                'redis_keys': 0
            }


class Cache:
    """Cache manager with fallback support."""
    
    def __init__(self, backend: Optional[CacheBackend] = None, use_memory_fallback: bool = True):
        self._backend = backend
        self._memory_fallback = MemoryCache() if use_memory_fallback else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try primary backend first
        if self._backend:
            value = self._backend.get(key)
            if value is not None:
                return value
        
        # Try memory fallback
        if self._memory_fallback:
            return self._memory_fallback.get(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if self._backend:
            self._backend.set(key, value, ttl)
        
        if self._memory_fallback:
            self._memory_fallback.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete value from cache."""
        if self._backend:
            self._backend.delete(key)
        
        if self._memory_fallback:
            self._memory_fallback.delete(key)
    
    def clear(self):
        """Clear all cache."""
        if self._backend:
            self._backend.clear()
        
        if self._memory_fallback:
            self._memory_fallback.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {}
        if self._backend:
            stats['backend'] = self._backend.get_stats()
        if self._memory_fallback:
            stats['memory'] = self._memory_fallback.get_stats()
        return stats


# Utility functions
def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: int = 3600, key_prefix: str = ''):
    """Decorator to cache function results."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or initialize cache
            cache = get_cache()
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache invalidation method
        def invalidate(*args, **kwargs):
            cache = get_cache()
            cache.delete(f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}")
        
        wrapper.invalidate = invalidate
        
        return wrapper
    return decorator


# Global cache instance - initialized with memory fallback by default
_global_cache: Optional[Cache] = Cache(backend=None, use_memory_fallback=True)

def get_cache(backend: Optional[CacheBackend] = None) -> Cache:
    """Get or create the global cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = Cache(backend=backend)
    return _global_cache


def init_redis_cache(host: str = 'localhost', port: int = 6379, 
                    db: int = 0, password: Optional[str] = None) -> Cache:
    """Initialize Redis cache."""
    if not REDIS_AVAILABLE:
        print("Warning: Redis not available, using memory cache")
        return get_cache()
    
    try:
        redis_backend = RedisCache(host, port, db, password)
        cache = Cache(backend=redis_backend)
        global _global_cache
        _global_cache = cache
        return cache
    except Exception as e:
        print(f"Warning: Failed to initialize Redis cache: {e}")
        return get_cache()
