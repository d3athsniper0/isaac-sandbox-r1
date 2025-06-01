# utils/async_utils.py
import asyncio
import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Coroutine

logger = logging.getLogger(__name__)

T = TypeVar('T')

async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """Run a coroutine with a timeout, returning default if it times out"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout} seconds")
        return default
    except Exception as e:
        logger.error(f"Error in with_timeout: {e}")
        return default

def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator to retry a coroutine with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                        raise
            raise last_exception  # This should never happen, but just in case
        return wrapper
    return decorator

class AsyncCache:
    """Simple in-memory async cache with TTL"""
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl  # seconds
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and is not expired"""
        if key in self.cache:
            entry = self.cache[key]
            # Check if the entry has expired
            if entry["expires"] > time.time():
                return entry["value"]
            # Remove expired entry
            del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with TTL"""
        self.cache[key] = {
            "value": value,
            "expires": time.time() + (ttl or self.ttl)
        }
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache"""
        if key in self.cache:
            del self.cache[key]