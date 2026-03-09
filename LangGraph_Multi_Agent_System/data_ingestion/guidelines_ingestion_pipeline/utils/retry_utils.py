"""
Retry utilities for resilient pipeline operations.

Provides exponential backoff decorator with jitter for retrying transient
failures (network timeouts, API rate limits, temporary resource unavailability).
"""

import functools
import random
import time
from typing import Callable, Tuple, Type, TypeVar, cast

import structlog


logger = structlog.get_logger(__name__)

T = TypeVar("T")


def retry_with_exponential_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles on each retry)
        max_delay: Maximum delay cap in seconds
        retryable_exceptions: Tuple of exception types to retry on
    
    Returns:
        Decorated function that will retry on failure
    
    Raises:
        The original exception if all retry attempts are exhausted
    
    Example:
        @retry_with_exponential_backoff(
            max_attempts=3,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
        def fetch_data():
            return api_client.get("/data")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception = Exception("No attempts made")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                        )
                        raise
                    
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=round(total_delay, 2),
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                    )
                    
                    time.sleep(total_delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator
