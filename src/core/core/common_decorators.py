"""Defines commonly used function decorators."""

import functools
import logging
import warnings


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)


def deprecated(message="This function will be removed in future versions."):
    """Decorator to mark functions as deprecated with a custom message."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {message}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def under_development(message="This function is under development and may not work as expected."):
    """Decorator to mark functions as under development with a custom message."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is under development: {message}",
                category=UserWarning,
                stacklevel=2
            )
            try:
                return func(*args, **kwargs)
            except Exception as err:
                LOG.warning(f'The WIP function {func.__name__} raised an error\n{err}')
        return wrapper
    return decorator
