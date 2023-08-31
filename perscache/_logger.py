"""perscache - logging utilities."""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import logging
from functools import wraps
from typing import (
    Any,
    Callable,
)

__all__ = ("trace",)

MAX_LEN: int = 200
TRACE_LEVEL: int = 1
TRACE_NAME: str = "TRACE"

logging.addLevelName(TRACE_LEVEL, TRACE_NAME)

logger = logging.getLogger("perscache")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


def _trim(string: Any) -> str:
    """Trim the supplied string to be under `MAX_LEN` characters."""
    if MAX_LEN < len(string := str(string)):
        string = f"{string[:MAX_LEN]}..."

    return string


def trace(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the supplied function with entry and exit logs."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        """Log entry and exit from the decorated function."""
        func_name = _trim(func.__name__)

        logger.log(
            TRACE_LEVEL,
            f"Entering {func_name}, args={_trim(args)}, kwargs={_trim(kwargs)}",
        )

        result = func(*args, **kwargs)

        logger.log(
            TRACE_LEVEL,
            f"Exiting {func_name}, result={_trim(result)}",
        )

        return result

    return wrapper
