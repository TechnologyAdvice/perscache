"""perscache - compatibility utilities"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import asyncio as aio
import collections
import datetime as dt
import os
import re
import inspect
import hashlib
from itertools import repeat
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType

# Third-Party Imports
import cloudpickle
import pandas as pd
from beartype.typing import (
    Optional,
    Self,
    Type,
    TypeVar,
    Callable,
    Awaitable,
    Any,
)
from boltons.ioutils import SpooledBytesIO

# Imports From Package Sub-Modules
from ._logger import logger
from .serializers import (
    CloudPickleSerializer,
    CSVSerializer,
    JSONSerializer,
    ParquetSerializer,
    PickleSerializer,
    YAMLSerializer,
)

__all__ = (
    # Utility Functions
    "hash_all",
    "is_async",
    # Type Definitions
    "PathLike",
    "CachedValue",
    "CachedCallable",
    "CachedFunction",
    "CachedAsyncCallable",
    # Enhanced / 'Shim' Types
    "AsyncCacheLock",
    "SpooledTempFile",
    # Data Type Samples
    "EXCLUSIONS",
    "DATAFRAMES",
    "DATA_TYPES",
    "NON_DATAFRAMES",
)

# <editor-fold desc="# Constants ...">

md5_pattern: re.Pattern = re.compile(
    r"\W(?P<hash>[0-9a-f]{32})\W",
    flags=re.I,
)

# </editor-fold desc="# Constants ...">

# <editor-fold desc="# Data Type Samples ...">

DATA_TYPES = {
    # NON-DATAFRAMES
    "num": 123,
    "str": "abc",
    "bool": True,
    "set": {1, 2, 3},
    "list": [1, 2, 3],
    "tuple": (1, 2, 3),
    "datetime": dt.datetime.now(),
    "dict": {"a": 1, "b": 2, "c": 3},
    "frozenset": frozenset((1, 2, 3)),
    "datetime_with_timezone": dt.datetime.now(dt.timezone.utc),
    "object": collections.namedtuple(
        "NamedTuple",
        ("a", "b", "c"),
        module="perscache.data_types",
    )(1, 2, 3),
    # DATAFRAMES
    "dataframe_no_dates": pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": ("A", "B", "C"),
        }
    ),
    "dataframe_with_dates": pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": ("A", "B", "C"),
            "c": tuple(repeat(dt.datetime.now(), 3)),
        }
    ),
}
NON_DATAFRAMES = {
    "str",
    "num",
    "set",
    "bool",
    "list",
    "dict",
    "tuple",
    "object",
    "datetime",
    "frozenset",
    "datetime_with_timezone",
}
DATAFRAMES = {
    "dataframe_no_dates",
    "dataframe_with_dates",
}

#######################################################
#               Serializer Exclusions                 #
#######################################################
#                                                     #
# These serializers will not work with the mentioned  #
# data types -- they will either raise an exception   #
# or the saved and loaded data will not be identical. #
#                                                     #
#######################################################

EXCLUSIONS = {
    # Binary (Non-Human Friendly)
    CloudPickleSerializer: {},
    PickleSerializer: {"object"},
    ParquetSerializer: NON_DATAFRAMES,
    # Human Readable
    JSONSerializer: {
        "set",
        "tuple",
        "object",
        "datetime",
        "datetime_with_timezone",
        *DATAFRAMES,
    },
    YAMLSerializer: {
        "tuple",
        "object",
        *DATAFRAMES,
    },
    CSVSerializer: {
        "dataframe_with_dates",
        *NON_DATAFRAMES,
    },
}

# </editor-fold desc="# Data Type Samples ...">

# <editor-fold desc="# Type Definitions ...">

Unset = TypeVar("Unset", bound=Type[ValueError])
PathLike = TypeVar("Pathlike", str, Path, os.PathLike)

CachedValue = TypeVar("CachedValue")
CachedCallable = Callable[..., Optional[CachedValue]]
CachedAsyncCallable = Callable[..., Awaitable[Optional[CachedValue]]]
CachedFunction = TypeVar("CachedFunction", CachedCallable, CachedAsyncCallable)

# </editor-fold desc="# Type Definitions ...">

# <editor-fold desc="# Utility Functions ...">


def hash_all(*data: Any) -> str:
    """Pickles and hashes all the data passed to it as args."""
    result = hashlib.md5()  # nosec B303

    for datum in data:
        result.update(cloudpickle.dumps(datum))

    return result.hexdigest()


def is_async(fn: CachedFunction) -> bool:
    """Checks if the function is async."""
    return inspect.iscoroutinefunction(fn) and not inspect.isgeneratorfunction(fn)


def key_hash(key: str) -> str:
    """Extract just the md5 hash from the provided string."""
    match = md5_pattern.search(key)

    if not match:
        return key

    return match.groupdict().get("hash") or key


# </editor-fold desc="# Utility Functions ...">

# <editor-fold desc="# Enhanced / 'Shim' Types ...">


class SpooledTempFile(SpooledBytesIO):
    """A spooled file-like-object w/ timestamped write operations.

    More info:
      https://boltons.readthedocs.io/en/latest/ioutils.html#spooled-temporary-files
    """

    __atime: Optional[dt.datetime]
    __mtime: Optional[dt.datetime]
    _timestamp: Callable[[], dt.datetime]

    def __init__(
        self,
        max_size: Optional[int] = None,
        directory: Optional[PathLike] = None,
    ) -> None:
        max_size = max_size or 5_000_000

        super().__init__(max_size, directory)

        stamp = self._timestamp()
        self.__atime, self.__mtime = stamp, stamp

    _timestamp = staticmethod(
        lambda: dt.datetime.utcnow().replace(
            tzinfo=dt.timezone.utc,
        )
    )

    @property
    def mtime(self) -> dt.datetime:
        """Timestamp of the spool's most recent content modification."""
        return self.__mtime

    @property
    def atime(self) -> dt.datetime:
        """Timestamp of the spool was most recently read."""
        return self.__atime

    def size(self) -> int:
        """The "file size" of the spool."""
        return self.len

    def read(self, count: int = -1) -> bytes:
        """Read the specified number of bytes from the spool."""
        self.__atime = self._timestamp()
        return super().read(count)

    def write(self, data: bytes) -> None:
        """Write the specified data into the spool."""
        super().write(data)
        self.__mtime = self._timestamp()

    def delete(self) -> None:
        """Emulate deletion as though the spool was a file."""
        self.truncate(0)
        self.seek(0)

    def readline(self, length: Optional[int] = None) -> bytes:
        self.__atime = self._timestamp()
        return super().readline(length)

    def readlines(self, size_hint: int = 0) -> bytes:
        self.__atime = self._timestamp()
        return super().readlines(size_hint)


class AsyncCacheLock(aio.Event, AbstractAsyncContextManager):
    """A context-managed asyncio `Event` that functions as a simple lock for a
    specific function/cache-key pair."""

    __slots__ = ("_value",)

    _value: bool | Type[Unset]
    _fn_name: str
    _cache_key: str

    def __init__(self, fn_name: str, cache_key: str) -> None:
        super().__init__()

        self._value, self._fn_name, self._cache_key = (
            Unset,
            fn_name,
            key_hash(cache_key),
        )

    def is_set(self) -> bool:
        """Determine if the `Event`'s internal flag is currently set."""
        return self._value is True

    def is_unset(self) -> bool:
        """Determine if the `Event`'s internal flag is currently (explicitly)
        *not* set."""
        return self._value in (False, None, Unset)

    async def __aenter__(self) -> Self:
        if self._value is Unset:
            self.set()

        if not self.is_set():
            logger.debug(
                f"Waiting for outstanding '{self._fn_name}' "  # no-reformat
                f"cache lock for key: {self._cache_key}"  # no-reformat
            )

            await self.wait()

            logger.debug(
                f"Outstanding '{self._fn_name}' cache "  # no-reformat
                f"lock released for key: {self._cache_key}"  # no-reformat
            )

        self.clear()

        logger.debug(f"Cache lock claimed for '{self._fn_name}' for key: {self._cache_key}")

        return self

    async def __aexit__(
        self,
        exc_type: Type[Exception] | None,
        error: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        self.set()
        logger.debug(f"Cache lock released for '{self._fn_name}' for key: {self._cache_key}")


# </editor-fold desc="# Compatibility / 'Shim' Types ...">
