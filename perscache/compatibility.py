"""perscache - compatibility utilities"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import collections
import datetime as dt
import hashlib
import inspect
import os
import re
from itertools import repeat
from pathlib import Path

# Third-Party Imports
import cloudpickle
import pandas as pd
from beartype.typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
)

# Imports From Package Sub-Modules
from .serializers import (
    CloudPickleSerializer,
    CSVSerializer,
    JSONSerializer,
    ParquetSerializer,
    PickleSerializer,
    YAMLSerializer,
)

try:
    import cloudpickle.compat as _

    PicklingError = cloudpickle.compat.pickle.PicklingError
except (ImportError, ModuleNotFoundError):
    try:
        import cloudpickle.cloudpickle as _

        PicklingError = cloudpickle.cloudpickle.pickle.PicklingError
    except (ImportError, ModuleNotFoundError):
        import pickle as _

        PicklingError = _.PicklingError
finally:
    del _


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
    )(
        1, 2, 3  # noqa
    ),
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
        try:
            result.update(cloudpickle.dumps(datum))
        except (TypeError, PicklingError):
            pass

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
