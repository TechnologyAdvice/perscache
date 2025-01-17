"""perscache - storage backends"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import datetime as dt
import pickle
import shelve
import sys
import tempfile
from abc import (
    ABC,
    abstractmethod,
)
from collections import namedtuple
from functools import (
    cache as memoize,
    reduce,
)
from itertools import chain
from pathlib import Path

# Third-Party Imports
from beartype import beartype
from beartype.typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Union,
    cast,
)

# Imports From Package Sub-Modules
from .compatibility import PathLike

CacheRecord = namedtuple(
    "CacheRecord",
    (
        "key",
        "data",
        "created_at",
        "updated_at",
    ),
    defaults=(None, None, None, None),
)
DefaultTempDir: Path = (Path(tempfile.gettempdir()) / "perscache").resolve()


class CacheError(LookupError):
    """Base exception type for cache errors."""

    ...


class CacheMiss(CacheError):
    """Raised when a cache file / entry does not exist."""

    ...


class CacheExpired(CacheError):
    """Raised when a cache file / entry exists but is expired."""

    ...


class Storage(ABC):
    """Cache storage."""

    _timestamp: Callable[[], dt.datetime]

    _timestamp = staticmethod(
        lambda: dt.datetime.utcnow().replace(
            tzinfo=dt.timezone.utc,
        )
    )

    @abstractmethod
    def read(
        self,
        path: PathLike,
        deadline: Optional[dt.datetime] = None,
    ) -> bytes:
        """Read the file at the given path and return its contents as bytes.

        If the file does not exist, raise CacheMiss. If the file is
        older than the given deadline, raise CacheExpired.
        """
        ...

    @abstractmethod
    def write(self, path: PathLike, data: bytes) -> None:
        """Write the file at the given path."""
        ...

    @staticmethod
    @memoize
    def _resolve_path(path: Optional[PathLike]) -> Path:
        """Ensure the specified path points to a valid directory."""
        path = Path(path or DefaultTempDir).resolve()

        target_path = path

        if target_path.is_file() or target_path.suffix:
            target_path = path.parent

        target_path.mkdir(
            mode=0o755,
            parents=True,
            exist_ok=True,
        )

        return path


class ShelvedStorage(Storage):
    """Cache storage backed by Python's stdlib `shelve` module."""

    __slots__ = (
        "_max_size",
        "_strategy",
        "_shelf_path",
        "__resolved_shelf",
    )

    _max_size: Optional[int]
    _strategy: str
    _shelf_path: Path
    __resolved_shelf: Optional[Path]

    @beartype
    def __init__(
        self,
        location: Optional[PathLike] = None,
        max_size: Optional[int] = None,
        always_fresh: bool = True,
        eviction_strategy: Optional[str] = None,
    ) -> None:
        self._shelf_path = self._resolve_path(location)
        self._strategy = eviction_strategy or "least_recently_updated"
        self._max_size = max_size if isinstance(max_size, int) else None
        self.__resolved_shelf = None

        if self._shelf_path.is_dir():
            self._shelf_path /= "perscache"

        if self._shelf_path.suffix:
            shelf_path = reduce(
                lambda value, suffix: value.replace(suffix, ""),
                self._shelf_path.suffixes,
                str(self._shelf_path),
            )
            self._shelf_path = Path(shelf_path)

        if always_fresh:
            try:
                self.shelf_path.unlink(missing_ok=True)
            except (AttributeError, FileNotFoundError):
                pass

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__

        try:
            size = round(self.size / 1024, 4)
        except FileNotFoundError:
            size = 0

        return f"<{cls_name}(size={size}mb, shelf={self.shelf_path})>"

    @property
    def _shelf(self) -> shelve.Shelf:
        """The instance's storage shelf."""
        return shelve.open(
            flag="c",
            writeback=False,
            protocol=pickle.HIGHEST_PROTOCOL,
            filename=cast(str, self._shelf_path),
        )

    @property
    def shelf_path(self) -> Path:
        """The on-disk path to the storage shelf."""
        if self.__resolved_shelf:
            return self.__resolved_shelf

        with self._shelf as shelf:
            shelf[f"shelf_{hex(id(self))[2:]}"] = str(self._shelf_path)

        shelf_name = self._shelf_path.stem
        globbed = chain(
            self._shelf_path.parent.rglob(f"**/{shelf_name}.*"),
            self._shelf_path.parent.rglob(f"**/{shelf_name}"),
        )
        shelf_path = self.__resolved_shelf = next(filter(Path.is_file, globbed), None)

        if not shelf_path:
            raise FileNotFoundError(f"No shelve file found for: {self._shelf_path}")

        return shelf_path

    @property
    def size(self) -> int:
        """The shelf's on-disk size."""
        return self.shelf_path.stat().st_size

    def read(
        self,
        path: PathLike,
        deadline: Optional[dt.datetime] = None,
    ) -> bytes:
        """Retrieve the shelved data for the given "path" key and return its
        contents as bytes.

        If the key does not exist, raise CacheMiss. If the shelved data
        is older than the given deadline, raise CacheExpired.
        """
        with self._shelf as shelf:
            cached: Optional[CacheRecord] = shelf.get(path)

        if not cached:
            raise CacheMiss

        if deadline and cached.updated_at <= deadline:
            raise CacheExpired

        return cached.data

    def write(self, path: PathLike, data: bytes) -> None:
        """Shelve the given data under the given path."""
        if self._max_size and self.size + len(data) >= self._max_size:
            self.shrink(target_size=self._max_size)

        with self._shelf as shelf:
            record = shelf.get(path)

            created_at = getattr(
                record,
                "created_at",
                self._timestamp(),
            )

            shelf[path] = CacheRecord(
                path,
                data,
                created_at,
                self._timestamp(),
            )

    def shrink(
        self,
        target_size: int,
        strategy: Optional[str] = None,
    ) -> None:
        """Shrink the shelf by evicting cached records according to the
        specified strategy until the target size is reached.

        If no strategy is specified, least-recently-updated will be
        used.
        """
        if not isinstance(target_size, int):
            raise TypeError(target_size, type(target_size))

        strategy = str(strategy or self._strategy).strip().casefold().replace("-", "_")

        if strategy in ("lru", "least_recently_updated"):
            while target_size <= self.size:
                with self._shelf as shelf:
                    evicted, _ = max(
                        shelf.items(),
                        key=lambda pair: sys.getsizeof(pair[1]),
                    )
                    del _, shelf[evicted]

        raise NotImplementedError


class FileStorage(Storage):
    @beartype
    def __init__(
        self,
        location: Optional[PathLike] = ".cache",
        max_size: Optional[int] = None,
    ):
        self.max_size = max_size
        self.location = Path(location)
        self.ensure_path(self.location)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(location={self.location}, max_size={self.max_size})>"

    def read(
        self,
        path: PathLike,
        deadline: Optional[dt.datetime] = None,
    ) -> bytes:
        final_path = self.location / path

        if not final_path.exists():
            final_path.touch(mode=0o755, exist_ok=True)

        if deadline is not None and self.mtime(final_path) < deadline:
            raise CacheExpired

        return self.read_file(self.location / path)

    def write(self, path: PathLike, data: bytes) -> None:
        final_path = self.location / path

        self.ensure_path(final_path.parent)

        if self.max_size and self.size(self.location) + len(data) > self.max_size:
            self.delete_least_recently_used(target_size=self.max_size)

        self.write_file(final_path, data)

    def delete_least_recently_used(self, target_size: int) -> None:
        """Removes the least recently used file from the cache. The least
        recently used file is the one with the smallest last access time.

        Args:
            target_size: The target size of the cache.
        """
        files = sorted(self.iterdir(self.location), key=self.atime, reverse=True)

        # find the set of most recently accessed files whose total size
        # is smaller than the target size
        i, size = 0, 0
        while size < target_size and i < len(files):
            size += self.size(files[i])
            i += 1

        # remove remaining files
        for f in files[i - 1 :]:
            self.delete(f)

    def clear(self) -> None:
        """Remove the directory with the cache along with all of its contents
        if it exists, otherwise just silently passes with no exceptions."""

        for f in self.iterdir(self.location):
            self.delete(f)
        self.rmdir(self.location)

    @abstractmethod
    def read_file(self, path: PathLike) -> bytes:
        """Read a file at a relative path inside the cache or raise
        FileNotFoundError if not found."""
        ...

    @abstractmethod
    def write_file(self, path: PathLike, data: bytes) -> None:
        """Write a file at a relative path inside the cache or raise
        FileNotFoundError if the cache directory doesn't exist."""
        ...

    @abstractmethod
    def ensure_path(self, path: PathLike) -> None:
        """Create an absolute path if it doesn't exist."""
        ...

    @abstractmethod
    def iterdir(self, path: PathLike) -> Union[Iterator[Path], list]:
        """Return an iterator through files within a directory indicated by
        path or an empty list if the path doesn't exist."""
        ...

    @abstractmethod
    def rmdir(self, path: PathLike) -> None:
        """Remove a directory.

        Silently pass if it doesn't exist.
        """
        ...

    @abstractmethod
    def mtime(self, path: PathLike) -> dt.datetime:
        """Get file last modified time."""
        ...

    @abstractmethod
    def atime(self, path: PathLike) -> dt.datetime:
        """Get file last accessed time."""
        ...

    @abstractmethod
    def size(self, path: PathLike) -> int:
        """Get the size in bytes for a file or a directory indicated by path.

        Zero if the path doesn't exist.
        """
        ...

    @abstractmethod
    def delete(self, path: PathLike) -> None:
        """Remove file or raise FileNotFoundError if not found."""
        ...


class LocalFileStorage(FileStorage):
    def read_file(self, path: Path) -> bytes:
        return path.read_bytes()

    def write_file(self, path: Path, data: bytes) -> None:
        path.write_bytes(data)

    def ensure_path(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def iterdir(self, path: Path) -> Iterable[Path]:
        return path.iterdir() if path.exists() else []

    def rmdir(self, path: Path) -> None:
        if path.exists():
            path.rmdir()

    def mtime(self, path: Path) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=dt.timezone.utc,
        )

    def atime(self, path: Path) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            path.stat().st_atime,
            tz=dt.timezone.utc,
        )

    def size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    def delete(self, path: Path) -> None:
        path.unlink()


class GoogleCloudStorage(FileStorage):
    @beartype
    def __init__(
        self,
        location: Optional[PathLike] = ".cache",
        max_size: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ) -> None:
        super().__init__(location, max_size)

        # Third-Party Imports
        import gcsfs

        self.fs = gcsfs.GCSFileSystem(**storage_options) if storage_options else gcsfs.GCSFileSystem()

    def read_file(self, path: PathLike) -> bytes:
        with self.fs.open(str(path), "rb") as f:
            return f.read()

    def write_file(self, path: PathLike, data: bytes) -> None:
        with self.fs.open(str(path), "wb") as f:
            f.write(data)

    def ensure_path(self, path: PathLike) -> None:
        if not self.fs.exists(str(path)):
            self.fs.makedirs(str(path), exist_ok=True)

    def iterdir(self, path: PathLike) -> Iterator[Path]:
        return self.fs.ls(str(path)) if self.fs.exists(str(path)) else []

    def rmdir(self, path: PathLike) -> None:
        self.fs.rm(str(path))

    def mtime(self, path: PathLike) -> dt.datetime:
        mtime = self.fs.info(str(path))["updated"]
        ts = dt.datetime.strptime(mtime.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        ts = ts.replace(tzinfo=dt.timezone.utc, microsecond=int(mtime[-4:-1]))
        return ts

    def atime(self, path: PathLike) -> dt.datetime:
        # fall back to mtime because last accessed (atime) is not available for GCS
        return self.mtime(str(path))

    def size(self, path: PathLike) -> int:
        return self.fs.info(str(path))["size"] if self.fs.exists(str(path)) else 0

    def delete(self, path: PathLike) -> None:
        self.fs.rm(str(path))
