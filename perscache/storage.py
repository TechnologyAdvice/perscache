"""perscache - storage backends"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import datetime as dt
import os
import tempfile
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import Path

# Third-Party Imports
from beartype import beartype
from beartype.typing import (
    Iterator,
    Iterable,
    Optional,
    Union,
)

# Imports From Package Sub-Modules
from .compatibility import (
    PathLike,
    SpooledTempFile,
)


class CacheExpired(Exception):
    """Raised when a cache file / entry exists but is expired."""

    ...


class Storage(ABC):
    """Cache storage."""

    @abstractmethod
    def read(self, path: PathLike, deadline: dt.datetime) -> bytes:
        """Read the file at the given path and return its contents as bytes.

        If the file does not exist, raise FileNotFoundError. If the file
        is older than the given deadline, raise CacheExpired.
        """
        ...

    @abstractmethod
    def write(self, path: PathLike, data: bytes) -> None:
        """Write the file at the given path."""
        ...


class FileStorage(Storage):
    @beartype
    def __init__(
        self,
        location: Optional[PathLike] = ".cache",
        max_size: Optional[int] = None,
    ):
        self.location = Path(location)
        self.max_size = max_size

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(location={self.location}, max_size={self.max_size})>"

    def read(self, path: PathLike, deadline: dt.datetime) -> bytes:
        final_path = self.location / path

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


class SpooledTempFileStorage(LocalFileStorage):
    """Temporary storage that acts like it's backed by a spooled temporary
    file."""

    max_size: int
    directory: PathLike
    __spools__: dict[PathLike, SpooledTempFile]

    @beartype
    def __init__(
        self,
        max_size: Optional[int] = None,
        directory: Optional[PathLike] = None,
    ) -> None:
        self.__spools__ = dict()
        self.max_size = max_size or 5_000_000
        self.directory = self._resolve_dir(directory)

        super().__init__()

    @staticmethod
    def _resolve_dir(path: Optional[PathLike]) -> Path:
        """Resolve the supplied path."""
        path = Path(path or Path(tempfile.gettempdir()) / "perscache")

        if path.is_file():
            path = path.parent

        path.mkdir(
            mode=0o755,
            parents=True,
            exist_ok=True,
        )

        return path

    def _spool_for(self, path: PathLike) -> SpooledTempFile:
        """Get the corresponding "spool" for the specified "path"."""
        spool = self.__spools__.get(path)

        if not spool:
            spool = self.__spools__[path] = SpooledTempFile(
                max_size=self.max_size,
                directory=self.directory,
            )

        return spool

    def read_file(self, path: PathLike) -> bytes:
        spool = self._spool_for(path=path)

        data = spool.getvalue()

        return data

    def write_file(self, path: PathLike, data: bytes) -> None:
        """Write the supplied data into the spooled cache."""
        spool = self._spool_for(path=path)
        spool.write(data=data)

    def iterdir(self, path: PathLike) -> Iterable[SpooledTempFile]:
        return self.__spools__.values()

    def rmdir(self, path: PathLike) -> None:
        spool = self.__spools__.pop(path, None)

        if spool is not None:
            spool.delete()

    def mtime(self, path: PathLike) -> dt.datetime:
        """Timestamp of the most recent content modification made to the specified `path`."""
        return self._spool_for(path).mtime

    def atime(self, path: PathLike) -> dt.datetime:
        """Timestamp of the most recent access of the specified `path`."""
        return self._spool_for(path).atime

    def size(self, path: PathLike) -> int:
        return self._spool_for(path).size()

    def delete(self, path: PathLike) -> None:
        return self._spool_for(path).delete()


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
