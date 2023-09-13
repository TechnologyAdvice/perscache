"""An easy to use decorator for persistent memoization.

Like `functools.lrucache`, but results can be saved in any format to any storage.
"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import asyncio as aio
import datetime as dt
import inspect
from functools import partial

# Third-Party Imports
import wrapt
from beartype import beartype
from beartype.typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    TypeVar,
)
from icontract import require

# Imports From Package Sub-Modules
from ._logger import (
    logger,
    trace,
)
from .compatibility import (
    CachedAsyncCallable,
    CachedCallable,
    CachedFunction,
    CachedValue,
    hash_all,
    is_async,
)
from .serializers import (
    CloudPickleSerializer,
    Serializer,
)
from .storage import (
    CacheExpired,
    CacheMiss,
    LocalFileStorage,
    Storage,
)


def valid_ttl(ttl: Optional[dt.timedelta] = None) -> bool:
    """Checks if the supplied ttl value is valid (greater than 0 seconds)."""
    return ttl is None or (isinstance(ttl, dt.timedelta) and ttl > dt.timedelta(seconds=0))


def valid_ignores(self: "_CachedFunction", fn: CachedFunction) -> bool:
    """Checks if any of the specified parameters to ignore are missing from the
    supplied function's calling signature."""

    ignored = set(self.ignore or tuple())
    parameters = dict(inspect.signature(fn).parameters)
    variadic_params = {
        parameters.pop(key)
        for key, value in tuple(parameters.items())
        if value.kind
        in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    }
    valid = ignored.issubset(parameters)

    return bool(valid or variadic_params)


WrappedInstance = TypeVar("WrappedInstance")


class Cache:
    """A cache that can be used to memoize functions."""

    __slots__ = (
        "storage",
        "hash_func",
        "serializer",
        "__locks_store__",
    )

    storage: Storage
    serializer: Serializer
    hash_func: Callable[..., str]
    __locks_store__: Optional[dict[str, aio.Future]]

    @beartype
    def __init__(
        self,
        serializer: Serializer = None,
        storage: Storage = None,
        hash_func: Optional[Callable[..., str]] = None,
    ) -> None:
        """Initialize the cache.

        Args:
            serializer: The serializer to use. If not specified, CloudPickleSerializer is used.
            storage: The storage to use. If not specified, LocalFileStorage is used.
        """

        storage, serializer = (
            (storage if storage is not None else LocalFileStorage()),
            (serializer if serializer is not None else CloudPickleSerializer()),
        )

        if not isinstance(storage, Storage):
            logger.warn(f"Unsupported storage backend: {storage!r}")
            storage = LocalFileStorage()
            logger.warn(f"Falling back to default storage: {storage!r}")

        if not isinstance(serializer, Serializer):
            logger.warn(f"Unsupported serializer: {serializer!r}")
            serializer = CloudPickleSerializer()
            logger.warn(f"Falling back to default serializer: {serializer!r}")

        hash_func = hash_func or hash_all

        if not callable(hash_func):
            logger.warn(f"Unsupported hash function: {hash_func!r}")
            hash_func = hash_all
            logger.warn(f"Falling back to default hash function: {hash_func!r}")

        self.__locks_store__ = None
        self.storage, self.serializer, self.hash_func = (
            storage,
            serializer,
            hash_func,
        )

    def __repr__(self) -> str:
        return f"<Cache(serializer={self.serializer}, storage={self.storage})>"

    @beartype
    @require(valid_ttl, "ttl must be positive.")
    def __call__(
        self,
        fn: Optional[CachedFunction] = None,
        *,
        storage: Optional[Storage] = None,
        ttl: Optional[dt.timedelta] = None,
        ignore: Optional[Iterable[str]] = None,
        serializer: Optional[Serializer] = None,
    ):
        """Cache the value of the wrapped function.

        Tries to find a cached result of the decorated function in persistent storage.
        Returns the saved result if it was found, or calls the decorated function
        and caches its result.

        The cache will be invalidated if the function code, its argument values or
        the cache serializer have been changed.

        Args:
            ignore: A list of argument names to ignore when hashing the function.
            serializer: The serializer to use. If not specified, the default serializer is used.
                    Defaults to None.
            storage: The storage to use. If not specified, the default storage is used.
                    Defaults to None.
            ttl: The expiration time of the cache. If None, the cache will never expire.
                    Defaults to None.
        """

        if isinstance(ignore, str):
            ignore = [ignore]

        wrapper = _CachedFunction(
            ttl=ttl,
            cache=self,
            ignore=ignore,
            storage=storage or self.storage,
            serializer=serializer or self.serializer,
        )

        # The decorator should work both with and without parentheses
        return wrapper if fn is None else wrapper(fn)

    cache = __call__  # Alias for backwards compatibility.

    @staticmethod
    @trace
    def _get(
        key: str,
        serializer: Serializer,
        storage: Storage,
        deadline: dt.datetime,
    ) -> CachedValue:
        """Get the specified key from the supplied storage."""
        try:
            data = storage.read(key, deadline)
            return serializer.loads(data)
        except FileNotFoundError as error:
            raise CacheMiss from error

    @staticmethod
    @trace
    def _set(
        key: str,
        value: CachedValue,
        serializer: Serializer,
        storage: Storage,
    ) -> None:
        """Set the specified key to the supplied value in the supplied
        storage."""
        data = serializer.dumps(value)
        storage.write(key, data)

    @property
    def _async_locks(self) -> dict[str, aio.Future]:
        """The cache instance's async lock store."""
        locks = self.__locks_store__

        if locks is None:
            locks = self.__locks_store__ = dict()
        return locks

    def _get_hash(
        self,
        fn: CachedFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        serializer: Serializer,
        ignore: Iterable[str] = tuple(),
        instance: Optional[WrappedInstance] = None,
    ) -> str:
        """Get the hash value for the supplied combination of function,
        arguments, and serializer."""

        ignore = ignore or tuple()
        fn = getattr(fn, "__func__", fn)
        instance = instance or WrappedInstance
        args = args if instance is WrappedInstance else (instance, *args)

        # Remove ignored arguments from the arguments tuple and kwargs dict
        fn_sig = inspect.signature(fn)
        fn_args = fn_sig.bind(*args, **kwargs).arguments
        arg_dict = {
            arg: id(value) if value is instance else value
            for (
                arg,
                value,
            ) in fn_args.items()
            if arg not in ignore
        }

        for kw_args in (
            arg_dict.pop(key, dict())
            for (
                key,
                param,
            ) in fn_sig.parameters.items()
            if param.kind.name == "VAR_KEYWORD"
        ):
            arg_dict.update(
                **{
                    key: value
                    for (
                        key,
                        value,
                    ) in kw_args.items()
                    if key not in ignore
                }
            )

        fn_src, serializer_name = inspect.getsource(fn), type(serializer).__name__

        hashed = self.hash_func(fn_src, serializer_name, arg_dict)

        return hashed

    @staticmethod
    def _get_filename(fn: CachedFunction, key: str, serializer: Serializer) -> str:
        return f"{fn.__name__}-{key}.{serializer.extension}"


class NoCache:
    """A class used to turn off caching.

    Example:
    ```
    cache = NoCache() if os.environ["DEBUG"] else Cache()

    @cache.cache
    def function():
        ...
    ```
    """

    @beartype
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __call__(self, *_: Any, **__: Any) -> CachedFunction:
        """Will call the decorated function every time and return its result
        without any caching."""

        def _decorator(fn: CachedFunction) -> CachedCallable | CachedAsyncCallable:
            return self._async_wrapper if is_async(fn) else self._wrapper

        return _decorator

    cache = __call__  # Alias for backwards compatibility.

    @wrapt.decorator
    def _wrapper(
        self,
        fn: CachedCallable,
        _: Optional[WrappedInstance],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Callable[[CachedCallable], CachedCallable]:
        return fn(*args, **kwargs)

    @wrapt.decorator
    async def _async_wrapper(
        self,
        fn: CachedAsyncCallable,
        _: Optional[WrappedInstance],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Callable[[CachedAsyncCallable], CachedAsyncCallable]:
        return await fn(*args, **kwargs)


class _CachedFunction:
    """An internal class used as a wrapper."""

    __slots__ = (
        "ttl",
        "cache",
        "ignore",
        "storage",
        "serializer",
    )

    ttl: Optional[dt.timedelta]
    cache: Cache
    ignore: Optional[Iterable[str]]
    storage: Storage
    serializer: Serializer

    @beartype
    def __init__(
        self,
        cache: Cache,
        serializer: Serializer,
        storage: Storage,
        ignore: Optional[Iterable[str]] = None,
        ttl: Optional[dt.timedelta] = None,
    ):
        self.ttl = ttl
        self.cache = cache
        self.ignore = ignore
        self.storage = storage
        self.serializer = serializer

    @require(valid_ignores, "Ignored parameters not found in the function signature.")
    def __call__(self, fn: CachedFunction, *args: Any, **kwargs: Any) -> CachedFunction:
        """Return the correct wrapper."""
        wrapper = self._wrapper if not is_async(fn) else self._async_wrapper
        wrapped = wrapper(fn, *args, **kwargs)
        return wrapped

    @wrapt.decorator
    def _wrapper(
        self,
        fn: CachedCallable,
        instance: Optional[WrappedInstance],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Callable[[CachedCallable], CachedCallable]:
        hashed = self.cache._get_hash(
            fn,
            args=args,
            kwargs=kwargs,
            instance=instance,
            ignore=self.ignore,
            serializer=self.serializer,
        )
        cache_key = self.cache._get_filename(
            fn,
            hashed,
            self.serializer,
        )

        fn_name = getattr(fn, "__qualname__", fn.__name__)
        logger.debug(f"Getting cached result for: <{fn_name}/{hashed}>")

        try:
            value = self.cache._get(
                cache_key,
                self.serializer,
                self.storage,
                self.deadline,
            )
        except (CacheMiss, CacheExpired) as exception:
            log_prefix = (
                "Cache miss"
                if isinstance(exception, CacheMiss)
                else "Cached data expired"
                if isinstance(exception, CacheExpired)
                else "Cache error"
            )
            logger.debug(f"{log_prefix} for <{fn_name}/{hashed}>: {exception}")

            value = fn(*args, **kwargs)

            self.cache._set(
                cache_key,
                value,
                self.serializer,
                self.storage,
            )
        return value

    @property
    def _async_locks(self) -> dict[str, aio.Future]:
        """The internal async lock store the function's associated cache
        instance."""
        return self.cache._async_locks

    @wrapt.decorator
    async def _async_wrapper(
        self,
        fn: CachedAsyncCallable,
        instance: Optional[WrappedInstance],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Callable[[CachedAsyncCallable], CachedAsyncCallable]:
        hashed = self.cache._get_hash(
            fn,
            args=args,
            kwargs=kwargs,
            instance=instance,
            ignore=self.ignore,
            serializer=self.serializer,
        )
        cache_key = self.cache._get_filename(
            fn,
            hashed,
            self.serializer,
        )

        fn_base_name = getattr(fn, "__qualname__", fn.__name__)
        fn_module = getattr(inspect.getmodule(fn), "__name__", None)
        fn_name = ".".join(filter(None, (fn_module, fn_base_name)))
        lock_key = f"{fn_name}::{hashed}"

        cache_lock: Optional[aio.Future] = self._async_locks.get(lock_key)

        if cache_lock is None or cache_lock.done():
            cache_lock = self._async_locks[lock_key] = aio.get_running_loop().create_future()
            logger.debug(f"Cache lock acquired: <{fn_base_name}/{hashed}>")
        else:
            logger.debug(f"Waiting for outstanding cache lock: <{fn_base_name}/{hashed}>")
            await cache_lock

        try:
            value = self.cache._get(
                cache_key,
                self.serializer,
                self.storage,
                self.deadline,
            )
        except (CacheMiss, CacheExpired) as exception:
            log_prefix = (
                "Cache miss"
                if isinstance(exception, CacheMiss)
                else "Cached data expired"
                if isinstance(exception, CacheExpired)
                else "Cache error"
            )
            logger.debug(f"{log_prefix} for <{fn_base_name}/{hashed}>: {exception}")

            value = await fn(*args, **kwargs)

            self.cache._set(
                cache_key,
                value,
                self.serializer,
                self.storage,
            )

        if not cache_lock.done():
            cache_lock.set_result(True)
            del self._async_locks[lock_key]
            logger.debug(f"Cache lock released: <{fn_base_name}/{hashed}>")

        return value

    @property
    def deadline(self) -> Optional[dt.datetime]:
        """Return the deadline for the cache."""
        if self.ttl:
            return dt.datetime.now(dt.timezone.utc) - self.ttl
