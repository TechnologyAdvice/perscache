# perscache

![](https://github.com/leshchenko1979/perscache/workflows/build/badge.svg)
![](https://img.shields.io/pypi/dm/perscache.svg)

An easy to use decorator for persistent memoization: like `functools.lrucache`, but results persist between runs and can be stored in any format to any storage.

- [Use cases](#use-cases)
- [Features](#features)
- [Getting started](#getting-started)
- [Make your own serialization and storage back-ends](#make-your-own-serialization-and-storage-backends)
- [API Reference](#api-reference)


## Use cases
- Cache the results of a function that uses a lot of resources: runs for a long time, consumes a lot of traffic, uses up paid API calls etc.
- Speed up retreival of data that doesn't change often.
- Inspect the results of a decorated function while debugging.

## Features
- Async functions supported (unlike in `joblib`).
- Automatic cache invalidation when the decorated function arguments or code have been changed.
- You can set to ignore changes in certain arguments of the decorated function.
- Various serialization formats: JSON, YAML, pickle, Parquet, CSV etc.
- Various storage backends:
    - local disk (_implemented_) or
    - cloud storage (_to be implemented soon_).
- You can set default serialization format and storage backend and then change them on a per-function basis.
- You can easily add new serialization formats and storage back-ends.
- Serialization and storage are separated into two different classes, so that you can mix various serialization formats and storage back-ends as you like - JSON to local storage, Pickle to AWS, Parquet to Google Cloud Storage etc.
- Local storage is file-based, so you can easily inspect cached results.
- Easy to swap out the storage back-end when switching environments.
- Automatic cleanup: results can be
    - removed from storage when the total storage size exceeds a given threshold (_implemented_) or
    - limited to one result per function (_to be implemented soon_)

## Getting started
### Installation
```bash
pip install perscache
```

### Basic usage
```python
from perscache import Cache

cache = Cache()

counter = 0

@cache.cache()
def get_data():
    print("Fetching data...")

    global counter
    counter += 1

    return "abc"

print(get_data())  # the function is called
# Fetching data...
# abc

print(get_data())  # the cache is used
# abc

print(counter)  # the function was called only once
# 1
```

### Changing parameters or the code of the function invalidates the cache
```python
@cache.cache()
def get_data(key):
    print("The function has been called...")
    return key

print(get_data("abc"))  # the function has been called
# The function has been called...
# abc

print(get_data("fgh"))  # the function has been called again
# The function has been called...
# fgh

print(get_data("abc"))  # using the cache
# abc

@cache.cache()
def get_data(key):
    print("This function has been changed...")
    return key

print(get_data("abc"))  # the function has been called again
# This function has been changed...
# abc

```
### Ignoring certain arguments
By specifying the arguments that should be ignored, you can still use the cache even in the values of these arguments have changed. **NOTE** that the decorated function should be called with ignored arguments specified as keyword arguments.
```python
@cache.cache(ignore=["ignore_this"])
def get_data(key, ignore_this):
    print("The function has been called...")
    return key

print(get_data("abc", ignore_this="ignore_1"))  # the function has been called
# The function has been called...
# abc

# using the cache although the the second argument is different
print(get_data("abc", ignore_this="ignore_2"))
# abc
```

### Changing the default serialization format and storage backend
```python
# set up serialization format and storage backend
cache = Cache(serializer=JSONSerializer(), storage=GCPStorage("bucket"))

...

# change the default serialization format
@cache.cache(serialization=PickleSerializer())
def get_data(key):
    ...
```

### Alternating cache settings depending on the environment
```python
import os

from perscache import Cache, NoCache
from perscache.storage import LocalFileStorage

if os.environ.get["DEBUG"]:
    cache = NoCache()
else:
    cache_location = (
        "gs://bucket/folder"
        if os.environ.get["GOOGLE_PROJECT_NAME"]
        else cache_location = "/tmp/cache"
    )
    cache = LocalFileStorage(location=cache_location)

@cache.cache()
def function():
    ...
```
### Inspecting cached results
When using `LocalFileStorage(location=...)`, the files are put into the directory specified by the `location` parameter.

The files are named like `<function_name>-<hash>.<serializer_extension>`, e.g. `get_data-9bf10a401d3d785317b2b35bcb5be1f2.json`.

### Automatic cleanup
When using `LocalFileStorage(max_size=...)`, the least recently used cache entries are automatically removed to keep the total cache size with the `max_size` limit.

## Make your own serialization and storage backends
Although you can use the standard `PickleSerializer()` for almost any type of data, sometimes you want to inspect the results of a decorated function by lookin into the cache files. This requires the data to be serialized in a human-readable format. But the included human-readable serializers (`JSONSerializer()`, `YAMLSerializer()`, `CSVSerializer()`) sometimes cannot process complex objects.

>To see which serializers are compatible with which data types, see the [compatibility.py](/perscache/compatibility.py) file.


That's when making your own serializer comes in handy.

To do this, you should:
1. Derive your own serialization classe from the abstract `Serializer` class and override the abstract methods. You should also provide the `extension` class variable that specifies the file extension.
2. Use your class with the `Cache` class.

```python
class MySerializer(Serializer):

    extension = "data"

    def dumps(self, data):
        ...

    def loads(self, data):
        ...

cache = Cache(serializer=MySerializer())
```

Making a custom storage backed is similar:
```python
class MyStorage(Storage):
    def read(self, filename):
        ...

    def write(self, filename, data):
        ...

cache = Cache(storage=MyStorage())
```

## API Reference
### class `Cache()`
#### Parameters
- `serializer (perscache.serializers.Serializer)`: a serializer class to use for cinverting stored data. Defaults to `perscache.serlializers.PickleSerializer`.

- `storage (perscache.storage.Storage)`: a storage back-end used to save and load data. Defaults to `perscache.storage.LocalFileStorage`.

#### decorator `Cache().cache()`
Tries to find a cached result of the decorated function in persistent storage. Returns the saved result if it was found, or calls the decorated function and caches its result.

The cache will be invalidated if the function code, its argument values or the cache serializer have been changed.
##### Arguments
- `ignore (Iterable[str])`: keyword arguments of the decorated function that will not be used in making the cache key. In other words, changes in these arguments will not invalidate the cache. Defaults to `None`.

- `serializer (perscache.serializers.Serializer)`: Overrides the default `Cache()` serializer. Defaults to `None`.

- `storage (perscache.storage.Storage)`: Overrides the default `Cache()` storage. Defaults to `None`.

### class `NoCache()`
This class has no parameters. It is useful to [alternate cache behaviour depending on the environment](#alternating-cache-settings-depending-on-the-environment).
#### decorator `NoCache().cache()`
The underlying function will be called every time the decorated function has been called and no caching will take place.

This decorator will ignore any parameters it has been given.
### Serializers
Serializers are imported from the `perscache.serializers` module.

See also [how to make your own serializer](#make-your-own-serialization-and-storage-backends).
#### class `perscache.serializers.JSONSerializer`
Uses the `json` module.
#### class `perscache.serializers.YAMLSerializer`
Uses the `yaml` module.
#### class `perscache.serializers.PickleSerializer`
Uses the `pickle` module.
#### class `perscache.serializers.CloudPickleSerializer`
Uses the `cloudpickle` module. It's the most capable serializer of all, able to process most of the data types. It's the default serializer for the `Cache` class.
#### class `perscache.serializers.CSVSerializer`
Uses the `pandas` module. Processes `pandas.DataFrame` objects.
#### class `perscache.serializers.ParquetSerializer`
Uses the `pyarrow` module. Processes `pandas.DataFrame` objects.
##### Parameters
- `compression (str)`: compression used by `pyarrow` to save the data. Defaults to `"brotli"`.

### Storage back-ends
Storage back-ends are imported from the `perscache.serializers` module.

See also [how to make your own storage back-end](#make-your-own-serialization-and-storage-backends).

#### class `perscache.storage.LocalFileStorage`
Keeps cache entries in separate files in a file system directory.

This is the default storage class used by `Cache`.
##### Parameters
- `location (str)`: a directory to store the cache files. Defaults to `".cache"`.

- `max_size (int)`: the maximum size for the cache. If set, then, before a new cache entry is written, the future size of the directory is calculated and the least recently used cache entries are removed. If `None`, the cache size grows indefinitely. Defaults to `None`.
