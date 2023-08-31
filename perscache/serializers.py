"""perscache - serializers"""

# Future Imports
from __future__ import annotations

# Standard Library Imports
import io
import json
import pickle
from abc import (
    ABC,
    abstractmethod,
)

# Third-Party Imports
import cloudpickle
from beartype import beartype
from beartype.typing import (
    Any,
    Callable,
    Optional,
    Type,
    cast,
)


class Serializer(ABC):
    """Abstract serialization base class."""

    extension: str = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(extension='{self.extension}')>"

    @abstractmethod
    def dumps(self, data: Any) -> bytes:
        """Serialize the supplied data to its byte-level representation."""
        ...

    @abstractmethod
    def loads(self, data: bytes) -> Any:
        """Deserialize the supplied data to its appropriate Python form."""
        ...


@beartype
def make_serializer(
    class_name: str,
    extension: str,
    dumper: Callable[[Any], bytes],
    loader: Callable[[bytes], Any],
) -> Type[Serializer]:
    """Create a serializer class.

    Args:
        class_name (str): The name of the serializer class.
        extension (str): The file extension of the serialized data.
        dumper (callable): The function to serialize data.
                Takes a single argument and returns a bytes object.
        loader (callable): The function to deserialize data.
                Takes a single bytes object as argument and returns an object.
    """
    cls = cast(
        Type[Serializer],
        type(
            class_name,
            (Serializer,),
            {
                "extension": extension,
                "dumps": staticmethod(dumper),
                "loads": staticmethod(loader),
            },
        ),
    )

    return cls


CloudPickleSerializer = make_serializer(
    "CloudPickleSerializer",
    "pickle",
    cloudpickle.dumps,
    cloudpickle.loads,
)

JSONSerializer = make_serializer(
    "JSONSerializer",
    "json",
    lambda data: json.dumps(data).encode("utf-8"),
    lambda data: json.loads(data.decode("utf-8")),
)

PickleSerializer = make_serializer(
    "PickleSerializer",
    "pickle",
    pickle.dumps,
    pickle.loads,
)


class YAMLSerializer(Serializer):
    extension = "yaml"

    def dumps(self, data: Any) -> bytes:
        # Third-Party Imports
        import yaml

        return yaml.dump(data).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        # Third-Party Imports
        import yaml

        return yaml.safe_load(data.decode("utf-8"))


class ParquetSerializer(Serializer):
    """Serializes a Pandas DataFrame to a Parquet format with adjustable
    compression."""

    extension = "parquet"

    @beartype
    def __init__(self, compression: Optional[str] = "brotli"):
        self.compression = compression

    def __repr__(self):
        return f"<ParquetSerializer(extension='parquet', compression='{self.compression}')>"

    def dumps(self, data: Any) -> bytes:
        # Third-Party Imports
        import pyarrow.parquet

        buf = pyarrow.BufferOutputStream()

        # noinspection PyArgumentList
        pyarrow.parquet.write_table(
            pyarrow.Table.from_pandas(data),
            buf,
            compression=self.compression,
        )

        buf.flush()

        return buf.getvalue()

    def loads(self, data: bytes) -> Any:
        # Third-Party Imports
        import pyarrow.parquet

        return pyarrow.parquet.read_table(pyarrow.BufferReader(data)).to_pandas()


class CSVSerializer(Serializer):
    extension = "csv"

    def dumps(self, data: Any) -> bytes:
        # Third-Party Imports
        import pandas as pd

        return pd.DataFrame(data).to_csv().encode("utf-8")

    def loads(self, data: bytes) -> Any:
        # Third-Party Imports
        import pandas as pd

        return pd.read_csv(io.StringIO(data.decode("utf-8")), index_col=0)
