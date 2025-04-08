from __future__ import annotations

import math
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NotRequired,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import tensorstore
    import zarr

    ZArrayLike = zarr.Array | tensorstore.TensorStore | "ArrayLike[TShape]"

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class StoreSpec(TypedDict):
    url: str
    engine: NotRequired[Literal["tensorstore", "zarr-python"]]
    configuration: NotRequired[Mapping[str, Any]]


if TYPE_CHECKING:
    from tensorstore import KvStore
    from zarr.abc.store import Store

    StoreLike = Store | KvStore | str | StoreSpec

__all__ = [
    "AccessMode",
    "ArrayV2Config",
    "ArrayV3Config",
    "IncEx",
    "StrictBase",
    "ensure_key_no_path",
    "ensure_member_name",
    "model_like",
]

IncEx: TypeAlias = set[int] | set[str] | dict[int, Any] | dict[str, Any] | None

AccessMode: TypeAlias = Literal["w", "w+", "r", "a"]


class StrictBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def ensure_member_name(data: Any) -> str:
    """
    If the input is a string, then ensure that it is a valid
    name for a subnode in a zarr group
    """
    if isinstance(data, str):
        if "/" in data:
            raise ValueError(
                f'Strings containing "/" are invalid. Got {data}, which violates this rule.'
            )
        if data in ("", ".", ".."):
            raise ValueError(f"The string {data} is not a valid member name.")
        return data
    raise TypeError(f"Expected a str, got {type(data)}.")


def ensure_key_no_path(data: Any) -> Any:
    if isinstance(data, Mapping):
        [ensure_member_name(key) for key in data]
    return data


def model_like(a: BaseModel, b: BaseModel, exclude: IncEx = None, include: IncEx = None) -> bool:
    """
    A similarity check for a pair of `pydantic.BaseModel`, parametrized over included or excluded fields.

    """

    a_dict = a.model_dump(exclude=exclude, include=include)
    b_dict = b.model_dump(exclude=exclude, include=include)

    return a_dict == b_dict


TShape = TypeVar("TShape", bound=tuple[int, ...])


@runtime_checkable
class ArrayLike(Protocol, Generic[TShape]):
    # shape has to be annotated with a generic type parameter for numpy
    shape: TShape

    @property
    def dtype(self) -> Any: ...


class GroupLike(Protocol):
    @property
    def attrs(self) -> Mapping[str, Any]: ...

    def members(
        self, max_depth: int | None
    ) -> tuple[tuple[str, ArrayLike[Any] | GroupLike], ...]: ...


def guess_chunks(shape: tuple[int, ...], item_size: int) -> tuple[int, ...]:
    """
    Calculate suitable automatic chunk sizes for an N-dimensional array.

    Parameters:
    shape :  tuple[int, ...]
        Shape of the array
    item_size : int
      Size of each element in bytes

    Returns:
    tuple: Chunk sizes for each axis of the array
    """
    # Calculate the total size of the array in bytes
    total_size = math.prod(shape) * item_size

    # Calculate the ideal chunk size in bytes, aiming for 256 KB to 100 MB
    ideal_chunk_size = min(max(256 * 1024, total_size // 100), 100 * 1024 * 1024)

    # Calculate the chunk size for each axis, trying to keep the number of chunks low
    chunk_sizes = []
    remaining_size = ideal_chunk_size
    for axis_size in shape:
        axis_chunk_size = min(axis_size, remaining_size // item_size)
        axis_chunk_size = 2 ** math.floor(math.log2(axis_chunk_size))
        chunk_sizes.append(axis_chunk_size)
        remaining_size -= axis_chunk_size * item_size

    return tuple(chunk_sizes)


class CodecConfigV2(TypedDict, total=False):
    id: str


class ArrayMetadataV2Config(TypedDict):
    zarr_format: Literal[2]
    shape: tuple[int, ...]
    dtype: str | tuple[object]
    chunks: tuple[int, ...]
    fill_value: object
    order: Literal["C", "F"]
    compressor: CodecConfigV2 | None
    filters: tuple[CodecConfigV2] | None
    dimension_separator: Literal["/", "."]


class ArrayV2Config(ArrayMetadataV2Config):
    attributes: NotRequired[Mapping[str, object]]


class GroupMetadataV2Config(TypedDict):
    zarr_format: Literal[2]


class GroupV2Config(GroupMetadataV2Config):
    attributes: NotRequired[Mapping[str, object]]
    members: NotRequired[Mapping[str, ArrayV2Config | GroupV2Config]]


class NamedConfig(TypedDict):
    name: str
    configuration: NotRequired[Mapping[str, object]]


class ArrayV3Config(TypedDict):
    zarr_format: Literal[3]
    node_type: Literal["array"]
    shape: tuple[int, ...]
    data_type: str | NamedConfig
    fill_value: object
    chunk_grid: NamedConfig
    chunk_key_encoding: NamedConfig
    codecs: tuple[NamedConfig, ...]
    attributes: Mapping[str, object]
    dimension_names: tuple[str] | None


class GroupV3Config(TypedDict):
    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: Mapping[str, object]


class RegularChunks(TypedDict):
    read_shape: tuple[int, ...]
    write_shape: tuple[int, ...] | None


class RectilinearChunks(TypedDict):
    read_shape: tuple[tuple[int, ...], ...]
    write_shape: tuple[tuple[int, ...], ...] | None
