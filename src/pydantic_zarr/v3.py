from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Never,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import zarr
from pydantic import BaseModel, BeforeValidator, Field

from pydantic_zarr.core import StrictBase
from pydantic_zarr.v2 import stringify_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import zarr
    from zarr.abc.store import Store


TAttr = TypeVar("TAttr", bound=dict[str, Any])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])

NodeType = Literal["group", "array"]

BoolFillValue = bool
IntFillValue = int
# todo: introduce a type that represents hexadecimal representations of floats
FloatFillValue = Literal["Infinity", "-Infinity", "NaN"] | float
ComplexFillValue = tuple[FloatFillValue, FloatFillValue]
RawFillValue = tuple[int, ...]

FillValue = BoolFillValue | IntFillValue | FloatFillValue | ComplexFillValue | RawFillValue


class NamedConfig(StrictBase):
    name: str
    configuration: dict[str, Any] | BaseModel | None


class RegularChunkingConfig(StrictBase):
    chunk_shape: list[int]


class RegularChunking(NamedConfig):
    name: Literal["regular"] = "regular"
    configuration: RegularChunkingConfig


class DefaultChunkKeyEncodingConfig(StrictBase):
    separator: Literal[".", "/"] = "/"


class DefaultChunkKeyEncoding(NamedConfig):
    name: Literal["default"] = "default"
    configuration: DefaultChunkKeyEncodingConfig = Field(default=DefaultChunkKeyEncodingConfig())


class NodeSpec(StrictBase):
    """
    The base class for V3 ArraySpec and GroupSpec.

    Attributes
    ----------

    zarr_format: Literal[3]
        The Zarr version represented by this node. Must be 3.
    """

    zarr_format: Literal[3] = 3


DtypeStr = Annotated[str, BeforeValidator(stringify_dtype)]


class ArraySpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 3 Array.

    Attributes
    ----------

    node_type: Literal['array']
        The node type. Must be the string 'array'.
    attributes: TAttr
        User-defined metadata associated with this array.
    shape: Sequence[int]
        The shape of this array.
    data_type: str
        The data type of this array.
    chunk_grid: NamedConfig
        A `NamedConfig` object defining the chunk shape of this array.
    chunk_key_encoding: NamedConfig
        A `NamedConfig` object defining the chunk_key_encoding for the array.
    fill_value: FillValue
        The fill value for this array.
    codecs: Sequence[NamedConfig]
        The sequence of codices for this array.
    storage_transformers: Optional[Sequence[NamedConfig]]
        An optional sequence of `NamedConfig` objects that define the storage
        transformers for this array.
    dimension_names: Optional[Sequence[str]]
        An optional sequence of strings that gives names to each axis of the array.
    """

    node_type: Literal["array"] = "array"
    attributes: TAttr = cast(TAttr, {})
    shape: tuple[int, ...]
    data_type: DtypeStr | NamedConfig
    chunk_grid: NamedConfig  # todo: validate this against shape
    chunk_key_encoding: NamedConfig  # todo: validate this against shape
    fill_value: FillValue  # todo: validate this against the data type
    codecs: tuple[NamedConfig, ...]
    storage_transformers: tuple[NamedConfig, ...]
    dimension_names: tuple[str | None, ...] | None  # todo: validate this against shape

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[Any],
        *,
        attributes: Literal["auto"] | TAttr = "auto",
        chunk_grid: Literal["auto"] | NamedConfig = "auto",
        chunk_key_encoding: Literal["auto"] | NamedConfig = "auto",
        fill_value: Literal["auto"] | FillValue = "auto",
        codecs: Literal["auto"] | Sequence[NamedConfig] = "auto",
        storage_transformers: Literal["auto"] | Sequence[NamedConfig] = "auto",
        dimension_names: Literal["auto"] | Sequence[str | None] = "auto",
    ) -> Self:
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array :
            object that conforms to the numpy array API.
            The shape and dtype of this object will be used to construct an ArraySpec.
            If the `chunks` keyword argument is not given, the shape of the array will
            be used for the chunks.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """
        if attributes == "auto":
            attributes_actual = cast(TAttr, auto_attributes(array))
        else:
            attributes_actual = attributes

        if chunk_grid == "auto":
            chunk_grid_actual = auto_chunk_grid(array)
        else:
            chunk_grid_actual = chunk_grid

        chunk_key_actual: NamedConfig
        if chunk_key_encoding == "auto":
            chunk_key_actual = DefaultChunkKeyEncoding()
        else:
            chunk_key_actual = chunk_key_encoding

        if fill_value == "auto":
            fill_value_actual = auto_fill_value(array)
        else:
            fill_value_actual = fill_value

        if codecs == "auto":
            codecs_actual = auto_codecs(array)
        else:
            codecs_actual = codecs

        storage_transformers_actual: Sequence[NamedConfig]
        if storage_transformers == "auto":
            storage_transformers_actual = auto_storage_transformers(array)
        else:
            storage_transformers_actual = storage_transformers

        dimension_names_actual: Sequence[str | None]
        if dimension_names == "auto":
            dimension_names_actual = auto_dimension_names(array)
        else:
            dimension_names_actual = dimension_names

        return cls(
            shape=array.shape,
            data_type=str(array.dtype),
            chunk_grid=chunk_grid_actual,
            attributes=attributes_actual,
            chunk_key_encoding=chunk_key_actual,
            fill_value=fill_value_actual,
            codecs=tuple(codecs_actual),
            storage_transformers=tuple(storage_transformers_actual),
            dimension_names=tuple(dimension_names_actual),
        )

    @classmethod
    def from_zarr(cls, array: zarr.Array) -> Self:
        """
        Create an ArraySpec from a `zarr.Array`.

        Parameters
        ----------
        array : zarr.Array

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided zarr
        array.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> ArraySpec.from_zarr(x)
        ArraySpec(zarr_format=2, attributes={}, shape=(10, 10), chunks=(10, 10), dtype='<f8', fill_value=0.0, order='C', filters=None, dimension_separator='.', compressor={'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0})

        """
        from zarr.core.metadata import ArrayV3Metadata

        if not isinstance(array.metadata, ArrayV3Metadata):
            raise ValueError("Only zarr v3 arrays are supported")  # noqa: TRY004

        print(array.compressors)

        return cls(
            attributes=array.attrs.asdict(),
            shape=array.shape,
            data_type=array.dtype.str,
            chunk_grid=array.metadata.chunk_grid.to_dict(),
            chunk_key_encoding=array.metadata.chunk_key_encoding.to_dict(),
            fill_value=array.fill_value,
            codecs=[c.to_dict() for c in array.compressors],
            storage_transformers=array.metadata.storage_transformers,
            dimension_names=array.metadata.dimension_names,
        )

    def to_zarr(self, store: Store, path: str, overwrite: bool = False) -> zarr.Array:
        """
        Serialize an ArraySpec to a zarr array at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.abc.store.Store
            The storage backend that will manifest the array.
        path : str
            The location of the array inside the store.
        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.

        Returns
        -------
        A zarr array that is structurally identical to the ArraySpec.
        This operation will create metadata documents in the store.
        """
        raise NotImplementedError


class GroupSpec(NodeSpec, Generic[TAttr, TItem]):
    """
    A model of a Zarr Version 3 Group.

    Attributes
    ----------

    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr
        The user-defined attributes of this group.
    members: dict[str, TItem]
        The members of this group. `members` is a dict with string keys and values that
        must inherit from either ArraySpec or GroupSpec.
    """

    node_type: Literal["group"] = "group"
    attributes: TAttr = cast(TAttr, {})
    members: dict[str, TItem] = {}  # noqa: RUF012

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> GroupSpec[TAttr, TItem]:
        """
        Create a GroupSpec from a zarr group. Subgroups and arrays contained in the zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .members attribute of the parent
        GroupSpec. This occurs recursively, so the entire zarr hierarchy below a given
        group can be represented as a GroupSpec.

        Parameters
        ----------
        group : zarr group

        Returns
        -------
        An instance of GroupSpec that represents the structure of the zarr hierarchy.
        """

        raise NotImplementedError

    def to_zarr(self, store: Store, path: str, overwrite: bool = False) -> Never:
        """
        Serialize a GroupSpec to a zarr group at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.abc.store.Store
            The storage backend that will manifest the group and its contents.
        path : str
            The location of the group inside the store.
        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.

        Returns
        -------
        A zarr group that is structurally identical to the GroupSpec.
        This operation will create metadata documents in the store.
        """
        raise NotImplementedError


"""
@overload
def from_zarr(element: zarr.Array) -> ArraySpec: ...


@overload
def from_zarr(element: zarr.Group) -> GroupSpec: ...
"""


def from_zarr(element: zarr.Array | zarr.Group) -> ArraySpec | GroupSpec:
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.

    Parameters
    ----------
    element : a zarr Array or zarr Group

    Returns
    -------
    An instance of GroupSpec or ArraySpec that represents the
    structure of the zarr group or array.
    """

    raise NotImplementedError


@overload
def to_zarr(
    spec: ArraySpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Group: ...


def to_zarr(
    spec: ArraySpec | GroupSpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Array | zarr.Group:
    """
    Serialize a GroupSpec or ArraySpec to a zarr group or array at a specific path in
    a zarr store.

    Parameters
    ----------
    spec : GroupSpec or ArraySpec
        The GroupSpec or ArraySpec that will be serialized to storage.
    store : instance of zarr.abc.store.Store
        The storage backend that will manifest the group or array.
    path : str
        The location of the group or array inside the store.
    overwrite : bool
        Whether to overwrite an existing array or group at the path. If overwrite is
        False and an array or group already exists at the path, an exception will be
        raised. Defaults to False.

    Returns
    -------
    A zarr Group or Array that is structurally equivalent to the spec object.
    This operation will create metadata documents in the store.

    """
    if isinstance(spec, (ArraySpec, GroupSpec)):
        result = spec.to_zarr(store, path, overwrite=overwrite)
    else:
        msg = ("Invalid argument for spec. Expected an instance of GroupSpec or ",)  # type: ignore[unreachable]
        f"ArraySpec, got {type(spec)} instead."
        raise TypeError(msg)

    return result


def auto_attributes(array: Any) -> dict[str, Any]:
    if hasattr(array, "attributes"):
        return array.attributes
    return {}


def auto_chunk_grid(array: Any) -> NamedConfig:
    if hasattr(array, "chunk_shape"):
        return array.chunk_shape
    elif hasattr(array, "shape"):
        return RegularChunking(configuration=RegularChunkingConfig(chunk_shape=list(array.shape)))
    raise ValueError("Cannot get chunk grid from object without .shape attribute")


def auto_fill_value(array: Any) -> FillValue:
    if hasattr(array, "fill_value"):
        return array.fill_value
    elif hasattr(array, "dtype"):
        kind = np.dtype(array.dtype).kind
        if kind == "?":
            return False
        elif kind in ["i", "u"]:
            return 0
        elif kind in ["f"]:
            return "NaN"
        elif kind in ["c"]:
            return ("NaN", "NaN")
        else:
            raise ValueError(f"Cannot determine default fill value for data type {kind}")
    raise ValueError("Cannot determine default data type for object without shape attribute.")


def auto_codecs(array: Any) -> Sequence[NamedConfig]:
    if hasattr(array, "codecs"):
        return array.codecs
    return []


def auto_storage_transformers(array: Any) -> list[NamedConfig]:
    if hasattr(array, "storage_transformers"):
        return array.storage_transformers
    return []


def auto_dimension_names(array: Any) -> list[str | None]:
    if hasattr(array, "dimension_names"):
        return array.dimension_names
    return [None] * np.asanyarray(array, copy=False).ndim
