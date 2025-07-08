from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Never,
    NotRequired,
    Self,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import zarr
from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import ReadOnly

from pydantic_zarr.core import IncEx, StrictBase, tuplify_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import zarr
    from zarr.abc.store import Store


TBaseAttr = Mapping[str, object]
TBaseMember = Union["GroupSpec[TBaseAttr, TBaseMember]", "ArraySpec[TBaseAttr]"]

TAttr = TypeVar("TAttr", bound=TBaseAttr)
TItem = TypeVar("TItem", bound=TBaseMember)

NodeType = Literal["group", "array"]

BoolFillValue = bool
IntFillValue = int
# todo: introduce a type that represents hexadecimal representations of floats
FloatFillValue = Literal["Infinity", "-Infinity", "NaN"] | float
ComplexFillValue = tuple[FloatFillValue, FloatFillValue]
RawFillValue = tuple[int, ...]

FillValue = BoolFillValue | IntFillValue | FloatFillValue | ComplexFillValue | RawFillValue


class ArrayConfig(TypedDict):
    order: NotRequired[ReadOnly[Literal["C", "F"]]]
    write_empty_chunks: NotRequired[ReadOnly[bool]]


class NamedConfig(StrictBase):
    name: str
    configuration: Mapping[str, object] | BaseModel


class RegularChunkingConfig(StrictBase):
    chunk_shape: tuple[int, ...]


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


def stringify_dtype_v3(dtype: npt.DTypeLike | Mapping[str, object]) -> Mapping[str, object] | str:
    """
    Todo: refactor this when the zarr python dtypes work is released
    """
    if isinstance(dtype, str | Mapping):
        return dtype
    else:
        match np.dtype(dtype):
            case np.dtypes.Int8DType():
                return "int8"
            case np.dtypes.Int16DType():
                return "int16"
            case np.dtypes.Int32DType():
                return "int32"
            case np.dtypes.Int64DType():
                return "int64"
            case np.dtypes.UInt8DType():
                return "uint8"
            case np.dtypes.UInt16DType():
                return "uint16"
            case np.dtypes.UInt32DType():
                return "uint32"
            case np.dtypes.UInt64DType():
                return "uint64"
            case np.dtypes.Float16DType():
                return "float16"
            case np.dtypes.Float32DType():
                return "float32"
            case np.dtypes.Float16DType():
                return "float64"
            case np.dtypes.Float32DType():
                return "complex64"
            case np.dtypes.Complex128DType():
                return "complex128"
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")


DtypeStr = Annotated[str, BeforeValidator(stringify_dtype_v3)]


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

    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",  # noqa: PYI051
        include: IncEx | None = None,  # type: ignore[override]
        exclude: IncEx | None = None,  # type: ignore[override]
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: (bool | Literal["none", "warn", "error"]) = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        """
        Override this method because the Zarr V3 spec requires that the dimension_names
        field be omitted from metadata entirely if it's empty.
        """
        d = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )

        if d["dimension_names"] is None:
            d.pop("dimension_names")
        return d

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
        from zarr.core.metadata.v3 import V3JsonEncoder

        if not isinstance(array.metadata, ArrayV3Metadata):
            raise ValueError("Only zarr v3 arrays are supported")  # noqa: TRY004
        meta_json = json.loads(
            json.dumps(array.metadata.to_dict(), cls=V3JsonEncoder), object_hook=tuplify_json
        )
        return cls(
            attributes=meta_json["attributes"],
            shape=array.shape,
            data_type=meta_json["data_type"],
            chunk_grid=meta_json["chunk_grid"],
            chunk_key_encoding=meta_json["chunk_key_encoding"],
            fill_value=array.fill_value,
            codecs=meta_json["codecs"],
            storage_transformers=meta_json["storage_transformers"],
            dimension_names=meta_json.get("dimension_names", None),
        )

    def to_zarr(
        self, store: Store, path: str, *, overwrite: bool = False, config: ArrayConfig | None = None
    ) -> zarr.Array:
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
        # This sucks! This should be easier!
        from zarr.core.array import Array, AsyncArray
        from zarr.core.metadata.v3 import ArrayV3Metadata
        from zarr.core.sync import sync
        from zarr.storage._common import ensure_no_existing_node, make_store_path

        store_path = sync(make_store_path(store, path=path))
        if overwrite:
            if store_path.store.supports_deletes:
                sync(store_path.delete_dir())
            else:
                sync(ensure_no_existing_node(store_path, zarr_format=3))
        else:
            sync(ensure_no_existing_node(store_path, zarr_format=3))

        meta: ArrayV3Metadata = ArrayV3Metadata.from_dict(self.model_dump())
        async_array = AsyncArray(metadata=meta, store_path=store_path, config=config)
        sync(async_array._save_metadata(meta))
        return Array(_async_array=async_array)


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
    attributes: TAttr = cast("TAttr", {})  # type: ignore[assignment]
    members: Mapping[str, TItem] = cast("Mapping[str, TItem]", {})

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


def from_zarr(
    element: zarr.Array | zarr.Group,
) -> ArraySpec[TBaseAttr] | GroupSpec[TBaseAttr, TBaseMember]:
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
    spec: ArraySpec[TBaseAttr],
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: GroupSpec[TBaseAttr, TBaseMember],
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Group: ...


def to_zarr(
    spec: ArraySpec[TBaseAttr] | GroupSpec[TBaseAttr, TBaseMember],
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
    return spec.to_zarr(store, path, overwrite=overwrite)


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
