from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
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

from pydantic import BeforeValidator

from pydantic_zarr.core import StrictBase
from pydantic_zarr.v2 import stringify_dtype

if TYPE_CHECKING:
    import numpy.typing as npt
    import zarr
    from zarr.storage import BaseStore


TAttr = TypeVar("TAttr", bound=dict[str, Any])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])

NodeType = Literal["group", "array"]

# todo: introduce a type that represents hexadecimal representations of floats
FillValue = (
    Literal["Infinity", "-Infinity", "NaN"]  # noqa: PYI051
    | bool
    | int
    | float
    | str
    | tuple[float, float]
    | tuple[int, ...]
)


class NamedConfig(StrictBase):
    name: str
    configuration: Mapping[str, Any] | None


class RegularChunkingConfig(StrictBase):
    chunk_shape: list[int]


class RegularChunking(NamedConfig):
    name: Literal["regular"] = "regular"
    configuration: RegularChunkingConfig  # type: ignore[assignment]


class DefaultChunkKeyEncodingConfig(StrictBase):
    separator: Literal[".", "/"]


class DefaultChunkKeyEncoding(NamedConfig):
    name: Literal["default"]
    configuration: DefaultChunkKeyEncodingConfig | None  # type: ignore[assignment]


class NodeSpec(StrictBase):
    """
    The base class for V3 ArraySpec and GroupSpec.

    Attributes
    ----------

    zarr_format: Literal[3]
        The Zarr version represented by this node. Must be 3.
    """

    zarr_format: Literal[3] = 3


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
    data_type: Annotated[str, BeforeValidator(stringify_dtype)]
    chunk_grid: NamedConfig  # todo: validate this against shape
    chunk_key_encoding: NamedConfig  # todo: validate this against shape
    fill_value: FillValue  # todo: validate this against the data type
    codecs: tuple[NamedConfig, ...]
    storage_transformers: tuple[NamedConfig, ...] | None = None
    dimension_names: tuple[str, ...] | None  # todo: validate this against shape

    @classmethod
    def from_array(cls, array: npt.NDArray[Any], **kwargs: Any) -> Self:
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array :
            object that conforms to the numpy array API.
            The shape and dtype of this object will be used to construct an ArraySpec.
            If the `chunks` keyword argument is not given, the shape of the array will
            be used for the chunks.
        **kwargs :
            keyword arguments to the ArraySpec class constructor.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """
        default_chunks = RegularChunking(
            configuration=RegularChunkingConfig(chunk_shape=list(array.shape))
        )
        return cls(
            shape=array.shape,
            data_type=str(array.dtype),
            chunk_grid=kwargs.pop("chunks", default_chunks),
            attributes=kwargs.pop("attributes", {}),
            **kwargs,
        )

    @classmethod
    def from_zarr(cls, zarray: zarr.Array) -> Never:
        """
        Create an ArraySpec from a zarr array.

        Parameters
        ----------
        zarray : zarr array

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided zarr
        array.

        """
        raise NotImplementedError

    def to_zarr(self, store: BaseStore, path: str, overwrite: bool = False) -> zarr.Array:
        """
        Serialize an ArraySpec to a zarr array at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.BaseStore
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

    def to_zarr(self, store: BaseStore, path: str, overwrite: bool = False) -> Never:
        """
        Serialize a GroupSpec to a zarr group at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.BaseStore
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
    ---------
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
    store: BaseStore,
    path: str,
    overwrite: bool = False,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: BaseStore,
    path: str,
    overwrite: bool = False,
) -> zarr.Group: ...


def to_zarr(
    spec: ArraySpec | GroupSpec,
    store: BaseStore,
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
    store : instance of zarr.BaseStore
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
