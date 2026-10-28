from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Self,
    Union,
    Unpack,
    cast,
    get_args,
    overload,
    override,
)

import numpy as np
import numpy.typing as npt
from packaging.version import Version
from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator
from typing_extensions import TypedDict, TypeIs, TypeVar
from zarr_metadata import JSONValue, ZarrV2ArrayDimensionSeparator, ZarrV2ArrayOrder

from pydantic_zarr.core import (
    ExtraFieldsBase,
    IncEx,
    StrictBase,
    ensure_key_no_path,
    guess_chunks,
    maybe_node,
    model_like,
    parse_dtype_v2,
    resolve_member_path,
    walk_zarr_hierarchy,
)

if TYPE_CHECKING:
    import zarr
    from numcodecs.abc import Codec
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams

# SerializeAsAny: see the note on the v3 TBaseAttr — a model held by a spec
# with un-narrowed attributes must serialize with its own fields, not the
# empty base schema.
type TBaseAttr = Mapping[str, object] | SerializeAsAny[BaseModel]
# TBaseItem's bare, recursive spelling serves as TItem's bound and default;
# see the notes on the v3 spellings.
type TBaseItem = Union["GroupSpec", "ArraySpec"]

# Aliases for specs of any specialization: ordinary wide types, equal to the
# bare class names, since the covariant parameters make a narrowed spec
# substitutable for a wider one — see the note on the v3 aliases.
type AnyArraySpec = "ArraySpec[TBaseAttr]"
type AnyGroupSpec = "GroupSpec[TBaseAttr, TBaseItem]"
type AnyBaseGroupSpec = "BaseGroupSpec[TBaseAttr]"

# covariant because the specs are frozen; see the note on the v3 TypeVars
TAttr = TypeVar("TAttr", bound=TBaseAttr, default=TBaseAttr, covariant=True)
TItem = TypeVar("TItem", bound=TBaseItem, default=TBaseItem, covariant=True)
TDefault = TypeVar("TDefault")

DtypeStr = Annotated[str, BeforeValidator(parse_dtype_v2)]

BoolFillValue = bool
IntFillValue = int
# todo: introduce a type that represents hexadecimal representations of floats
FloatFillValue = Literal["Infinity", "-Infinity", "NaN"] | float
ComplexFillValue = tuple[FloatFillValue, FloatFillValue]
RawFillValue = tuple[int, ...]

FillValue = (
    BoolFillValue | IntFillValue | FloatFillValue | ComplexFillValue | RawFillValue | str | None
)

DimensionSeparator = ZarrV2ArrayDimensionSeparator
MemoryOrder = ZarrV2ArrayOrder


def dictify_codec(value: dict[str, Any] | Codec) -> dict[str, Any]:
    """
    Ensure that a `numcodecs.abc.Codec` is converted to a `dict`. If the input is not an
    instance of `numcodecs.abc.Codec`, then it is assumed to be a `dict` with string keys
    and it is returned unaltered.

    Parameters
    ----------

    value : dict[str, Any] | numcodecs.abc.Codec
        The value to be dictified if it is not already a dict.

    Returns
    -------
    dict[str, Any]
        If the input was a `Codec`, then the result of calling `get_config()` on that
        object is returned. This should be a dict with string keys. All other values pass
        through unaltered.
    """

    if (numcodecs := sys.modules.get("numcodecs")) and isinstance(value, numcodecs.abc.Codec):
        return value.get_config()
    # a Codec instance cannot exist unless numcodecs is imported, so value is a dict here
    return cast("dict[str, Any]", value)


def parse_dimension_separator(data: Any) -> DimensionSeparator:
    """
    Parse the dimension_separator metadata as per the Zarr version 2 specification.
    If the input is `None`, this returns ".".
    If the input is either "." or "/", this returns it.
    Otherwise, raises a ValueError.

    Parameters
    ----------
    data : Any
        The input data to parse.

    Returns
    -------
    Literal["/", "."]
    """
    if data is None:
        return "."
    if data in get_args(DimensionSeparator):
        return cast("DimensionSeparator", data)
    raise ValueError(f'Invalid data, expected one of ("/", ".", None), got {data}')


CodecDict = Annotated[dict[str, Any], BeforeValidator(dictify_codec)]


class NodeSpec(StrictBase):
    """
    The base class for V2 `ArraySpec` and `GroupSpec`.

    Attributes
    ----------

    zarr_format: Literal[2]
        The Zarr version represented by this node. Must be 2.
    """

    zarr_format: Literal[2] = 2


class ArraySpecUpdate(TypedDict, total=False):
    """The fields an `ArraySpec.update` call may replace."""

    attributes: TBaseAttr
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: npt.DTypeLike | list[tuple[Any, ...]]
    fill_value: FillValue
    order: MemoryOrder
    filters: Sequence[Mapping[str, Any]] | None
    dimension_separator: DimensionSeparator
    compressor: Mapping[str, Any] | None
    extra_fields: Mapping[str, JSONValue]


class BaseGroupSpecUpdate(TypedDict, total=False):
    """The fields a `BaseGroupSpec.update` call may replace."""

    attributes: TBaseAttr


class GroupSpecUpdate(TypedDict, total=False):
    """The fields a `GroupSpec.update` call may replace."""

    attributes: TBaseAttr
    members: Mapping[str, TBaseItem]


class ArraySpec(NodeSpec, ExtraFieldsBase, Generic[TAttr]):
    """
    A model of a Zarr Version 2 Array. The specification for the data structure being modeled by
    this class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#arrays).

    Attributes
    ----------
    attributes: TAttr, default = {}
        User-defined metadata associated with this array. Should be
        JSON-serializable. The default is the empty dict, matching a stored
        array with no attributes document; the default is validated against
        `TAttr`, so a model whose attributes type does not accept the empty
        dict still requires them.
    shape: tuple[int, ...]
        The shape of this array.
    dtype: str
        The data type of this array.
    chunks: Tuple[int, ...]
        The chunk size for this array.
    order: "C" | "F", default = "C"
        The memory order of this array. Must be either "C", which designates "C order",
        AKA lexicographic ordering or "F", which designates "F order", AKA colexicographic ordering.
        The default is "C".
    fill_value: FillValue, default = 0
        The fill value for this array. The default is 0.
    compressor: CodecDict | None
        A JSON-serializable representation of a compression codec, or None. The default is None.
    filters: List[CodecDict] | None, default = None
        A list of JSON-serializable representations of compression codec, or None.
        The default is None.
    dimension_separator: "." | "/", default = "/"
        The character used for partitioning the different dimensions of a chunk key.
        Must be either "/" or ".", or absent, in which case it is interpreted as ".".
        The default is "/".
    extra_fields: Mapping[str, JSONValue], default = {}
        The members of the array metadata document that the Zarr V2 spec
        does not define. The spec says such keys "SHOULD NOT be present
        within the metadata object and SHOULD be ignored by implementations",
        so any key with a JSON value is accepted, given no meaning, and kept
        for `model_dump`, `model_dump_json`, `update` and `like`.
        zarr-python drops them when it reads or writes an array, so
        `from_zarr` never finds any and `to_zarr` does not write them. Group
        metadata, where "Other keys MUST NOT be present", takes none.
    """

    attributes: TAttr = Field(default={}, validate_default=True)
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr | list[tuple[Any, ...]]
    fill_value: FillValue = 0
    order: MemoryOrder = "C"
    filters: list[CodecDict] | None = None
    dimension_separator: Annotated[
        DimensionSeparator, BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None

    @field_validator("filters", mode="after")
    @classmethod
    def validate_filters(cls, value: list[CodecDict] | None) -> list[CodecDict] | None:
        # Make sure filters is never an empty list
        if value == []:
            return None
        return value

    @model_validator(mode="after")
    def check_ndim(self) -> Self:
        """
        Check that the `shape` and `chunks` and attributes have the same length.
        """
        if (lshape := len(self.shape)) != (lchunks := len(self.chunks)):
            msg = (
                f"Length of shape must match length of chunks. Got {lshape} elements "
                f"for shape and {lchunks} elements for chunks."
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[Any] | zarr.Array,
        chunks: Literal["auto"] | tuple[int, ...] = "auto",
        # TBaseAttr, not TAttr: covariant parameters may not appear in
        # argument positions. Validation enforces the class's attribute type.
        attributes: Literal["auto"] | TBaseAttr = "auto",
        fill_value: Literal["auto"] | float | None = "auto",
        order: Literal["auto"] | MemoryOrder = "auto",
        filters: Literal["auto"] | list[CodecDict] | None = "auto",
        dimension_separator: Literal["auto"] | DimensionSeparator = "auto",
        compressor: Literal["auto"] | CodecDict | None = "auto",
    ) -> Self:
        """
        Create an `ArraySpec` from an array-like object. This is a convenience method for when Zarr array will be modelled from an existing array.
        This method takes nearly the same arguments as the `ArraySpec` constructor, minus `shape` and `dtype`, which will be inferred from the `array` argument.
        Additionally, this method accepts the string "auto" as a parameter for all other `ArraySpec` attributes, in which case these attributes will be
        inferred from the `array` argument, with a fallback value equal to the default `ArraySpec` parameters.

        Parameters
        ----------
        array : an array-like object
            Must have `shape` and `dtype` attributes.
            The `shape` and `dtype` of this object will be used to construct an `ArraySpec`.
        chunks : "auto" | tuple[int, ...], default = "auto"
            The chunks for this `ArraySpec`. If `chunks` is "auto" (the default), then this method first checks if `array` has a `chunksize` attribute, using it if present.
            This supports copying chunk sizes from dask arrays. If `array` does not have `chunksize`, then a routine from `zarr-python` is used to guess the chunk size,
            given the `shape` and `dtype` of `array`. If `chunks` is not auto, then it should be a tuple of ints.
        attributes : "auto" | TBaseAttr, default = "auto"
            User-defined metadata associated with this array. Should be JSON-serializable. The default is "auto", which means that `array.attributes` will be used,
            with a fallback value of the empty dict `{}`.
        fill_value : "auto" | int | float | None, default = "auto"
            The fill value for this array. Either "auto" or FillValue. The default is "auto", which means that `array.fill_value` will be used if that attribute exists, with a fallback value of 0.
        order : "auto" | "C" | "F", default = "auto"
            The memory order of the `ArraySpec`. One of "auto", "C", or "F". The default is "auto", which means that, if present, `array.order`
            will be used, falling back to "C" if `array` does not have an `order` attribute.
        filters : "auto" | List[CodecDict] | None, default = "auto"
            The filters for this `ArraySpec`. One of "auto", a list of JSON-serializable representations of compression codec, or `None`. The default is "auto", which means that the `array.filters` attribute will be
            used, with a fallback value of `None`.
        dimension_separator : "auto" | "." | "/", default = "auto"
            Sets the character used for partitioning the different dimensions of a chunk key.
            Must be one of "auto", "/" or ".". The default is "auto", which means that `array.dimension_separator` is used, with a fallback value of "/".
        compressor : "auto" | CodecDict | None, default = "auto"
            The compressor for this `ArraySpec`. One of "auto", a JSON-serializable representation of a compression codec, or `None`. The default is "auto", which means that `array.compressor` attribute will be used, with a fallback value of `None`.

        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with `shape` and `dtype` attributes derived from `array`.

        Examples
        --------
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> import numpy as np
        >>> x = ArraySpec.from_array(np.arange(10))
        >>> x
        ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)


        """
        attributes_actual: BaseModel | Mapping[str, object]
        shape_actual = array.shape
        dtype_actual = array.dtype

        if chunks == "auto":
            chunks_actual = auto_chunks(array)
        else:
            chunks_actual = chunks

        if attributes == "auto":
            attributes_actual = auto_attributes(array)
        else:
            attributes_actual = attributes

        if fill_value == "auto":
            fill_value_actual = auto_fill_value(array)
        else:
            fill_value_actual = fill_value

        if compressor == "auto":
            compressor_actual = auto_compresser(array)
        else:
            compressor_actual = compressor

        if filters == "auto":
            filters_actual = auto_filters(array)
        else:
            filters_actual = filters

        if order == "auto":
            order_actual = auto_order(array)
        else:
            order_actual = order

        if dimension_separator == "auto":
            dimension_separator_actual = auto_dimension_separator(array)
        else:
            dimension_separator_actual = dimension_separator

        return cls(
            shape=shape_actual,
            dtype=parse_dtype_v2(dtype_actual),
            chunks=chunks_actual,
            attributes=attributes_actual,
            fill_value=fill_value_actual,
            order=order_actual,
            compressor=None if compressor_actual is None else dictify_codec(compressor_actual),
            filters=None if filters_actual is None else [dictify_codec(f) for f in filters_actual],
            dimension_separator=dimension_separator_actual,
        )

    @classmethod
    def from_zarr(cls, array: zarr.Array) -> Self:
        """
        Create an `ArraySpec` from an instance of `zarr.Array`.

        Parameters
        ----------
        array : zarr.Array

        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with properties derived from `array`.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> ArraySpec.from_zarr(x)
        ArraySpec(zarr_format=2, attributes={}, shape=(10, 10), chunks=(10, 10), dtype='<f8', fill_value=0.0, order='C', filters=None, dimension_separator='.', compressor={'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0})

        """
        try:
            from zarr.core.metadata import ArrayV2Metadata
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        if not isinstance(array.metadata, ArrayV2Metadata):
            msg = "Array is not a Zarr format 2 array"
            raise TypeError(msg)

        if Version(version("zarr")) < Version("3.1.0"):
            from zarr.core.buffer import default_buffer_prototype

            stored_meta = array.metadata.to_buffer_dict(prototype=default_buffer_prototype())
            meta_json = json.loads(stored_meta[".zarray"].to_bytes()) | {
                "attributes": array.attrs.asdict()
            }
        else:
            meta_json = array.metadata.to_dict()

        return cls.model_validate(meta_json)

    def to_zarr(
        self,
        store: Store,
        path: str,
        *,
        overwrite: bool = False,
        config: ArrayConfigParams | None = None,
    ) -> zarr.Array:
        """
        Serialize an `ArraySpec` to a Zarr array at a specific path in a Zarr store. This operation
        will create metadata documents in the store, but will not write any chunks.

        Parameters
        ----------
        store : instance of zarr.abc.store.Store
            The storage backend that will manifest the array.
        path : str
            The location of the array inside the store.
        overwrite : bool, default = False
            Whether to overwrite existing objects in storage to create the Zarr array.
        config : ArrayConfigParams | None, default = None
            An instance of `ArrayConfigParams` that defines the runtime configuration for the array.

        Returns
        -------
        zarr.Array
            A Zarr array that is structurally identical to `self`.
        """
        try:
            import zarr
            from zarr.core.array import Array, AsyncArray
            from zarr.core.metadata import ArrayV2Metadata
            from zarr.core.sync import sync
            from zarr.errors import ContainsArrayError, ContainsGroupError
            from zarr.storage._common import make_store_path
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        store_path = sync(make_store_path(store, path=path))

        extant_node = maybe_node(store, path, zarr_format=2)
        if isinstance(extant_node, zarr.Array):
            # zarr-python neither writes nor reads extra fields, so the stored
            # array is this spec as written when it matches in everything else
            if not self.like(extant_node, exclude={"extra_fields"}) and not overwrite:
                raise ContainsArrayError(store, path)
            else:
                # If there's an existing array that is identical to the model, and overwrite is False,
                # we can just return that existing array.
                if not overwrite:
                    return extant_node
        if isinstance(extant_node, zarr.Group) and not overwrite:
            raise ContainsGroupError(store, path)

        meta: ArrayV2Metadata = ArrayV2Metadata.from_dict(self.model_dump())
        async_array = AsyncArray(metadata=meta, store_path=store_path, config=config)
        sync(async_array._save_metadata(meta))
        return Array(_async_array=async_array)

    def update(self, **changes: Unpack[ArraySpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so correlated fields (for an
        array, a new `data_type` with a matching `fill_value`, or a new
        `shape` with a matching `chunk_grid`) change together atomically.
        Unlike `model_copy(update=...)`, the result is validated. Passing
        `extra_fields` replaces the extra fields as a whole.
        """
        return self._updated(changes)

    def like(
        self,
        other: ArraySpec | zarr.Array,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare an `ArraySpec` to another `ArraySpec` or a `zarr.Array`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to `dict` via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other : ArraySpec | zarr.Array
            The array (model or actual) to compare with. If other is a `zarr.Array`, it will be
            converted to `ArraySpec` first.
        include : IncEx, default = None
            A specification of fields to include in the comparison. The default value is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude : IncEx, default = None
            A specification of fields to exclude from the comparison. The default value is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise, given
            the set of fields specified by the `include` and `exclude` keyword arguments.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> x.attrs.put({'foo': 10})
        >>> x_model = ArraySpec.from_zarr(x)
        >>> print(x_model.like(x_model)) # it is like itself.
        True
        >>> print(x_model.like(x))
        True
        >>> y = zarr.create((10,10))
        >>> y.attrs.put({'foo': 11}) # x and y are the same, other than their attrs
        >>> print(x_model.like(y))
        False
        >>> print(x_model.like(y, exclude={'attributes'}))
        True
        """

        other_parsed: ArraySpec
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Array):
            other_parsed = ArraySpec.from_zarr(other)
        else:
            other_parsed = other  # type: ignore[assignment]

        return model_like(self, other_parsed, include=include, exclude=exclude)


class BaseGroupSpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 2 Group's own metadata, without members.

    This is the shape of a group in a flattened hierarchy (`to_flat` emits it
    for every group), and of a group loaded without traversing its children.
    `GroupSpec` subclasses this model and adds `members`.

    Attributes
    ----------
    attributes: TAttr, default = {}
        The user-defined attributes of this group. The default is the empty
        dict, matching a stored group with no attributes document; the default
        is validated against `TAttr`, so a model whose attributes type does
        not accept the empty dict still requires them.
    """

    attributes: TAttr = Field(default={}, validate_default=True)

    def update(self, **changes: Unpack[BaseGroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        Unlike `model_copy(update=...)`, the result is validated, so an
        invalid change is refused.
        """
        return type(self).model_validate({**self.model_dump(), **changes})

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> Self:
        """
        Create a `BaseGroupSpec` from a zarr group, without reading its members.
        """
        return cls(attributes=group.attrs.asdict())

    def to_zarr(self, store: Store, path: str, *, overwrite: bool = False) -> zarr.Group:
        """
        Serialize this group's own metadata to a zarr group at a specific path
        in a zarr store. No members are written; `GroupSpec.to_zarr` writes a
        whole hierarchy.
        """
        try:
            import zarr
            from zarr.errors import ContainsArrayError, ContainsGroupError
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        extant_node = maybe_node(store, path, zarr_format=2)
        if isinstance(extant_node, zarr.Group):
            if not self.like(extant_node):
                if not overwrite:
                    raise ContainsGroupError(store, path)
            elif not overwrite:
                return extant_node
        elif isinstance(extant_node, zarr.Array) and not overwrite:
            raise ContainsArrayError(store, path)
        else:
            zarr.create_group(store=store, overwrite=overwrite, path=path, zarr_format=2)

        result = zarr.group(store=store, path=path, overwrite=overwrite, zarr_format=2)
        result.attrs.put(self.model_dump()["attributes"])
        return result

    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec | zarr.Group,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare this group's own metadata to another group's. Members never
        participate in the comparison: a `GroupSpec` or `zarr.Group` is reduced
        to its own metadata first.
        """
        other_parsed: AnyBaseGroupSpec
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Group):
            other_parsed = BaseGroupSpec(attributes=other.attrs.asdict())
        elif isinstance(other, GroupSpec):
            other_parsed = BaseGroupSpec(attributes=other.attributes)
        else:
            other_parsed = cast("AnyBaseGroupSpec", other)

        return model_like(self, other_parsed, include=include, exclude=exclude)


class GroupSpec(BaseGroupSpec[TAttr], Generic[TAttr, TItem]):
    """
    A model of a Zarr Version 2 Group.
    The specification for the data structure being modeled by this
    class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#groups).

    Attributes
    ----------

    attributes: TAttr
        The user-defined attributes of this group. Should be JSON-serializable.
    members: dict[str, TItem], default = {}
        The members of this group: a `Mapping` with string keys and values that are
        `ArraySpec` or `GroupSpec`. A `GroupSpec` is a complete hierarchy, so an empty
        mapping means the group has no members. A group whose members are unknown is a
        `BaseGroupSpec`, which has no `members` field and is never a member.
    """

    members: Mapping[str, TItem] = {}

    @model_validator(mode="after")
    def validate_member_names(self) -> Self:
        """
        Validate member names (no ``"/"``, not empty, not ``"."``/``".."``).

        This runs as a model validator rather than field metadata so that a
        subclass narrowing the `members` annotation (for example to a closed
        `TypedDict` of members) cannot drop it.
        """
        ensure_key_no_path(self.members)
        return self

    @override
    def update(self, **changes: Unpack[GroupSpecUpdate]) -> Self:  # type: ignore[override]
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so `attributes` and `members`
        change together atomically. Unlike `model_copy(update=...)`, the
        result is validated.
        """
        return type(self).model_validate({**self.model_dump(), **changes})

    def resolve_path(self, path: str) -> AnyArraySpec | AnyGroupSpec:
        """
        Return the node at a `/`-separated path of member names relative to
        this group (`"a"`, `"a/b/c"`; leading and trailing `/` are ignored).
        Metadata conventions that reference nodes by relative path resolve
        with this method.

        Raises
        ------
        ValueError
            If the path holds no member names, or an empty one (`"a//b"`).
        KeyError
            Naming the segment that failed: a member name that does not
            exist, or a segment that descends into a node that is not a
            group.
        """
        node = resolve_member_path(self, path)
        return cast("AnyArraySpec | AnyGroupSpec", node)

    def __getitem__(self, path: str) -> AnyArraySpec | AnyGroupSpec:
        """`group[path]` resolves a member path: see `resolve_path`."""
        return self.resolve_path(path)

    def get_path(
        self, path: str, default: TDefault | None = None
    ) -> AnyArraySpec | AnyGroupSpec | TDefault | None:
        """
        `resolve_path` that returns `default` instead of raising when the path
        does not resolve (a missing member, or a non-group on the way); a
        malformed path still raises `ValueError`. This non-raising
        form suits pydantic validators, which turn `ValueError` into a
        validation error but let `KeyError` escape.
        """
        try:
            return self.resolve_path(path)
        except KeyError:
            return default

    @classmethod
    @override
    def from_zarr(cls, group: zarr.Group) -> Self:
        """
        Create a GroupSpec from an instance of `zarr.Group`. Subgroups and arrays contained in the
        Zarr group will be converted to instances of `GroupSpec` and `ArraySpec`, respectively,
        and these spec instances will be stored in the `members` attribute of the parent
        `GroupSpec`.

        This reads the entire hierarchy under `group`, which may be slow for large hierarchies
        on high-latency storage. To read part of a hierarchy, use `from_zarr_flat`; to read a
        group without its members, use `BaseGroupSpec.from_zarr`.

        Parameters
        ----------
        group : zarr.Group
            The Zarr group to model.

        Returns
        -------
        An instance of GroupSpec that represents the structure of the Zarr hierarchy.
        """
        try:
            import zarr
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        attributes = group.attrs.asdict()
        members = {}

        for name, item in group.members():
            if isinstance(item, zarr.Array):
                # convert to dict before the final typed GroupSpec construction
                item_out = ArraySpec.from_zarr(item).model_dump()
            elif isinstance(item, zarr.Group):
                # convert to dict before the final typed GroupSpec construction
                item_out = GroupSpec.from_zarr(item).model_dump()
            else:
                msg = (  # type: ignore[unreachable]
                    f"Unparsable object encountered: {type(item)}. Expected zarr.Array"
                    " or zarr.Group."
                )

                raise ValueError(msg)  # noqa: TRY004
            members[name] = item_out

        return cls(attributes=attributes, members=members)

    @override
    def to_zarr(
        self, store: Store, path: str, *, overwrite: bool = False, **kwargs: Any
    ) -> zarr.Group:
        """
        Serialize this `GroupSpec` to a Zarr group at a specific path in a `zarr.abc.store.Store`.
        This operation will create metadata documents in the store.

        Parameters
        ----------
        store : zarr.abc.store.Store
            The storage backend that will manifest the group and its contents.
        path : str
            The location of the group inside the store.
        overwrite : bool, default = False
            Whether to overwrite existing objects in storage to create the Zarr group.
        **kwargs : Any
            Additional keyword arguments that will be passed to `zarr.create` for creating
            sub-arrays.

        Returns
        -------
        zarr.Group
            A zarr group that is structurally identical to `self`.

        """
        try:
            import zarr
            from zarr.errors import ContainsArrayError, ContainsGroupError
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        spec_dict = self.model_dump(exclude={"members": True})
        attrs = spec_dict.pop("attributes")
        extant_node = maybe_node(store, path, zarr_format=2)
        if isinstance(extant_node, zarr.Group):
            if not self.like(extant_node):
                if not overwrite:
                    """
                    msg = (
                        f"A group already exists at path {path}. "
                        "That group is structurally dissimilar to the group you are trying to store."
                        "Call to_zarr with overwrite=True to overwrite that group."
                    )
                    """
                    # Zarr's contains group error uses questionable design and doesn't take a message
                    raise ContainsGroupError(store, path)
            else:
                if not overwrite:
                    # if the extant group is structurally identical to self, and overwrite is false,
                    # then just return the extant group
                    return extant_node

        elif isinstance(extant_node, zarr.Array) and not overwrite:
            """
            msg = (
                f"An array already exists at path {path}. "
                "Call to_zarr with overwrite=True to overwrite the array."
            )
            """
            # Zarr's contains array error uses questionable design and doesn't take a message
            raise ContainsArrayError(store, path)
        else:
            zarr.create_group(store=store, overwrite=overwrite, path=path, zarr_format=2)

        result = zarr.group(store=store, path=path, overwrite=overwrite, zarr_format=2)
        result.attrs.put(attrs)
        for name, member in self.members.items():
            subpath = f"{path.rstrip('/')}/{name.lstrip('/')}"
            member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

        return result

    @override
    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec | zarr.Group,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare a `GroupSpec` to another `GroupSpec`, `BaseGroupSpec` or `zarr.Group`,
        parameterized over the fields to exclude or include in the comparison. Models are first
        converted to dict via the `model_dump` method of `pydantic.BaseModel`, then compared with
        the `==` operator. A `BaseGroupSpec` — a group considered without members — compares as
        that projection: by the group's own metadata alone, whatever this side's members hold.

        Parameters
        ----------
        other : GroupSpec | BaseGroupSpec | zarr.Group
            The group (model or actual) to compare with. If other is a `zarr.Group`, it will be
            converted to a `GroupSpec`.
        include : IncEx, default = None
            A specification of fields to include in the comparison. The default is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude : IncEx, default = None
            A specification of fields to exclude from the comparison. The default is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import GroupSpec
        >>> import numpy as np
        >>> z1 = zarr.group(path='z1')
        >>> z1a = z1.array(name='foo', data=np.arange(10))
        >>> z1_model = GroupSpec.from_zarr(z1)
        >>> print(z1_model.like(z1_model)) # it is like itself
        True
        >>> print(z1_model.like(z1))
        True
        >>> z2 = zarr.group(path='z2')
        >>> z2a = z2.array(name='foo', data=np.arange(10))
        >>> print(z1_model.like(z2))
        True
        >>> z2.attrs.put({'foo' : 100}) # now they have different attributes
        >>> print(z1_model.like(z2))
        False
        >>> print(z1_model.like(z2, exclude={'attributes'}))
        True
        """

        if isinstance(other, BaseGroupSpec) and not isinstance(other, GroupSpec):
            # a group without members compares by the group's own metadata;
            # this side projects to its own metadata too, or the comparison
            # would always fail on the members field itself
            projection = BaseGroupSpec(attributes=self.attributes)
            return projection.like(other, include=include, exclude=exclude)
        other_parsed: GroupSpec
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Group):
            other_parsed = GroupSpec.from_zarr(other)
        else:
            other_parsed = other  # type: ignore[assignment]

        return model_like(self, other_parsed, include=include, exclude=exclude)

    def to_flat(self, root_path: str = "") -> dict[str, AnyArraySpec | AnyBaseGroupSpec]:
        """
        Flatten this `GroupSpec`.
        This method returns a `dict` with string keys and values that are `GroupSpec` or
        `ArraySpec`.

        Then the resulting `dict` will contain the input's own metadata, as a `BaseGroupSpec`,
        under the key `root_path`, as well as copies of the result of calling `node.to_flat` on each
        element of `node.members`, each under a key created by joining `root_path` with a '/`
        character to the name of each member, and so on recursively for each sub-member.

        Parameters
        ----------
        root_path : `str`, default = ''
            The root path. The keys in `self.members` will be
            made relative to `root_path` when used as keys in the result dictionary.

        Returns
        -------
        Dict[str, ArraySpec | GroupSpec]
            A flattened representation of the hierarchy.

        Examples
        --------

        >>> from pydantic_zarr.v2 import GroupSpec
        >>> g1 = GroupSpec(attributes={'foo': 'bar'})
        >>> g1.to_flat()
        {'': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
        >>> g1.to_flat(root_path='baz')
        {'baz': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
        >>> GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}).to_flat()
        {'': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'}), '/g1': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
        """
        return to_flat(self, root_path=root_path)

    @classmethod
    def from_flat(cls, data: Mapping[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation. The flattened hierarchy is a
        `dict` with the following constraints: keys must be valid paths; values must
        be `ArraySpec` or `BaseGroupSpec` instances, as `to_flat` emits them. A `GroupSpec`
        value is read as its own metadata: its members are ignored, since a group's members
        are the entries under its path.

        Parameters
        ----------
        data : Dict[str, ArraySpec | BaseGroupSpec]
            A flattened representation of a Zarr hierarchy.

        Returns
        -------
        GroupSpec
            A `GroupSpec` representation of the hierarchy.

        Examples
        --------
        >>> from pydantic_zarr.v2 import BaseGroupSpec, GroupSpec, ArraySpec
        >>> import numpy as np
        >>> flat = {'': BaseGroupSpec(attributes={'foo': 10})}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={})
        >>> flat = {
        ...     '': BaseGroupSpec(attributes={'foo': 10}),
        ...     '/a': ArraySpec.from_array(np.arange(10))}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={'a': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
        """
        from_flated = from_flat_group(data)
        return cls.model_validate(from_flated.model_dump())


@overload
def from_zarr(element: zarr.Array) -> AnyArraySpec: ...


@overload
def from_zarr(element: zarr.Group) -> AnyGroupSpec: ...


def from_zarr(element: zarr.Array | zarr.Group) -> AnyArraySpec | AnyGroupSpec:
    """
    Recursively parse a `zarr.Group` or `zarr.Array` into an `ArraySpec` or `GroupSpec`.
    A group is read with its entire hierarchy; to read part of one, use `from_zarr_flat`.

    Parameters
    ----------
    element : zarr.Array | zarr.Group
        The `zarr.Array` or `zarr.Group` to model.

    Returns
    -------
    ArraySpec | GroupSpec
        An instance of `GroupSpec` or `ArraySpec` that models the structure of the input Zarr group
        or array.
    """
    import zarr

    if isinstance(element, zarr.Array):
        return ArraySpec.from_zarr(element)
    else:
        return GroupSpec.from_zarr(element)


def from_zarr_flat(
    element: zarr.Array | zarr.Group, *, depth: int = -1
) -> dict[str, AnyArraySpec | AnyBaseGroupSpec]:
    """
    Read a zarr array or group, and the members under it up to `depth` levels deep, into a
    flat hierarchy: a `dict` from paths to `ArraySpec` and `BaseGroupSpec` values, in the form
    `to_flat` produces and `from_flat` takes.

    This is how to read part of a hierarchy. A `GroupSpec` is always a complete hierarchy, so
    `GroupSpec.from_zarr` reads every member; a flat hierarchy makes no claim that it lists
    every node, so it can stop at any depth. `from_flat` turns a complete one into a
    `GroupSpec`. It reads its input as the complete hierarchy, so passing it a result read
    with a `depth` limit yields a `GroupSpec` whose groups at that limit have no members.

    Parameters
    ----------
    element : zarr.Array | zarr.Group
        The zarr array or group to read. Its path in the result is `""`.
    depth : int, default = -1
        How many levels of members to read: -1 reads the whole hierarchy, 0 reads `element`
        alone, 1 its members, and so on. This has no effect if `element` is an array.

    Returns
    -------
    dict[str, ArraySpec | BaseGroupSpec]
        The nodes read, keyed by path relative to `element`.

    Raises
    ------
    ValueError
        If `depth` is lower than -1.
    """
    import zarr

    return {
        path: ArraySpec.from_zarr(node)
        if isinstance(node, zarr.Array)
        else BaseGroupSpec.from_zarr(node)
        for path, node in walk_zarr_hierarchy(element, depth=depth)
    }


@overload
def to_zarr(
    spec: AnyArraySpec,
    store: Store,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: AnyGroupSpec,
    store: Store,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> zarr.Group: ...


def to_zarr(
    spec: AnyArraySpec | AnyGroupSpec,
    store: Store,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> zarr.Array | zarr.Group:
    """
    Serialize a `GroupSpec` or `ArraySpec` to a Zarr group or array at a specific path in
    a Zarr store.

    Parameters
    ----------
    spec : ArraySpec | GroupSpec
        The `GroupSpec` or `ArraySpec` that will be serialized to storage.
    store : zarr.abc.store.BaseStore
        The storage backend that will manifest the Zarr group or array modeled by `spec`.
    path : str
        The location of the Zarr group or array inside the store.
    overwrite : bool, default = False
        Whether to overwrite existing objects in storage to create the Zarr group or array.
    **kwargs
        Additional keyword arguments will be passed to member `to_zarr` calls and `zarr.create`.

    Returns
    -------
    zarr.Array | zarr.Group
        A `zarr.Group` or `zarr.Array` that is structurally equivalent to `spec`.
        This operation will create metadata documents in the store.

    """
    return spec.to_zarr(store, path, overwrite=overwrite, **kwargs)


def to_flat(
    node: AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec, root_path: str = ""
) -> dict[str, AnyArraySpec | AnyBaseGroupSpec]:
    """
    Flatten a `GroupSpec` or `ArraySpec`.

    Converts a `GroupSpec` or `ArraySpec` and a string, into a `dict` with string keys and
    values that are `GroupSpec` or `ArraySpec`.

    If the input is an `ArraySpec`, then this function just returns the dict `{root_path: node}`.

    If the input is a `GroupSpec`, then the resulting `dict` will contain the input's own
    metadata, as a `BaseGroupSpec`, under the key `root_path`, as well as copies of the result of
    calling `flatten_node` on each element of `node.members`, each under a key created by joining
    `root_path` with a '/` character to the name of each member, and so on recursively for each
    sub-member.

    Parameters
    ----------
    node : `GroupSpec` | `ArraySpec`
        The node to flatten.
    root_path : `str`, default = ''
        The root path. If `node` is a `GroupSpec`, then the keys in `node.members` will be
        made relative to `root_path` when used as keys in the result dictionary.

    Returns
    -------
    Dict[str, ArraySpec | GroupSpec]
        A flattened representation of the hierarchy.

    Examples
    --------

    >>> from pydantic_zarr.v2 import to_flat, GroupSpec
    >>> g1 = GroupSpec(attributes={'foo': 'bar'})
    >>> to_flat(g1)
    {'': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
    >>> to_flat(g1, root_path='baz')
    {'baz': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
    >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
    {'': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'}), '/g1': BaseGroupSpec(zarr_format=2, attributes={'foo': 'bar'})}
    """
    result = {}
    model_copy: AnyArraySpec | AnyBaseGroupSpec
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        model_copy = BaseGroupSpec(attributes=node.model_copy(deep=True).attributes)
        if isinstance(node, GroupSpec):
            for name, value in node.members.items():
                result.update(to_flat(value, f"{root_path}/{name}"))

    result[root_path] = model_copy
    # sort by increasing key length
    return dict(sorted(result.items(), key=lambda v: len(v[0])))


def from_flat(
    data: dict[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec],
) -> AnyArraySpec | AnyGroupSpec:
    """
    Wraps `from_flat_group`, handling the special case where a Zarr array is defined at the root of
    a hierarchy and thus is not contained by a Zarr group.

    Parameters
    ----------

    data : Dict[str, ArraySpec | BaseGroupSpec]
        A flat representation of a Zarr hierarchy. This is a `dict` with keys that are strings,
        and values that are either `BaseGroupSpec` or `ArraySpec` instances. A `GroupSpec`
        value is read as its own metadata: its members are ignored.

    Returns
    -------
    ArraySpec | GroupSpec
        The `ArraySpec` or `GroupSpec` representation of the input data.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # special case of a Zarr array at the root of the hierarchy
    ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # note that an implicit Group is created
    GroupSpec(zarr_format=2, attributes={}, members={'foo': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """

    # minimal check that the keys are valid
    invalid_keys = [key for key in data if key.endswith("/")]
    if len(invalid_keys) > 0:
        msg = f'Invalid keys {invalid_keys} found in data. Keys may not end with the "/"" character'
        raise ValueError(msg)

    if tuple(data.keys()) == ("",) and isinstance(first := next(iter(data.values())), ArraySpec):
        return first
    else:
        return from_flat_group(data)


def from_flat_group(
    data: Mapping[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec],
) -> AnyGroupSpec:
    """
    Generate a `GroupSpec` from a flat representation of a hierarchy, i.e. a `dict` with
    string keys (paths) and `ArraySpec` / `GroupSpec` values (nodes).

    Parameters
    ----------
    data : Dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy rooted at a Zarr group.

    Returns
    -------
    GroupSpec
        A `GroupSpec` that represents the hierarchy described by `data`.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat_group, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat_group(tree) # note that an implicit Group is created
    GroupSpec(zarr_format=2, attributes={}, members={'foo': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """
    root_name = ""
    sep = "/"
    # arrays that will be members of the returned GroupSpec
    member_arrays: dict[str, AnyArraySpec] = {}
    # groups, and their members, that will be members of the returned GroupSpec.
    # this dict is populated by recursively applying `from_flat_group` function.
    member_groups: dict[str, AnyGroupSpec] = {}
    # this dict collects the arrayspecs and groupspecs that belong to one of the members of the
    # groupspecs we are constructing. They will later be aggregated in a recursive step that
    # populates member_groups
    submember_by_parent_name: dict[
        str, dict[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec]
    ] = {}
    # copy the input to ensure that mutations are contained inside this function
    data_copy = dict(data)
    # Get the root node
    try:
        # The root node is a GroupSpec with the key ""
        root_node = data_copy.pop(root_name)
        if isinstance(root_node, ArraySpec):
            raise ValueError("Got an ArraySpec as the root node. This is invalid.")  # noqa: TRY004
    except KeyError:
        # If a root node was not found, create a default one
        root_node = BaseGroupSpec(attributes={})

    # partition the tree (sans root node) into 2 categories: (arrays, groups + their members).
    for key, value in data_copy.items():
        key_parts = key.split(sep)
        if key_parts[0] != root_name:
            raise ValueError(f'Invalid path: {key} does not start with "{root_name}{sep}".')

        subparent_name = key_parts[1]
        if len(key_parts) == 2:
            # this is an array or group that belongs to the group we are ultimately returning
            if isinstance(value, ArraySpec):
                member_arrays[subparent_name] = value
            elif isinstance(value, BaseGroupSpec):
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                submember_by_parent_name[subparent_name][root_name] = value
            else:
                raise ValueError(
                    f"Value at '{key}' is not a v2 ArraySpec or GroupSpec (got {type(value)=})"
                )
        else:
            # these are groups or arrays that belong to one of the member groups
            # not great that we repeat this conditional dict initialization
            if subparent_name not in submember_by_parent_name:
                submember_by_parent_name[subparent_name] = {}
            submember_by_parent_name[subparent_name][sep.join([root_name, *key_parts[2:]])] = value

    # recurse
    for subparent_name, submemb in submember_by_parent_name.items():
        member_groups[subparent_name] = from_flat_group(submemb)

    return GroupSpec(members={**member_groups, **member_arrays}, attributes=root_node.attributes)


def auto_chunks(data: Any) -> tuple[int, ...]:
    """
    Guess chunks from:
      input with a `chunksize` attribute, or
      input with a `chunks` attribute, or,
      input with `shape` and `dtype` attributes
    """
    if hasattr(data, "chunksize"):
        return data.chunksize
    if hasattr(data, "chunks"):
        return data.chunks
    return guess_chunks(data.shape, np.dtype(data.dtype).itemsize)


def auto_attributes(data: Any) -> Mapping[str, Any]:
    """
    Guess attributes from:
        input with an `attrs` attribute, or
        input with an `attributes` attribute,
        or anything (returning {})
    """
    if hasattr(data, "attrs"):
        return data.attrs
    if hasattr(data, "attributes"):
        return data.attributes
    return {}


def auto_fill_value(data: Any) -> Any:
    """
    Guess fill value from an input with a `fill_value` attribute, returning 0 otherwise.
    """
    if hasattr(data, "fill_value"):
        return data.fill_value
    return 0


def auto_compresser(data: Any) -> Codec | None:
    """
    Guess compressor from an input with a `compressor` attribute, returning `None` otherwise.
    """
    if hasattr(data, "compressor"):
        return data.compressor
    return None


def auto_filters(data: Any) -> list[Codec] | None:
    """
    Guess filters from an input with a `filters` attribute, returning `None` otherwise.
    """
    if hasattr(data, "filters"):
        return data.filters
    return None


def auto_order(data: Any) -> MemoryOrder:
    """
    Guess array order from an input with an `order` attribute, returning "C" otherwise.
    """
    if hasattr(data, "order"):
        if data.order in get_args(MemoryOrder):
            return cast("MemoryOrder", data.order)
        else:
            raise ValueError(f"Order attribute not in {get_args(MemoryOrder)}")
    return "C"


def auto_dimension_separator(data: Any) -> DimensionSeparator:
    """
    Guess dimension separator from an input with a `dimension_separator` attribute, returning "/" otherwise.
    """
    if hasattr(data, "dimension_separator"):
        if data.dimension_separator in get_args(DimensionSeparator):
            return cast("DimensionSeparator", data.dimension_separator)
        else:
            raise ValueError(f"Dimension separator attribute not in {get_args(DimensionSeparator)}")
    return "/"


def is_array_spec(node: object) -> TypeIs[AnyArraySpec]:
    """Whether `node` is an `ArraySpec`, narrowing the type in both branches."""
    return isinstance(node, ArraySpec)


def is_group_spec(node: object) -> TypeIs[AnyGroupSpec]:
    """
    Whether `node` is a `GroupSpec` (a group with members), narrowing the type
    in both branches. A `BaseGroupSpec` that is not a `GroupSpec` returns
    `False`: use this to tell the two apart in a flattened hierarchy.
    """
    return isinstance(node, GroupSpec)


def is_base_group_spec(node: object) -> TypeIs[AnyBaseGroupSpec]:
    """
    Whether `node` models a group, with or without members: `GroupSpec` is a
    subclass of `BaseGroupSpec`, so this holds for both.
    """
    return isinstance(node, BaseGroupSpec)
