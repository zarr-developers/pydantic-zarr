from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Generic,
    Literal,
    Self,
    Union,
    Unpack,
    cast,
    override,
)

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator
from typing_extensions import TypedDict, TypeIs, TypeVar
from zarr_metadata import (
    ZARR_V2_ARRAY_DEFAULT_DIMENSION_SEPARATOR,
    ZARR_V2_ARRAY_DIMENSION_SEPARATOR,
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZARR_V2_ARRAY_ORDER,
    ZARR_V2_ATTRIBUTES_STORE_KEY,
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    JSONValue,
    MetadataValidationError,
    ZarrV2ArrayDimensionSeparator,
    ZarrV2ArrayMetadataJSON,
    ZarrV2ArrayOrder,
    ZarrV2DataTypeMetadata,
    ZarrV2GroupMetadataJSON,
    ZarrV2ZArrayJSON,
    ZarrV2ZAttrsJSON,
    ZarrV2ZGroupJSON,
)
from zarr_metadata.model import ZarrV2ArrayMetadata as ZarrV2ArrayMetadataModel
from zarr_metadata.v2.array import (
    is_zarr_v2_array_dimension_separator,
    is_zarr_v2_array_order,
)
from zarr_metadata.v2.data_type.scalar import ZarrV2FloatSpecial

from pydantic_zarr._metadata import format_problems
from pydantic_zarr.core import (
    ExtraFieldsBase,
    IncEx,
    StoredNodes,
    StoredNodeType,
    StrictBase,
    as_dtype_like,
    as_shape,
    canonicalize_fill_value,
    copy_members,
    ensure_key_no_path,
    ensure_members_mapping,
    guess_chunks,
    model_like,
    parse_dtype_v2,
    read_attribute,
    resolve_member_path,
    split_store_key,
    stored_array,
    stored_flat,
    stored_root,
    tuplify_json,
)

if TYPE_CHECKING:
    import zarr
    from numcodecs.abc import Codec

# SerializeAsAny: see the note on the v3 TBaseAttr — a model held by a spec
# with un-narrowed attributes must serialize with its own fields, not the
# empty base schema.
type TBaseAttr = Mapping[str, object] | SerializeAsAny[BaseModel]
# TBaseMembers, the widest members mapping, serves as TMembers' bound and
# default; see the notes on the v3 spellings.
type TBaseItem = Union["GroupSpec", "ArraySpec"]
type TBaseMembers = Mapping[str, TBaseItem]

# Aliases for specs of any specialization: ordinary wide types, equal to the
# bare class names, since the covariant parameters make a narrowed spec
# substitutable for a wider one — see the note on the v3 aliases.
type AnyArraySpec = "ArraySpec[TBaseAttr]"
type AnyGroupSpec = "GroupSpec[TBaseAttr, TBaseMembers]"
type AnyBaseGroupSpec = "BaseGroupSpec[TBaseAttr]"

# covariant because the specs are frozen; see the note on the v3 TypeVars
TAttr = TypeVar("TAttr", bound=TBaseAttr, default=TBaseAttr, covariant=True)
TMembers = TypeVar("TMembers", bound=TBaseMembers, default=TBaseMembers, covariant=True)
TDefault = TypeVar("TDefault")

DtypeStr = Annotated[str, BeforeValidator(parse_dtype_v2)]

BoolFillValue = bool
IntFillValue = int
# todo: introduce a type that represents hexadecimal representations of floats
FloatFillValue = ZarrV2FloatSpecial | float
ComplexFillValue = tuple[FloatFillValue, FloatFillValue]
RawFillValue = tuple[int, ...]

FillValue = (
    BoolFillValue | IntFillValue | FloatFillValue | ComplexFillValue | RawFillValue | str | None
)

FillValueLike = FillValue | np.generic | complex
"""
A fill value as an input takes it: its JSON form, or a numpy scalar or a Python
complex number, which validation writes in JSON form.
"""

DimensionSeparator = ZarrV2ArrayDimensionSeparator
MemoryOrder = ZarrV2ArrayOrder

type ZarrV2StoredDocumentJSON = ZarrV2ZArrayJSON | ZarrV2ZGroupJSON | ZarrV2ZAttrsJSON
"""Any document a Zarr V2 hierarchy stores: a node's `.zarray`, `.zgroup` or `.zattrs`."""

# The documents a store holds for a node, keyed by store key. The keys are not
# identifiers, hence the functional syntax. The array type is closed, and the
# group type admits only its members' documents beyond its own, so each is
# assignable to a `Mapping[str, ...]` of its value types.
ZarrV2ArrayStoredJSON = TypedDict(
    "ZarrV2ArrayStoredJSON",
    {".zarray": ZarrV2ZArrayJSON, ".zattrs": ZarrV2ZAttrsJSON},
    closed=True,
)
"""A Zarr V2 array's metadata as a store holds it: its `.zarray` and `.zattrs` documents."""
ZarrV2GroupStoredJSON = TypedDict(
    "ZarrV2GroupStoredJSON",
    {".zgroup": ZarrV2ZGroupJSON, ".zattrs": ZarrV2ZAttrsJSON},
    extra_items=ZarrV2StoredDocumentJSON,
)
"""
A Zarr V2 group's metadata as a store holds it: its own `.zgroup` and
`.zattrs` documents, and each member's prefixed by the member's normalized
logical path relative to the group, which has no leading `/`
(`"a/b/.zarray"` or `"a/b/.zgroup"`, and `"a/b/.zattrs"`).
"""


def dictify_codec(value: object) -> object:
    """
    Ensure that a `numcodecs.abc.Codec` is converted to a `dict`. Any other
    input is returned unaltered, for validation to judge.

    Parameters
    ----------

    value : object
        The value to be dictified if it is a `numcodecs.abc.Codec`.

    Returns
    -------
    object
        If the input was a `Codec`, then the result of calling `get_config()` on that
        object is returned. This should be a dict with string keys. All other values pass
        through unaltered.
    """
    if _is_codec(value):
        return value.get_config()
    return value


def _is_codec(obj: object) -> TypeIs[Codec]:
    """
    Whether `obj` is a `numcodecs.abc.Codec`, without importing numcodecs: an
    instance cannot exist unless numcodecs is imported.
    """
    numcodecs = sys.modules.get("numcodecs")
    return numcodecs is not None and isinstance(obj, numcodecs.abc.Codec)


def parse_dimension_separator(data: object) -> DimensionSeparator:
    """
    Parse the dimension_separator metadata as per the Zarr version 2 specification.
    If the input is `None`, this returns ".".
    If the input is either "." or "/", this returns it.
    Otherwise, raises a ValueError.

    Parameters
    ----------
    data : object
        The input data to parse.

    Returns
    -------
    Literal["/", "."]
    """
    if data is None:
        return ZARR_V2_ARRAY_DEFAULT_DIMENSION_SEPARATOR
    if is_zarr_v2_array_dimension_separator(data):
        return data
    expected = (*ZARR_V2_ARRAY_DIMENSION_SEPARATOR, None)
    raise ValueError(f"Invalid data, expected one of {expected}, got {data}")


CodecDict = Annotated[dict[str, object], BeforeValidator(dictify_codec)]


class NodeSpec(StrictBase):
    """
    The base class for V2 `ArraySpec` and `GroupSpec`.

    Attributes
    ----------

    zarr_format: Literal[2]
        The Zarr version represented by this node. Must be 2.
    """

    zarr_format: Literal[2] = 2

    def __replace__(self, **changes: object) -> Self:
        """
        `copy.replace(spec, **changes)` (Python 3.13 and later): the spec's own
        `replace`, so it validates the changes, where pydantic's
        `BaseModel.__replace__` copies without validating. It calls
        `self.replace`, so a subclass that overrides `replace` is honored.
        """
        # `Any`: each subclass's `replace` takes its own TypedDict of changes,
        # which no signature declared here can name, and validates `changes`
        return cast("Any", self).replace(**changes)


class ArraySpecUpdate(TypedDict, total=False, closed=True):
    """
    The fields an `ArraySpec.replace` call may replace, of the types an input
    takes: validation writes them as the types the spec holds (a numpy dtype as
    its typestr, a numpy scalar fill value in its JSON form, a numcodecs codec
    as its configuration mapping, a sequence as a tuple).
    """

    attributes: TBaseAttr
    shape: Sequence[int]
    chunks: Sequence[int]
    dtype: npt.DTypeLike | list[tuple[object, ...]]
    fill_value: FillValueLike
    order: MemoryOrder
    filters: Sequence[Mapping[str, object] | Codec] | None
    dimension_separator: DimensionSeparator
    compressor: Mapping[str, object] | Codec | None
    extra_fields: Mapping[str, JSONValue]


class BaseGroupSpecUpdate(TypedDict, total=False, closed=True):
    """The fields a `BaseGroupSpec.replace` call may replace."""

    attributes: TBaseAttr


class GroupSpecUpdate(TypedDict, total=False, closed=True):
    """The fields a `GroupSpec.replace` call may replace."""

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
    fill_value: FillValue
        The fill value for this array, in its Zarr V2 JSON form: numpy scalars
        are stored as Python scalars, complex numbers as ``(real, imaginary)``
        pairs, and non-finite floats as ``"NaN"``, ``"Infinity"`` or
        ``"-Infinity"``. It must be a fill value the dtype accepts. The
        default is the one `zarr-metadata` derives for the dtype (see
        `dtype_default_fill_value`).
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
        for `model_dump`, `model_dump_json`, `replace` and `like`.
        zarr-python drops them when it reads or writes an array. Group
        metadata, where "Other keys MUST NOT be present", takes none.
    """

    attributes: TAttr = Field(default={}, validate_default=True)
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr | list[tuple[object, ...]]
    # validation replaces this declared default with the dtype's (see
    # `default_fill_value`); `model_construct` and the JSON schema, which do
    # not validate, still report 0 for every dtype
    fill_value: Annotated[FillValue, BeforeValidator(canonicalize_fill_value)] = 0
    order: MemoryOrder = "C"
    filters: list[CodecDict] | None = None
    dimension_separator: Annotated[
        DimensionSeparator, BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None

    @model_validator(mode="before")
    @classmethod
    def default_fill_value(cls, data: object) -> object:
        """
        Give a spec built without a `fill_value` the default `zarr-metadata`
        derives for its dtype (see `dtype_default_fill_value`). A subclass that
        declares its own `fill_value` default, or makes the field required,
        keeps that instead.
        """
        declared = cls.model_fields["fill_value"].default
        if type(declared) is not int or declared != 0:
            return data
        if isinstance(data, Mapping) and "fill_value" not in data and "dtype" in data:
            data = {**data, "fill_value": dtype_default_fill_value(data["dtype"])}
        return data

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

    @model_validator(mode="after")
    def validate_metadata_document(self) -> Self:
        """
        Validate this spec as a Zarr V2 array metadata document, via `zarr-metadata`.

        The document assembled from this spec's fields is read in
        `zarr_metadata`'s core Zarr V2 scope, which judges the fill value
        against the dtype among other things, so a spec holds only what a
        Zarr V2 `.zarray` document can store. The `attributes` field is not
        part of this check: its type and contents are governed by this
        class's `TAttr` parameter alone.
        """
        document = self.model_dump()
        document["attributes"] = {}
        try:
            ZarrV2ArrayMetadataModel.from_json(document)
        except MetadataValidationError as e:
            raise ValueError(format_problems(e, zarr_format=2)) from e
        return self

    def to_metadata_json(self) -> ZarrV2ArrayMetadataJSON:
        """
        This array's Zarr V2 metadata, as one document holding its `.zarray`
        and, under `attributes`, its `.zattrs`.

        Unlike `model_dump`, this takes no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV2ArrayMetadataJSON`. The stored
        `.zarray` is this document without `attributes` (the type
        `zarr_metadata.ZarrV2ZArrayJSON`), and the stored `.zattrs` is
        `attributes`. The JSON arrays the type declares as tuples (the
        `filters` pipeline, and a structured `dtype`) are tuples here too;
        JSON arrays inside a codec's configuration are left as they are.
        Its `extra_fields` are written as top-level members.
        """
        document = self.model_dump()
        document["dtype"] = tuplify_json(document["dtype"])
        if document["filters"] is not None:
            document["filters"] = tuple(document["filters"])
        return cast("ZarrV2ArrayMetadataJSON", document)

    def to_stored_metadata_json(self) -> ZarrV2ArrayStoredJSON:
        """
        This array's Zarr V2 metadata as a store holds it: a mapping from each
        store key to the JSON document stored there.

        The `.zarray` document is `to_metadata_json()` without `attributes`,
        and the `.zattrs` document is `attributes`. `.zattrs` is written even
        when the attributes are empty, as zarr-python writes it.
        `from_stored_metadata_json` reads it back.
        """
        document = self.to_metadata_json()
        zattrs = document.pop("attributes", {})
        return {
            ZARR_V2_ARRAY_METADATA_STORE_KEY: cast("ZarrV2ZArrayJSON", document),
            ZARR_V2_ATTRIBUTES_STORE_KEY: zattrs,
        }

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read an array from its metadata as a store holds it: a mapping from
        store key to parsed JSON document, as `to_stored_metadata_json`
        returns it. A missing `.zattrs` reads as empty attributes. Raises
        `ValueError` when `data` holds no array's `.zarray` at the root, or
        holds documents of any other node, and `ValidationError` when the
        documents are not a valid array.
        """
        return cls.model_validate(stored_array(_stored_nodes(data)))

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.generic] | zarr.Array,
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
            The fill value for this array. Either "auto" or FillValue. The default is "auto", which means that `array.fill_value` will be used if that attribute exists, with a fallback of the dtype's default (see `dtype_default_fill_value`).
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

        # validated as a document: the fill value and codecs read from `array`
        # are whatever its attributes hold, for validation to judge
        return cls.model_validate(
            {
                "shape": shape_actual,
                "dtype": parse_dtype_v2(dtype_actual),
                "chunks": chunks_actual,
                "attributes": attributes_actual,
                "fill_value": fill_value_actual,
                "order": order_actual,
                "compressor": None
                if compressor_actual is None
                else dictify_codec(compressor_actual),
                "filters": None
                if filters_actual is None
                else [dictify_codec(f) for f in filters_actual],
                "dimension_separator": dimension_separator_actual,
            }
        )

    def replace(self, **changes: Unpack[ArraySpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so correlated fields (a new
        `dtype` with a matching `fill_value`, or a new `shape` with matching
        `chunks`) change together atomically. A new `dtype` keeps the current
        `fill_value`, which is not re-derived: when the new dtype does not
        accept it, pass a `fill_value` in the same call. Unlike
        `model_copy(update=...)`, the result is validated. Passing
        `extra_fields` replaces the extra fields as a whole.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

    def like(
        self,
        other: ArraySpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare an `ArraySpec` to another `ArraySpec`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to `dict` via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other : ArraySpec
            The array spec to compare with.
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
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = ArraySpec(shape=(10, 10), chunks=(10, 10), dtype='uint8', attributes={'foo': 10})
        >>> print(x.like(x)) # it is like itself.
        True
        >>> y = ArraySpec(shape=(10, 10), chunks=(10, 10), dtype='uint8', attributes={'foo': 11})
        >>> print(x.like(y)) # x and y are the same, other than their attributes
        False
        >>> print(x.like(y, exclude={'attributes'}))
        True
        """
        return model_like(self, other, include=include, exclude=exclude)


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

    def replace(self, **changes: Unpack[BaseGroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        Unlike `model_copy(update=...)`, the result is validated, so an
        invalid change is refused.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return type(self).model_validate({**self.model_dump(), **changes})

    def to_metadata_json(self) -> ZarrV2GroupMetadataJSON:
        """
        This group's Zarr V2 metadata, as one document holding its `.zgroup`
        and, under `attributes`, its `.zattrs`.

        Unlike `model_dump`, this takes no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV2GroupMetadataJSON`. The stored
        `.zgroup` is this document without `attributes` (the type
        `zarr_metadata.ZarrV2ZGroupJSON`), and the stored `.zattrs` is
        `attributes`. A group's members are not part of its metadata, so a
        `GroupSpec`'s `members` are left out.
        """
        return cast("ZarrV2GroupMetadataJSON", self.model_dump(exclude={"members"}))

    def to_stored_metadata_json(self) -> ZarrV2GroupStoredJSON:
        """
        This group's Zarr V2 metadata as a store holds it: a mapping from each
        store key to the JSON document stored there.

        The `.zgroup` document is `to_metadata_json()` without `attributes`,
        and the `.zattrs` document is `attributes`. `.zattrs` is written even
        when the attributes are empty, as zarr-python writes it. A
        `BaseGroupSpec` holds no members, so these are the only keys;
        `GroupSpec` adds each member's documents.
        """
        document = self.to_metadata_json()
        return {
            # `.zgroup` holds no other members
            ZARR_V2_GROUP_METADATA_STORE_KEY: {"zarr_format": document["zarr_format"]},
            ZARR_V2_ATTRIBUTES_STORE_KEY: document.get("attributes", {}),
        }

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a group's own metadata from a stored layout: a mapping from store
        key to parsed JSON document, as `to_stored_metadata_json` returns it.
        Only the root's `.zgroup` and `.zattrs` are read; members' documents
        are left unread. A missing `.zattrs`
        reads as empty attributes. Raises `ValueError` when `data` holds no
        group's `.zgroup` at the root, and `ValidationError` when the
        documents are not a valid group.
        """
        own = {key: document for key, document in data.items() if "/" not in key}
        return cls.model_validate(stored_root(_stored_nodes(own), "group"))

    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare this group's own metadata to another group's. Members never
        participate in the comparison: a `GroupSpec` is reduced to its own
        metadata first.
        """
        other_parsed: AnyBaseGroupSpec
        if isinstance(other, GroupSpec):
            other_parsed = BaseGroupSpec(attributes=other.attributes)
        else:
            other_parsed = other

        return model_like(self, other_parsed, include=include, exclude=exclude)


class GroupSpec(BaseGroupSpec[TAttr], Generic[TAttr, TMembers]):
    """
    A model of a Zarr Version 2 Group.
    The specification for the data structure being modeled by this
    class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#groups).

    The second type parameter is the type of the whole `members` mapping. Leave it
    out for a group of any members, give a `Mapping` to constrain every member's
    type (`GroupSpec[Attrs, Mapping[str, MyArraySpec]]`), or give a closed
    `TypedDict` to fix the member names (`GroupSpec[Attrs, MyMembers]`).

    Attributes
    ----------

    attributes: TAttr
        The user-defined attributes of this group. Should be JSON-serializable.
    members: TMembers, default = {}
        The members of this group: a mapping from member name to member, of the
        type `TMembers`. With the default `TMembers` every member is validated as an
        `ArraySpec` or a `GroupSpec`; under `Any` the values are not validated. The
        spec holds its own `dict` copy of the mapping it is given. A `GroupSpec` is a
        complete hierarchy, so an empty mapping means the group has no members. A
        group whose members are unknown is a `BaseGroupSpec`, which has no `members`
        field and is never a member. The default is validated against `TMembers`, so
        a group whose members type requires a member still requires `members`.
    """

    # the default is cast so that a checker does not solve TMembers from it:
    # an unparameterized GroupSpec() then has the default members type
    members: TMembers = Field(default=cast("TMembers", {}), validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def _copy_members(cls, data: object) -> object:
        # a model validator, not a field one: pyrefly and ty widen the
        # `__init__` parameter of a field with a before validator to `Any`
        if isinstance(data, Mapping) and "members" in data:
            return {**data, "members": copy_members(data["members"])}
        return data

    @model_validator(mode="after")
    def validate_member_names(self) -> Self:
        """
        Validate member names (no ``"/"``, not empty, not ``"."``/``".."``).

        This runs as a model validator rather than field metadata so that a
        subclass narrowing the `members` annotation (for example to a closed
        `TypedDict` of members) cannot drop it.
        """
        ensure_members_mapping(self.members)
        ensure_key_no_path(self.members)
        return self

    @override
    def replace(self, **changes: Unpack[GroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so `attributes` and `members`
        change together atomically. Unlike `model_copy(update=...)`, the
        result is validated.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
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

    @override
    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare a `GroupSpec` to another `GroupSpec` or `BaseGroupSpec`,
        parameterized over the fields to exclude or include in the comparison. Models are first
        converted to dict via the `model_dump` method of `pydantic.BaseModel`, then compared with
        the `==` operator. A `BaseGroupSpec` — a group considered without members — compares as
        that projection: by the group's own metadata alone, whatever this side's members hold.

        Parameters
        ----------
        other : GroupSpec | BaseGroupSpec
            The group spec to compare with.
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
        >>> from pydantic_zarr.v2 import ArraySpec, GroupSpec
        >>> foo = ArraySpec(shape=(10,), chunks=(10,), dtype='int64')
        >>> z1 = GroupSpec(members={'foo': foo})
        >>> print(z1.like(z1)) # it is like itself
        True
        >>> z2 = GroupSpec(members={'foo': foo})
        >>> print(z1.like(z2))
        True
        >>> z2 = z2.replace(attributes={'foo': 100}) # now they have different attributes
        >>> print(z1.like(z2))
        False
        >>> print(z1.like(z2, exclude={'attributes'}))
        True
        """
        if not isinstance(other, GroupSpec):
            # a group without members compares by the group's own metadata;
            # this side projects to its own metadata too, or the comparison
            # would always fail on the members field itself
            projection = BaseGroupSpec(attributes=self.attributes)
            return projection.like(other, include=include, exclude=exclude)

        return model_like(self, other, include=include, exclude=exclude)

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

    @override
    def to_stored_metadata_json(self) -> ZarrV2GroupStoredJSON:
        """
        This hierarchy's Zarr V2 metadata as a store holds it: a mapping from
        each store key to the JSON document stored there. That is this
        group's `.zgroup` and `.zattrs`, and each member's at any depth,
        prefixed by the member's path (`"a/.zgroup"`, `"a/b/.zarray"`,
        `"a/b/.zattrs"`). The prefix is the member's path as `to_flat` spells
        it (`"/a/b"`) normalized as the Zarr V2 spec normalizes logical paths,
        which strips the leading `/`
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L250-L276).
        These are the metadata documents zarr-python writes,
        so a consumer can write the hierarchy by writing each document, as
        JSON, at its key under the hierarchy's root.
        `from_stored_metadata_json` reads it back.
        """
        members: dict[str, ZarrV2StoredDocumentJSON] = {
            f"{path.removeprefix('/')}/{key}": document
            for path, node in self.to_flat().items()
            if path
            for key, document in _node_documents(node)
        }
        # each member's document is of the type's `extra_items`; the cast is for
        # checkers that do not yet type writes to `extra_items` (pyrefly)
        return cast("ZarrV2GroupStoredJSON", {**super().to_stored_metadata_json(), **members})

    @override
    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a hierarchy from its metadata as a store holds it: a mapping from
        store key to parsed JSON document, as `to_stored_metadata_json`
        returns it. Each node's documents are read, a missing `.zattrs` as
        empty attributes, and the nodes are assembled with `from_flat`. A
        layout cannot tell a group whose members were not read from one with
        none, so every group is read as a `GroupSpec`, with the members the
        layout holds.

        Raises `ValueError` when `data` holds no group's `.zgroup` at the
        root, a key with a leading `/` or an empty segment, a key other than
        a `.zarray`, `.zgroup` or `.zattrs`, a path
        with both a `.zarray` and a `.zgroup`, a `.zattrs` with neither, or a
        node whose parent is not a group in `data` (a store holds no implicit
        groups), and `ValidationError` when a node's documents are not valid.
        """
        return cls.from_flat(stored_flat(_stored_nodes(data), ArraySpec, BaseGroupSpec))

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


def _node_documents(
    node: AnyArraySpec | AnyBaseGroupSpec,
) -> tuple[tuple[str, ZarrV2StoredDocumentJSON], ...]:
    """A node's own stored documents, as (store key, document) pairs."""
    if isinstance(node, ArraySpec):
        array = node.to_stored_metadata_json()
        return (
            (ZARR_V2_ARRAY_METADATA_STORE_KEY, array[ZARR_V2_ARRAY_METADATA_STORE_KEY]),
            (ZARR_V2_ATTRIBUTES_STORE_KEY, array[ZARR_V2_ATTRIBUTES_STORE_KEY]),
        )
    group = node.to_stored_metadata_json()
    return (
        (ZARR_V2_GROUP_METADATA_STORE_KEY, group[ZARR_V2_GROUP_METADATA_STORE_KEY]),
        (ZARR_V2_ATTRIBUTES_STORE_KEY, group[ZARR_V2_ATTRIBUTES_STORE_KEY]),
    )


_STORED_NODE_KEYS = (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    ZARR_V2_ATTRIBUTES_STORE_KEY,
)


def _stored_nodes(data: Mapping[str, object]) -> StoredNodes:
    """
    Read a stored layout, a mapping from store key to parsed JSON document,
    as each node's type and merged metadata document (its `.zarray` or
    `.zgroup` with its `.zattrs` under `attributes`, empty when missing),
    keyed by the node's key prefix (see `split_store_key`): `""` for the
    root, `"a/b"` for the node at logical path `a/b`. Raises `ValueError` naming
    a malformed key, a key other than a node's `.zarray`, `.zgroup` or
    `.zattrs`, a path with both a `.zarray` and a `.zgroup`, or a `.zattrs`
    with neither.
    """
    files_by_path: dict[str, dict[str, object]] = {}
    for key, document in data.items():
        path, name = split_store_key(key)
        if name not in _STORED_NODE_KEYS:
            expected = ", ".join(map(repr, _STORED_NODE_KEYS))
            msg = (
                f"Unrecognized store key {key!r}: a Zarr V2 node's metadata is stored at "
                f"{expected}."
            )
            raise ValueError(msg)
        files_by_path.setdefault(path, {})[name] = document
    nodes: dict[str, tuple[StoredNodeType | None, object]] = {}
    for path, files in files_by_path.items():
        where = repr(path) if path else "the root"
        node_type: StoredNodeType
        if ZARR_V2_ARRAY_METADATA_STORE_KEY in files and ZARR_V2_GROUP_METADATA_STORE_KEY in files:
            msg = (
                f"The layout holds both a {ZARR_V2_ARRAY_METADATA_STORE_KEY} and a "
                f"{ZARR_V2_GROUP_METADATA_STORE_KEY} at {where}: a node is an array or a "
                "group, not both."
            )
            raise ValueError(msg)
        if ZARR_V2_ARRAY_METADATA_STORE_KEY in files:
            node_type, metadata = "array", files[ZARR_V2_ARRAY_METADATA_STORE_KEY]
        elif ZARR_V2_GROUP_METADATA_STORE_KEY in files:
            node_type, metadata = "group", files[ZARR_V2_GROUP_METADATA_STORE_KEY]
        else:
            msg = (
                f"The layout holds a {ZARR_V2_ATTRIBUTES_STORE_KEY} at {where}, but no "
                f"{ZARR_V2_ARRAY_METADATA_STORE_KEY} or {ZARR_V2_GROUP_METADATA_STORE_KEY}."
            )
            raise ValueError(msg)
        attributes = files.get(ZARR_V2_ATTRIBUTES_STORE_KEY, {})
        # a document that is not a JSON object is passed on for validation to refuse
        document = (
            {**metadata, "attributes": attributes} if isinstance(metadata, Mapping) else metadata
        )
        nodes[path] = (node_type, document)
    return nodes


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


def auto_chunks(data: object) -> tuple[int, ...]:
    """
    Guess chunks from:
      input with a `chunksize` attribute, or
      input with a `chunks` attribute, or,
      input with `shape` and `dtype` attributes
    """
    if hasattr(data, "chunksize"):
        return as_shape(read_attribute(data, "chunksize"))
    if hasattr(data, "chunks"):
        return as_shape(read_attribute(data, "chunks"))
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        return guess_chunks(
            as_shape(read_attribute(data, "shape")),
            np.dtype(as_dtype_like(read_attribute(data, "dtype"))).itemsize,
        )
    msg = "Cannot guess chunks for an object without `chunksize`, `chunks`, or `shape` and `dtype`"
    raise ValueError(msg)


def auto_attributes(data: object) -> Mapping[str, object] | BaseModel:
    """
    Guess attributes from:
        input with an `attrs` attribute, or
        input with an `attributes` attribute,
        or anything (returning {})

    The attributes found are a mapping, or a pydantic model (the attributes of
    a spec). Any other value is skipped.
    """
    for name in ("attrs", "attributes"):
        if hasattr(data, name):
            attributes = read_attribute(data, name)
            if isinstance(attributes, Mapping | BaseModel):
                return attributes
    return {}


_DATA_TYPE_ADAPTER: Final = TypeAdapter(ZarrV2DataTypeMetadata)
"""Checks that a value is a Zarr V2 data type in its JSON form."""


def dtype_default_fill_value(dtype: npt.DTypeLike) -> FillValue:
    """
    The fill value a spec of data type `dtype` gets when none is given: the
    default `zarr-metadata` derives for the data type, the value zarr-python
    uses (zero, `False`, the empty string, null bytes, `"NaT"`), or `None`
    for `|O`, whose values may be of any type. A `dtype` that is not a valid
    Zarr V2 data type gets `0`, and validation reports the data type.
    """
    try:
        data_type = _DATA_TYPE_ADAPTER.validate_python(parse_dtype_v2(dtype))
        return cast(
            "FillValue", ZarrV2ArrayMetadataModel.create_default(dtype=data_type).fill_value
        )
    # a pydantic ValidationError is a ValueError
    except (MetadataValidationError, TypeError, ValueError):
        return 0


def auto_fill_value(data: object) -> object:
    """
    Guess fill value from an input with a `fill_value` attribute. Otherwise
    return the default for the input's dtype (see `dtype_default_fill_value`),
    or 0 for an input with no dtype.
    """
    if hasattr(data, "fill_value"):
        return read_attribute(data, "fill_value")
    if hasattr(data, "dtype"):
        return dtype_default_fill_value(as_dtype_like(read_attribute(data, "dtype")))
    return 0


def _as_codec(value: object) -> Mapping[str, object] | Codec:
    """
    `value`, a codec read from a duck-typed array: a `numcodecs` codec, or the
    JSON form of one (as a spec holds it).

    Raises
    ------
    TypeError
        If `value` is neither.
    """
    if _is_codec(value) or isinstance(value, Mapping):
        return value
    msg = f"Expected a numcodecs codec or a mapping, got {value!r}."
    raise TypeError(msg)


def auto_compresser(data: object) -> Mapping[str, object] | Codec | None:
    """
    Guess compressor from an input with a `compressor` attribute, returning `None` otherwise.
    The compressor found is a `numcodecs` codec or the JSON form of one.

    Raises
    ------
    TypeError
        If the `compressor` attribute holds anything else.
    """
    if hasattr(data, "compressor"):
        compressor = read_attribute(data, "compressor")
        return None if compressor is None else _as_codec(compressor)
    return None


def auto_filters(data: object) -> list[Mapping[str, object] | Codec] | None:
    """
    Guess filters from an input with a `filters` attribute, returning `None` otherwise.
    Each filter found is a `numcodecs` codec or the JSON form of one.

    Raises
    ------
    TypeError
        If the `filters` attribute is not `None` or an iterable of those.
    """
    if hasattr(data, "filters"):
        filters = read_attribute(data, "filters")
        if filters is None:
            return None
        if not isinstance(filters, Iterable) or isinstance(filters, str | Mapping):
            msg = f"Expected an iterable of codecs, got {filters!r}."
            raise TypeError(msg)
        return [_as_codec(f) for f in filters]
    return None


def auto_order(data: object) -> MemoryOrder:
    """
    Guess array order from an input with an `order` attribute, returning "C" otherwise.
    """
    if hasattr(data, "order"):
        order = read_attribute(data, "order")
        if is_zarr_v2_array_order(order):
            return order
        raise ValueError(f"Order attribute not in {ZARR_V2_ARRAY_ORDER}")
    return "C"


def auto_dimension_separator(data: object) -> DimensionSeparator:
    """
    Guess dimension separator from an input with a `dimension_separator` attribute, returning "/" otherwise.
    """
    if hasattr(data, "dimension_separator"):
        separator = read_attribute(data, "dimension_separator")
        if is_zarr_v2_array_dimension_separator(separator):
            return separator
        raise ValueError(
            f"Dimension separator attribute not in {ZARR_V2_ARRAY_DIMENSION_SEPARATOR}"
        )
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
