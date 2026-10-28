from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    NotRequired,
    Self,
    Union,
    Unpack,
    cast,
    overload,
    override,
)

import numpy as np
import numpy.typing as npt
from packaging.version import Version
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    ValidationError,
    ValidationInfo,
    model_serializer,
    model_validator,
)
from typing_extensions import TypedDict, TypeIs, TypeVar
from zarr_metadata import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    DefaultChunkKeyEncodingMetadata,
    JSONValue,
    MetadataValidationError,
    RegularChunkGridMetadata,
    V2ChunkKeyEncodingMetadata,
    ZarrV3ExtensionField,
)
from zarr_metadata.definition._envelope import envelope_problems
from zarr_metadata.model import ZarrV3ArrayMetadata as ZarrV3ArrayMetadataModel
from zarr_metadata.model import ZarrV3GroupMetadata as ZarrV3GroupMetadataModel
from zarr_metadata.model import validate_node_name_v3
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGridConfiguration
from zarr_metadata.v3.chunk_key_encoding.default import (
    DefaultChunkKeyEncodingConfiguration,
    DefaultChunkKeyEncodingObject,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2ChunkKeyEncodingConfiguration,
    V2ChunkKeyEncodingObject,
)
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    ArrayExtraFieldDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    GroupExtraFieldDefinition,
    StorageTransformerDefinition,
)

from pydantic_zarr._metadata import CONTEXT_KEY as CONTEXT_KEY
from pydantic_zarr._metadata import format_lines, format_problems, scope_from_context
from pydantic_zarr.core import (
    ExtraFieldsBase,
    IncEx,
    ensure_multiple,
    flatten_extra_fields,
    guess_chunks,
    maybe_node,
    model_like,
    resolve_member_path,
    tuplify_json,
    walk_zarr_hierarchy,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    import zarr
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams
    from zarr_metadata.v3.definition import ZarrV3Context


# The BaseModel arm is SerializeAsAny so that an attribute model held by a
# spec whose attributes type is not narrowed (a bare ArraySpec / GroupSpec)
# serializes with the model's own fields; annotated as plain BaseModel it
# would serialize through the base schema, silently dumping {}.
type TBaseAttr = Mapping[str, object] | SerializeAsAny[BaseModel]
# BaseGroupSpec is not a member type: a GroupSpec is a complete hierarchy,
# and a group whose members are unknown is never part of one.
type TBaseItem = Union["GroupSpec", "ArraySpec"]

# Aliases for specs of any specialization. Since the spec parameters are
# covariant (specs are frozen, so a narrowed spec IS-A wider one), these are
# ordinary wide types, not gradual-typing escapes: a value typed with one
# keeps checked, TBaseAttr/TBaseItem-typed reads. The names remain the public
# spelling for heterogeneous APIs (`resolve_path`, `from_flat`, `like`,
# `is_array_spec`) and are now equal to the bare class names.
type AnyArraySpec = "ArraySpec[TBaseAttr]"
type AnyGroupSpec = "GroupSpec[TBaseAttr, TBaseItem]"
type AnyBaseGroupSpec = "BaseGroupSpec[TBaseAttr]"

# Covariant because the specs are frozen: TAttr and TItem appear only in read
# positions on instances, so ArraySpec[CFAttrs] is substitutable for
# ArraySpec[TBaseAttr], and TBaseItem serves as both TItem's bound and its
# default (its bare, recursive spelling also makes pydantic validate a bare
# GroupSpec's nested members as specs all the way down;
# tests/typing_conformance.py guards the narrowed-parameterization contract).
# Constructing classmethods take TBaseAttr instead of TAttr, since a covariant
# parameter may not appear in argument positions; the class's real attribute
# type is still enforced, by validation.
TAttr = TypeVar("TAttr", bound=TBaseAttr, default=TBaseAttr, covariant=True)
TItem = TypeVar("TItem", bound=TBaseItem, default=TBaseItem, covariant=True)
TDefault = TypeVar("TDefault")

NodeType = Literal["group", "array"]

FillValue = JSONValue
"""
The JSON form of a fill value. Which values are valid depends on the array's
data type, and is judged by that data type's definition when the whole
metadata document is validated; at the field level any JSON value is a
fill value. `zarr-metadata` exports per-data-type fill value types
(``Float32FillValue`` and so on) for callers that want the narrow forms.
"""

TName = TypeVar("TName", bound=str, default=str)
TConfig = TypeVar("TConfig", bound=Mapping[str, object], default=Mapping[str, object])


class NamedConfig(TypedDict, Generic[TName, TConfig]):
    """
    A Zarr V3 metadata object.

    This class is parametrized by two type parameters: `TName` and `TConfig`.

    Attributes
    ----------
    name: TName
        The name of the metadata object.
    configuration: NotRequired[TConfig]
        The configuration of the metadata object.
    """

    name: TName
    configuration: NotRequired[TConfig]


class AnyNamedConfig(NamedConfig[str, Mapping[str, object]]):
    """
    This class models any Zarr metadata object that takes the form of a
    {"name": ..., "configuration": ...} dict, where the "configuration" key is not required.

    `zarr_metadata.ZarrV3NamedConfigJSON` is this shape's canonical type; this
    local equivalent exists because that one marks its members `ReadOnly`,
    which pydantic answers with an advisory warning at every schema build
    (each parametrization or subclassing of a model would warn).
    """


CodecLike = str | AnyNamedConfig
"""A codec declaration: an extension name, or a named configuration object."""

# The structured Zarr V3 metadata types are defined once, by `zarr-metadata`;
# the names this module exported before it delegated to that library are kept
# as aliases. The aliased TypedDicts are *closed*: a member their definition
# does not declare is rejected rather than silently dropped.
RegularChunkingConfig = RegularChunkGridConfiguration
RegularChunking = RegularChunkGridMetadata
DefaultChunkKeyEncodingConfig = DefaultChunkKeyEncodingConfiguration
DefaultChunkKeyEncoding = DefaultChunkKeyEncodingObject
V2ChunkKeyEncodingConfig = V2ChunkKeyEncodingConfiguration
V2ChunkKeyEncoding = V2ChunkKeyEncodingObject


class NodeSpec(ExtraFieldsBase):
    """
    The base class for V3 ArraySpec and GroupSpec.

    Attributes
    ----------

    zarr_format: Literal[3]
        The Zarr version represented by this node. Must be 3.
    zarr_scope: ClassVar[ZarrV3Context | None]
        The scope this class validates metadata in when the validation
        context carries none, making the scope a property of the model
        rather than of each ``model_validate`` call: a subclass that sets
        it keeps its scope through direct construction and every derived
        operation (``update``, ``from_shape_and_data_type``, ``like``, ...),
        which re-validate as the class. An explicit validation context
        still wins, and ``None`` (the default) means ``zarr-metadata``'s
        ``CORE_AND_EXTENSIONS``. A group's members are validated as their
        own classes, so a scoped tree narrows its member types to
        scope-bearing classes (an explicit context, which pydantic carries
        into nested validation, reaches the members either way).
    zarr_scope_strict: ClassVar[bool]
        When true, validation additionally refuses every extension-point
        name — data type, chunk grid, chunk key encoding, codec (a sharding
        codec's inner and index pipelines included) and storage transformer
        — that nothing in the operative scope claims, where the default is
        to leave such names unjudged. Together with `zarr_scope` this makes
        a class a conformance contract: ``zarr_scope = CORE`` plus strict
        accepts exactly the documents built from names the core spec
        defines. The variable-length data types and their codecs are
        extensions (and zarr-python's legacy spellings are claimed by no
        scope), so a strict core class refuses them. Enforced by
        `ArraySpec`'s document validation; a group document has no
        extension points, and its members are validated as their own
        classes. Strict validation also refuses every extra field whose key
        nothing in the scope claims.

    Notes
    -----
    A metadata document may hold members the Zarr V3 spec does not define
    (its "additional fields"), and a spec accepts the ones the spec permits,
    judged in the same scope as the rest of the document. An extra field
    whose key a definition in the scope claims is judged by that definition
    (``CORE`` claims a group's ``consolidated_metadata``). Any other extra
    field is one the reader does not recognize, which the spec admits only
    as an extension object, ``{"name": ..., "configuration": ...}``, that
    sets ``"must_understand": false``; every other unknown key is refused,
    which also catches a misspelled field name. Accepted extra fields are
    kept in the `extra_fields` mapping (see `ExtraFieldsBase`) and written
    back as top-level members wherever the document is written:
    ``model_dump``, ``model_dump_json``, ``update``, ``like``, ``to_zarr``
    and ``from_zarr``.
    """

    zarr_format: Literal[3] = 3
    zarr_scope: ClassVar[ZarrV3Context | None] = None
    zarr_scope_strict: ClassVar[bool] = False

    @classmethod
    def _scope(cls, context: object) -> ZarrV3Context | None:
        """
        The scope a validation of this class reads metadata in: the validation
        context's scope when it carries one, else this class's `zarr_scope`.
        `None` means the readers' default (``CORE_AND_EXTENSIONS``).
        """
        scope = scope_from_context(context)
        return cls.zarr_scope if scope is None else scope

    def _refuse_unjudged(
        self,
        kind: ExtraFieldKind,
        scope: ZarrV3Context | None,
        unclaimed_extension_points: Sequence[str] = (),
    ) -> None:
        """
        Refuse what a whole-document read in `scope` (``CORE_AND_EXTENSIONS``
        when `None`) leaves unjudged and the spec or this class does not
        allow: the extra fields the spec's rule for fields a reader does not
        recognize refuses, and, on a class with `zarr_scope_strict` set, every
        extra field and extension-point name (`unclaimed_extension_points`)
        nothing in `scope` claims. Extra fields a definition in `scope` claims
        are judged by that definition when the whole document is read, not
        here.
        """
        extra = self.extra_fields
        scope_actual = CORE_AND_EXTENSIONS if scope is None else scope
        node_type = "array" if kind is ArrayExtraFieldDefinition else "group"
        refused = _refused_extra_fields(extra, kind, scope_actual)
        if refused:
            raise ValueError(format_lines(refused, node_type))
        if type(self).zarr_scope_strict:
            unclaimed = [
                *unclaimed_extension_points,
                *_unclaimed_extra_fields(extra, kind, scope_actual),
            ]
            if unclaimed:
                msg = (
                    "Names the scope leaves unjudged, refused because this class "
                    "sets `zarr_scope_strict`:\n" + "\n".join(unclaimed)
                )
                raise ValueError(msg)


def parse_dtype_v3(
    dtype: npt.DTypeLike | Mapping[str, object],
    *,
    scope: ZarrV3Context | None = None,
) -> Mapping[str, object] | str:
    """
    Convert a numpy data type declaration to its Zarr V3 metadata form.

    Only numpy data types are parsed: a numpy data type maps to its Zarr V3
    name, or, for ``datetime64`` / ``timedelta64``, to the named configuration
    object the Zarr V3 form of those data types requires. A declaration
    already in Zarr V3 form — a data type name or a named configuration
    mapping — needs no parsing and passes through unchanged: like every other
    field, it is judged against the definitions in `scope` (``zarr-metadata``'s
    ``CORE_AND_EXTENSIONS`` when `scope` is `None`) when the whole metadata
    document is validated. One guard runs here instead of there: a string
    that names no data type in the scope but that numpy reads as a data type
    (``"i8"``, ``"float"``, ``">f4"``) is ambiguous — far more likely a numpy
    spelling than an unknown extension — and is refused, with the Zarr V3
    spelling suggested when one exists. The spellings zarr-python has stored
    for the variable-length data types (``"str"``, ``"variable_length_bytes"``)
    are recognized although no scope claims them, so documents zarr-python
    wrote pass through.
    """
    if isinstance(dtype, Mapping):
        return dtype
    if isinstance(dtype, str):
        if dtype in _VARIABLE_LENGTH_DATA_TYPES:
            return dtype
        scope_actual = CORE_AND_EXTENSIONS if scope is None else scope
        if scope_actual.claimant(DataTypeDefinition, dtype) is not None:
            return dtype
        try:
            as_numpy = np.dtype(dtype)
        except TypeError:
            # not a numpy spelling either: an unknown extension data type
            # name, passed through for document validation to judge in scope
            return dtype
        try:
            suggestion = parse_dtype_v3(as_numpy)
        except ValueError:
            msg = (
                f"Data type {dtype!r} names no data type in the scope, and numpy reads it "
                f"as {as_numpy!r}, which has no Zarr V3 form: pass a data type the scope "
                "defines."
            )
            raise ValueError(msg) from None
        msg = (
            f"Data type {dtype!r} names no data type in the scope, but numpy reads it as "
            f"{as_numpy!r}, whose Zarr V3 form is {suggestion!r}: pass that instead if it "
            "is what you meant."
        )
        raise ValueError(msg)
    try:
        parsed = np.dtype(dtype)
    except TypeError as e:
        # raise ValueError so pydantic reports this as a validation error
        # instead of letting numpy's TypeError escape the validator
        raise ValueError(f"Unsupported dtype: {dtype!r}") from e
    if parsed.kind in ("M", "m"):
        # numpy time dtypes take the named-configuration form in Zarr V3. These are
        # handled outside the match statement because numpy's stubs reject the
        # DateTime64DType / TimeDelta64DType classes in class patterns.
        unit, scale_factor = np.datetime_data(
            cast("np.dtype[np.datetime64 | np.timedelta64]", parsed)
        )
        name = "numpy.datetime64" if parsed.kind == "M" else "numpy.timedelta64"
        return {"name": name, "configuration": {"unit": unit, "scale_factor": scale_factor}}
    else:
        match parsed:
            case np.dtypes.BoolDType():
                return "bool"
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
            case np.dtypes.Float64DType():
                return "float64"
            case np.dtypes.Complex64DType():
                return "complex64"
            case np.dtypes.Complex128DType():
                return "complex128"
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")


def canonicalize_fill_value(fill_value: object) -> object:
    """
    Normalize a fill value to the form the Zarr V3 spec writes in JSON.

    Numpy scalars become the equivalent Python scalars, and non-finite floats
    become the string forms the spec requires (``"NaN"``, ``"Infinity"``,
    ``"-Infinity"``), since a metadata document holds only finite JSON numbers.
    Sequences (complex pairs, raw byte values) are normalized element-wise.
    Everything else passes through unchanged.
    """
    if isinstance(fill_value, np.generic):
        fill_value = fill_value.item()
    if isinstance(fill_value, float) and not math.isfinite(fill_value):
        if math.isnan(fill_value):
            return "NaN"
        return "Infinity" if fill_value > 0 else "-Infinity"
    if isinstance(fill_value, (tuple, list)):
        return tuple(canonicalize_fill_value(item) for item in fill_value)
    return fill_value


def _numpy_dtype_from_v3(data_type: str | Mapping[str, object]) -> np.dtype[Any] | None:
    """
    Best-effort inverse of `parse_dtype_v3`: the numpy data type equivalent to
    a Zarr V3 data type declaration, or `None` when there is none (an unknown
    extension data type).
    """
    if isinstance(data_type, str):
        try:
            candidate = np.dtype(data_type)
        except TypeError:
            return None
        try:
            # only names whose numpy reading maps back to the same Zarr V3
            # name are equivalents; np.dtype accepts spellings (the V3
            # variable-length "bytes", say) that mean something else entirely
            if parse_dtype_v3(candidate) != data_type:
                return None
        except ValueError:
            return None
        return candidate
    name = data_type.get("name")
    config = data_type.get("configuration")
    if name in ("numpy.datetime64", "numpy.timedelta64") and isinstance(config, Mapping):
        unit = config.get("unit")
        scale = config.get("scale_factor", 1)
        code = "M8" if name == "numpy.datetime64" else "m8"
        try:
            return np.dtype(f"{code}[{unit}]" if scale == 1 else f"{code}[{scale}{unit}]")
        except TypeError:
            return None
    return None


_SINGLE_BYTE_DATA_TYPES = ("bool", "int8", "uint8")

# Derived defaults for the variable-length data types, which have no numpy
# equivalent: fill value and codec pipeline, zarrs-style. zarr-python spells
# the variable-length bytes data type "variable_length_bytes" (its V3
# specification is still pending); "bytes" is the name zarrs and the
# zarr-extensions drafts use. Both derive the same defaults. "str" is the
# spelling zarr-python 3.0 stored for the variable-length string data type
# (renamed "string" in 3.1), so documents it wrote still hold it; membership
# here also exempts a name from the ambiguous-numpy-spelling guard in
# `parse_dtype_v3` (numpy reads "str" as a zero-length unicode type).
_VARIABLE_LENGTH_DATA_TYPES: Mapping[str, tuple[str, tuple[CodecLike, ...]]] = {
    "string": ("", ({"name": "vlen-utf8"},)),
    "str": ("", ({"name": "vlen-utf8"},)),
    "bytes": ("", ({"name": "vlen-bytes"},)),
    "variable_length_bytes": ("", ({"name": "vlen-bytes"},)),
}
# chunk guessing needs an element size; variable-length elements have none,
# so use the size of a pointer, as numpy object arrays do
_VARIABLE_LENGTH_NOMINAL_ITEM_SIZE = 8


DTypeLike = str | AnyNamedConfig
"""
A data type declaration in Zarr V3 form (name or named configuration).
`ArraySpec` accepts numpy data types as input and converts them with
`parse_dtype_v3` in a model validator, so the scope the conversion reads
in can come from the class's `zarr_scope` as well as the validation context.
"""


def _unclaimed_extension_point_names(
    document: Mapping[str, object], scope: ZarrV3Context
) -> list[str]:
    """
    The extension-point names in a dumped array metadata document that nothing
    in `scope` claims, each as a ``<loc>: <message>`` line. These are the
    names document validation leaves unjudged; a strict class refuses them
    instead. Codec pipelines are walked through a sharding codec's inner and
    index pipelines.
    """
    problems: list[str] = []

    def check(kind: type[Any], label: str, name: object, loc: str) -> None:
        if isinstance(name, str) and scope.claimant(kind, name) is None:
            problems.append(f"{loc}: {name!r} names no {label} in the scope")

    def walk_codecs(codecs: object, loc: str) -> None:
        if isinstance(codecs, str) or not isinstance(codecs, Sequence):
            return
        for index, codec in enumerate(codecs):
            if not isinstance(codec, Mapping):
                continue
            here = f"{loc}.{index}"
            check(CodecDefinition, "codec", codec.get("name"), f"{here}.name")
            config = codec.get("configuration")
            if codec.get("name") == "sharding_indexed" and isinstance(config, Mapping):
                walk_codecs(config.get("codecs"), f"{here}.configuration.codecs")
                walk_codecs(config.get("index_codecs"), f"{here}.configuration.index_codecs")

    data_type = document.get("data_type")
    if isinstance(data_type, Mapping):
        check(DataTypeDefinition, "data type", data_type.get("name"), "data_type.name")
    else:
        check(DataTypeDefinition, "data type", data_type, "data_type")
    for field, kind, label in (
        ("chunk_grid", ChunkGridDefinition, "chunk grid"),
        ("chunk_key_encoding", ChunkKeyEncodingDefinition, "chunk key encoding"),
    ):
        value = document.get(field)
        if isinstance(value, Mapping):
            check(kind, label, value.get("name"), f"{field}.name")
    walk_codecs(document.get("codecs"), "codecs")
    transformers = document.get("storage_transformers")
    if isinstance(transformers, Sequence) and not isinstance(transformers, str):
        for index, transformer in enumerate(transformers):
            if isinstance(transformer, Mapping):
                check(
                    StorageTransformerDefinition,
                    "storage transformer",
                    transformer.get("name"),
                    f"storage_transformers.{index}.name",
                )
    return problems


type ExtraFieldKind = type[ArrayExtraFieldDefinition[Any] | GroupExtraFieldDefinition[Any]]


def _refused_extra_fields(
    extra_fields: Mapping[str, object], kind: ExtraFieldKind, scope: ZarrV3Context
) -> list[str]:
    """
    The extra fields of a node document that the Zarr V3 spec's rules for
    additional fields refuse in `scope`, each as a ``<loc>: <message>`` line.

    An extra field whose key a definition in `scope` claims is one the reader
    recognizes: that definition judges it when the whole document is read, so
    it is skipped here. Every other extra field is one the reader does not
    recognize, which the spec admits only as an extension object (a ``name``
    that is an extension name, an optional ``configuration`` object, and no
    other members) that explicitly sets ``"must_understand": false``: "An
    implementation MUST fail to open Zarr groups or arrays if any metadata
    fields are present which (a) the implementation does not recognize and (b)
    are not explicitly set to ``"must_understand": false``." The envelope is
    checked by the same rule `zarr-metadata` applies to every other extension
    object.
    """
    problems: list[str] = []
    for key, value in extra_fields.items():
        if scope.claimant(kind, key) is not None:
            continue
        if not isinstance(value, Mapping):
            problems.append(
                f"{key}: not a field of this spec, and not an extra field the spec permits: "
                'an extra field is an extension object with "must_understand": false, '
                f"got {value!r}"
            )
            continue
        if value.get("must_understand") is not False:
            problems.append(
                f"{key}: no {kind.label} in the scope recognizes this field, and it does not "
                'set "must_understand": false, so the spec requires a reader to refuse it'
            )
            continue
        for problem in envelope_problems(value, allow_must_understand_false=True):
            loc = ".".join(str(part) for part in (key, *problem.loc))
            problems.append(f"{loc}: {problem.message} (an extra field is an extension object)")
    return problems


def _unclaimed_extra_fields(
    extra_fields: Mapping[str, object], kind: ExtraFieldKind, scope: ZarrV3Context
) -> list[str]:
    """
    The extra fields of a node document whose key nothing in `scope` claims,
    each as a ``<loc>: <message>`` line: the extra fields a reader in `scope`
    does not recognize, which a strict class refuses even when they set
    ``"must_understand": false``.
    """
    return [
        f"{key}: no {kind.label} in the scope claims {key!r}"
        for key in extra_fields
        if scope.claimant(kind, key) is None
    ]


def reject_bare_codec(data: object) -> object:
    """
    Refuse a single codec where a pipeline is expected: iterating a lone
    mapping or string would validate its keys or characters as codec names.
    """
    if isinstance(data, (str, Mapping)):
        msg = f"`codecs` takes a sequence of codecs; got a single codec ({data!r}). Wrap it in a sequence."
        # pydantic turns ValueError into a validation error; TypeError would escape raw
        raise ValueError(msg)  # noqa: TRY004
    return data


CodecTuple = Annotated[
    tuple[CodecLike, ...], BeforeValidator(ensure_multiple), BeforeValidator(reject_bare_codec)
]


class ArraySpecOverrides(TypedDict, total=False):
    """
    The fields `ArraySpec.from_shape_and_data_type` may override: every
    `ArraySpec` field except the two the method takes positionally.
    """

    attributes: TBaseAttr
    chunk_grid: RegularChunking
    chunk_key_encoding: DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata
    fill_value: FillValue
    codecs: Sequence[CodecLike]
    storage_transformers: Sequence[AnyNamedConfig]
    dimension_names: Sequence[str | None] | None
    extra_fields: Mapping[str, ZarrV3ExtensionField]


class ArraySpecUpdate(ArraySpecOverrides, total=False):
    """The fields an `ArraySpec.update` call may replace."""

    shape: tuple[int, ...]
    data_type: npt.DTypeLike | Mapping[str, object]


class BaseGroupSpecUpdate(TypedDict, total=False):
    """The fields a `BaseGroupSpec.update` call may replace."""

    attributes: TBaseAttr
    extra_fields: Mapping[str, ZarrV3ExtensionField]


class GroupSpecUpdate(TypedDict, total=False):
    """The fields a `GroupSpec.update` call may replace."""

    attributes: TBaseAttr
    members: Mapping[str, TBaseItem]
    extra_fields: Mapping[str, ZarrV3ExtensionField]


class ArraySpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 3 Array.

    Attributes
    ----------

    node_type: Literal['array']
        The node type. Must be the string 'array'.
    attributes: TAttr, default = {}
        User-defined metadata associated with this array. The default is the
        empty dict, matching the Zarr V3 spec, where the `attributes` key is
        optional; the default is validated against `TAttr`, so a model whose
        attributes type does not accept the empty dict still requires them.
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
        The sequence of codecs for this array.
    storage_transformers: Optional[Sequence[NamedConfig]]
        An optional sequence of `NamedConfig` objects that define the storage
        transformers for this array.
    dimension_names: Optional[Sequence[str]]
        An optional sequence of strings that gives names to each axis of the array.
    """

    node_type: Literal["array"] = "array"
    attributes: TAttr = Field(default={}, validate_default=True)
    shape: tuple[int, ...]
    data_type: DTypeLike
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata
    fill_value: Annotated[FillValue, BeforeValidator(canonicalize_fill_value)]
    codecs: CodecTuple
    storage_transformers: tuple[AnyNamedConfig, ...] = ()
    dimension_names: tuple[str | None, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_data_type(cls, data: object, info: ValidationInfo) -> object:
        """
        Convert a numpy `data_type` declaration to its Zarr V3 form via
        `parse_dtype_v3`, in the scope this validation reads in. A model
        validator rather than field metadata so the class's `zarr_scope`
        can serve as the scope when the validation context carries none
        (a field validator cannot see the class).
        """
        if not isinstance(data, Mapping) or "data_type" not in data:
            return data
        return {
            **data,
            "data_type": parse_dtype_v3(data["data_type"], scope=cls._scope(info.context)),
        }

    @model_validator(mode="before")
    @classmethod
    def canonicalize_codecs(cls, data: object) -> object:
        """
        Normalize codec spellings the way zarr-python stores them, so a spec
        constructed with either spelling compares equal — to one constructed
        with the other, and to what `from_zarr` reads back. Two rules: an
        empty `configuration` mapping is dropped from any codec, and the
        `bytes` codec's `endian` configuration, a no-op for a single-byte
        data type, is dropped for those data types. Both reach inside a
        sharding codec's inner pipeline (same data type); its index codecs,
        which encode the uint64 index, keep their endianness.
        (`zarr-metadata`'s canonical document model already equates the
        empty-configuration spellings, but not the endian ones; `==` on
        specs compares fields literally, hence the normalization here.)
        """
        if not isinstance(data, Mapping) or "codecs" not in data:
            return data
        codecs = data["codecs"]
        if isinstance(codecs, (str, Mapping)) or not isinstance(codecs, Sequence):
            # a lone codec is refused by the field validator; leave it alone
            return data
        try:
            parsed = parse_dtype_v3(data["data_type"]) if "data_type" in data else None
        except (ValueError, TypeError):
            parsed = None
        drop_endian = parsed in _SINGLE_BYTE_DATA_TYPES

        def normalize(codec: object, *, drop_endian: bool) -> object:
            if not isinstance(codec, Mapping):
                return codec
            name = codec.get("name")
            config = codec.get("configuration")
            if isinstance(config, Mapping) and not config:
                return {key: value for key, value in codec.items() if key != "configuration"}
            if (
                drop_endian
                and name == "bytes"
                and isinstance(config, Mapping)
                and set(config) == {"endian"}
            ):
                return {"name": "bytes"}
            if name == "sharding_indexed" and isinstance(config, Mapping):
                new_config = dict(config)
                inner = config.get("codecs")
                if isinstance(inner, Sequence) and not isinstance(inner, str):
                    new_config["codecs"] = tuple(
                        normalize(inner_codec, drop_endian=drop_endian) for inner_codec in inner
                    )
                index = config.get("index_codecs")
                if isinstance(index, Sequence) and not isinstance(index, str):
                    new_config["index_codecs"] = tuple(
                        normalize(index_codec, drop_endian=False) for index_codec in index
                    )
                return {**codec, "configuration": new_config}
            return codec

        return {
            **data,
            "codecs": tuple(normalize(codec, drop_endian=drop_endian) for codec in codecs),
        }

    @model_validator(mode="after")
    def validate_dimension_names(self) -> Self:
        if self.dimension_names is not None and len(self.dimension_names) != len(self.shape):
            msg = (
                "Invalid `dimension names` attribute. "
                f"Length of dimension names ({len(self.dimension_names)}), "
                f"does not match number of array dimensions ({len(self.shape)})."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_metadata_document(self, info: ValidationInfo) -> Self:
        """
        Validate this spec as a Zarr V3 array metadata document, via `zarr-metadata`.

        The document assembled from this spec's fields is read in a scope:
        the fill value is judged against the data type, the chunk grid against
        the shape, and the codecs as a pipeline, each extension point by the
        definition that claims its name. A name nothing in the scope claims is
        left unjudged, so unknown extensions pass; a claimed name with an
        invalid configuration is refused. The scope comes from the pydantic
        validation context (a `zarr_metadata` `ZarrV3Context`, directly or
        under the key `pydantic_zarr.v3.CONTEXT_KEY`), falling back to this
        class's `zarr_scope`, then to `zarr_metadata`'s `CORE_AND_EXTENSIONS`.

        Extra fields (members the spec does not define) are judged in the
        same scope: one whose key a definition in the scope claims, by that
        definition; any other, by the spec's rule for fields a reader does
        not recognize (an extension object with ``"must_understand": false``).

        On a class with `zarr_scope_strict` set, names the scope leaves
        unjudged are refused instead: the document must be built from
        extension-point names and extra-field keys the operative scope claims.

        The `attributes` field is not part of this check: its type and
        contents are governed by this class's `TAttr` parameter alone.
        """
        scope = self._scope(info.context)
        self._canonical_document(context=scope)
        unclaimed: list[str] = []
        if type(self).zarr_scope_strict:
            unclaimed = _unclaimed_extension_point_names(
                self.model_dump(), CORE_AND_EXTENSIONS if scope is None else scope
            )
        self._refuse_unjudged(ArrayExtraFieldDefinition, scope, unclaimed)
        return self

    def _canonical_document(
        self, *, context: ZarrV3Context | None = None
    ) -> ZarrV3ArrayMetadataModel:
        """
        This spec, attributes excluded, as a validated `zarr-metadata` document model.

        With no `context`, the document is read in the class's `zarr_scope`
        (the readers' default scope when that is `None` too), so a bare call
        — `like`'s canonical comparison — judges by the class's own scope.

        Raises `ValueError` listing every problem when the document is invalid.
        """
        if context is None:
            context = type(self).zarr_scope
        document = self.model_dump()
        document["attributes"] = {}
        try:
            return ZarrV3ArrayMetadataModel.from_json(document, context=context)
        except MetadataValidationError as e:
            raise ValueError(format_problems(e)) from e

    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """
        Omit `dimension_names` when it is `None`: the Zarr V3 spec allows the
        key to be absent, not null. This is a model serializer rather than a
        `model_dump` override so the omission also applies to arrays nested as
        members in a group's dump, and to JSON serialization. Like every
        spec's, it writes the extra fields as top-level members.
        """
        document = flatten_extra_fields(handler(self))
        if document.get("dimension_names") is None:
            document.pop("dimension_names", None)
        return document

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[Any] | zarr.Array,
        *,
        # TBaseAttr, not TAttr: covariant parameters may not appear in
        # argument positions. Validation enforces the class's attribute type.
        attributes: Literal["auto"] | TBaseAttr = "auto",
        chunk_grid: Literal["auto"] | AnyNamedConfig = "auto",
        chunk_key_encoding: Literal["auto"] | AnyNamedConfig = "auto",
        fill_value: Literal["auto"] | FillValue = "auto",
        codecs: Literal["auto"] | Sequence[CodecLike] = "auto",
        storage_transformers: Literal["auto"] | Sequence[AnyNamedConfig] = "auto",
        dimension_names: Literal["auto"] | Sequence[str | None] = "auto",
    ) -> Self:
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array :
            object that conforms to the numpy array API, or a `zarr.Array`.
            A `zarr.Array` carries its complete metadata document, which is
            read whole (via `from_zarr`), so chunk grid, chunk key encoding,
            fill value, codecs and attributes are all taken from the array;
            for other objects they are probed from same-named attributes.
            Keyword arguments other than `"auto"` override what the array
            provides either way.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """
        if (zarr_mod := sys.modules.get("zarr")) is not None and isinstance(array, zarr_mod.Array):
            spec = cls.from_zarr(array)
            changes: dict[str, object] = {}
            if attributes != "auto":
                changes["attributes"] = attributes
            if chunk_grid != "auto":
                changes["chunk_grid"] = chunk_grid
            if chunk_key_encoding != "auto":
                changes["chunk_key_encoding"] = chunk_key_encoding
            if fill_value != "auto":
                changes["fill_value"] = fill_value
            if codecs != "auto":
                changes["codecs"] = tuple(codecs)
            if storage_transformers != "auto":
                changes["storage_transformers"] = tuple(storage_transformers)
            if dimension_names != "auto":
                changes["dimension_names"] = dimension_names
            if changes:
                return cls.model_validate({**spec.model_dump(), **changes})
            return spec

        if attributes == "auto":
            attributes_actual = cast("TAttr", auto_attributes(array))
        else:
            # the constructor validates this against the class's TAttr
            attributes_actual = cast("TAttr", attributes)

        if chunk_grid == "auto":
            chunk_grid_actual = auto_chunk_grid(array)
        else:
            chunk_grid_actual = chunk_grid

        chunk_key_actual: AnyNamedConfig
        if chunk_key_encoding == "auto":
            chunk_key_actual = {"name": "default", "configuration": {"separator": "/"}}
        else:
            chunk_key_actual = chunk_key_encoding

        if fill_value == "auto":
            fill_value_actual = auto_fill_value(array)
        else:
            fill_value_actual = fill_value

        codecs_actual: tuple[CodecLike, ...]
        if codecs == "auto":
            codecs_actual = auto_codecs(array)
        else:
            codecs_actual = tuple(codecs)
        storage_transformers_actual: Sequence[AnyNamedConfig]
        if storage_transformers == "auto":
            storage_transformers_actual = auto_storage_transformers(array)
        else:
            storage_transformers_actual = storage_transformers

        dimension_names_actual: Sequence[str | None] | None
        if dimension_names == "auto":
            dimension_names_actual = auto_dimension_names(array)
        else:
            dimension_names_actual = dimension_names

        return cls(
            shape=array.shape,
            data_type=parse_dtype_v3(array.dtype),
            chunk_grid=chunk_grid_actual,
            attributes=attributes_actual,
            chunk_key_encoding=chunk_key_actual,
            fill_value=fill_value_actual,
            codecs=tuple(codecs_actual),
            storage_transformers=tuple(storage_transformers_actual),
            dimension_names=dimension_names_actual,
        )

    @classmethod
    def from_shape_and_data_type(
        cls,
        shape: Sequence[int],
        data_type: npt.DTypeLike | Mapping[str, object],
        **overrides: Unpack[ArraySpecOverrides],
    ) -> Self:
        """
        Create an `ArraySpec` from a shape and a data type alone: the
        metadata-only counterpart to `from_array`, for when there is no array
        object to describe.

        `overrides` takes the fields an `update` call takes, minus the two
        this method takes positionally. Every field absent from it gets a
        spec default or one derived from the data type: a size-guessed regular chunk grid (the same heuristic
        zarr-python applies when no chunks are given), the `default` chunk key
        encoding with the `/` separator, and the fill value and codecs
        `from_array` derives (`"NaN"` / `0` / `False` / NaT fills; a `bytes`
        codec, carrying explicit endianness for multi-byte data types, or the
        matching variable-length codec and empty fill for the `string` and
        `bytes` data types; compression is not assumed). Everything lands in
        one validation pass,
        so a subclass that narrows a field (required attributes, say) takes
        the narrowed values through `overrides`. A data type name that is not
        a Zarr V3 name but that numpy reads as some other type (`"i8"`,
        `"float"`) is refused as a likely mistake, with the V3 spelling
        suggested. For a data type with no
        numpy equivalent (an unknown extension data type) no fill value,
        codecs or chunk guess can be derived: pass `fill_value` and `codecs`
        there, and the chunk grid defaults to one whole-array chunk.

        Examples
        --------
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> ArraySpec.from_shape_and_data_type((10, 10), "uint16", dimension_names=("y", "x"))
        ArraySpec(zarr_format=3, node_type='array', attributes={}, shape=(10, 10), data_type='uint16', chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10, 10)}}, chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}}, fill_value=0, codecs=({'name': 'bytes', 'configuration': {'endian': 'little'}},), storage_transformers=(), dimension_names=('y', 'x'))
        """
        parsed_data_type = parse_dtype_v3(data_type, scope=cls.zarr_scope)
        np_dtype = _numpy_dtype_from_v3(parsed_data_type)
        variable_length = (
            _VARIABLE_LENGTH_DATA_TYPES.get(parsed_data_type)
            if np_dtype is None and isinstance(parsed_data_type, str)
            else None
        )
        document: dict[str, object] = {
            "shape": tuple(shape),
            "data_type": parsed_data_type,
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        }
        if "chunk_grid" not in overrides:
            if np_dtype is not None:
                item_size = np_dtype.itemsize
            elif variable_length is not None:
                item_size = _VARIABLE_LENGTH_NOMINAL_ITEM_SIZE
            else:
                item_size = None
            chunk_shape = (
                guess_chunks(tuple(shape), item_size) if item_size is not None else tuple(shape)
            )
            document["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": chunk_shape},
            }
        underived = tuple(name for name in ("fill_value", "codecs") if name not in overrides)
        if underived:
            if variable_length is not None:
                fill_default, codecs_default = variable_length
                if "fill_value" in underived:
                    document["fill_value"] = fill_default
                if "codecs" in underived:
                    document["codecs"] = codecs_default
            elif np_dtype is None:
                msg = (
                    f"Data type {parsed_data_type!r} has no numpy equivalent, so no default "
                    f"can be derived for it: pass {' and '.join(underived)} explicitly."
                )
                raise ValueError(msg)
            else:
                stand_in = SimpleNamespace(dtype=np_dtype)
                if "fill_value" in underived:
                    document["fill_value"] = auto_fill_value(stand_in)
                if "codecs" in underived:
                    document["codecs"] = auto_codecs(stand_in)
        return cls.model_validate({**document, **overrides})

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
        >>> import zarr.storage
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> store = zarr.storage.MemoryStore()
        >>> x = zarr.create(store=store, shape=(10, 10), zarr_format=3)
        >>> ArraySpec.from_zarr(x)
        ArraySpec(zarr_format=3, node_type='array', attributes={}, shape=(10, 10), data_type='float64', chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10, 10)}}, chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}}, fill_value=0.0, codecs=({'name': 'bytes', 'configuration': {'endian': 'little'}}, {'name': 'zstd', 'configuration': {'level': 0, 'checksum': False}}), storage_transformers=(), dimension_names=None)

        """
        try:
            from zarr.core.metadata import ArrayV3Metadata
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        meta_json: Mapping[str, object]

        if not isinstance(array.metadata, ArrayV3Metadata):
            raise ValueError("Only zarr v3 arrays are supported")  # noqa: TRY004
        if Version(version("zarr")) < Version("3.1.0"):
            # this class was removed from zarr python 3.1.0
            from zarr.core.metadata.v3 import V3JsonEncoder  # type: ignore[attr-defined]

            meta_json = tuplify_json(
                json.loads(json.dumps(array.metadata.to_dict(), cls=V3JsonEncoder))
            )
        else:
            meta_json = tuplify_json(array.metadata.to_dict())
        meta_any = cast("dict[str, Any]", meta_json)
        # zarr-python 3.1 and later keep a document's extra fields, and
        # write them as top-level keys in `to_dict`
        extra_fields = {
            key: value for key, value in meta_any.items() if key not in cls.model_fields
        }
        return cls(
            attributes=meta_any["attributes"],
            shape=array.shape,
            data_type=meta_any["data_type"],
            chunk_grid=meta_any["chunk_grid"],
            chunk_key_encoding=meta_any["chunk_key_encoding"],
            fill_value=meta_any["fill_value"],
            codecs=meta_any["codecs"],
            storage_transformers=meta_any["storage_transformers"],
            dimension_names=meta_any.get("dimension_names", None),
            extra_fields=extra_fields,
        )

    def to_zarr(
        self,
        store: Store,
        path: str,
        *,
        overwrite: bool = False,
        config: ArrayConfigParams | None = None,
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
        config : ArrayConfigParams | None, default = None
            An instance of `ArrayConfigParams` that defines the runtime configuration for the array.

        Returns
        -------
        A zarr array that is structurally identical to the ArraySpec.
        This operation will create metadata documents in the store.
        """
        try:
            import zarr
            from zarr.core.array import Array, AsyncArray
            from zarr.core.metadata.v3 import ArrayV3Metadata
            from zarr.core.sync import sync
            from zarr.errors import ContainsArrayError, ContainsGroupError
            from zarr.storage._common import make_store_path
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        self._check_storable()
        store_path = sync(make_store_path(store, path=path))
        extant_node = maybe_node(store, path, zarr_format=3)
        if isinstance(extant_node, zarr.Array):
            if not self.like(extant_node) and not overwrite:
                raise ContainsArrayError(store, path)
            else:
                # If there's an existing array that is identical to the model, and overwrite is False,
                # we can just return that existing array.
                if not overwrite:
                    return extant_node
        if isinstance(extant_node, zarr.Group) and not overwrite:
            raise ContainsGroupError(store, path)

        meta: ArrayV3Metadata = ArrayV3Metadata.from_dict(self.model_dump())
        async_array = AsyncArray(metadata=meta, store_path=store_path, config=config)
        sync(async_array._save_metadata(meta))
        return Array(_async_array=async_array)

    def _check_storable(self) -> None:
        """
        Refuse, before anything is written, an array whose extra fields
        `to_zarr` cannot store: zarr-python keeps an array's extra fields
        from 3.1.
        """
        from zarr.core.metadata.v3 import ArrayV3Metadata

        if self.extra_fields and not _stores_extra_fields(ArrayV3Metadata):
            msg = (
                f"This array holds extra fields ({', '.join(map(repr, self.extra_fields))}), "
                f"which zarr-python {version('zarr')} cannot store; zarr-python 3.1 and "
                "later can."
            )
            raise ValueError(msg)

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

    def with_sharding_codec(
        self,
        subchunk_shape: Sequence[int],
        *,
        index_location: Literal["start", "end"] = "end",
    ) -> Self:
        """
        Return a copy of this spec sharded with the `sharding_indexed` codec:
        the existing chunk shape becomes the shard shape, the existing codec
        pipeline moves inside the shards as the subchunk pipeline, and the
        shard index takes the default encoding (little-endian `bytes` plus
        `crc32c`, at `index_location`). The subchunk shape must evenly divide
        the chunk shape, which document validation enforces. Applying this to
        an already-sharded spec nests the shards (the sharding codec is
        recursive); the new subchunk shape must then be divisible by the
        existing inner chunk shape, which document validation also enforces.
        """
        sharding_codec: AnyNamedConfig = {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": tuple(subchunk_shape),
                "codecs": self.codecs,
                "index_codecs": (
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ),
                "index_location": index_location,
            },
        }
        return self.update(codecs=(sharding_codec,))

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
            converted to `ArraySpec` first; one whose metadata does not validate as a spec (one
            holding an extra field the spec's rules refuse, say) is like no spec.
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
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> x = zarr.create((10,10), zarr_format=3)
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
            try:
                other_parsed = ArraySpec.from_zarr(other)
            except ValidationError:
                # a stored array that does not validate as a spec (one holding
                # an extra field the spec's rules refuse, say) is like no spec
                return False
        else:
            other_parsed = other  # type: ignore[assignment]

        if include is None and exclude is None:
            # Compare the metadata documents by meaning rather than spelling:
            # `zarr-metadata` canonicalizes what it interprets, so a fill value
            # written "NaN" equals one written "0x7fc00000", and a codec equals
            # itself with a no-op configuration key dropped. Attributes are not
            # interpreted, so they are compared separately, as plain fields.
            return self._canonical_document() == other_parsed._canonical_document() and model_like(
                self, other_parsed, include={"attributes"}
            )

        return model_like(self, other_parsed, include=include, exclude=exclude)


def _stores_extra_fields(metadata_class: type) -> bool:
    """
    Whether a zarr-python metadata class keeps a document's extra fields:
    ``ArrayV3Metadata`` does from zarr-python 3.1, ``GroupMetadata`` from 3.4.
    """
    return any(field.name == "extra_fields" for field in dataclasses.fields(metadata_class))


def _stored_group_extra_fields(group: zarr.Group) -> dict[str, object]:
    """
    The extra fields of a stored group's metadata document, as zarr-python
    read them (it keeps them from zarr-python 3.4), except
    ``consolidated_metadata``: zarr-python's record of the hierarchy below
    the group, which a spec reads as members instead.
    """
    document = group.metadata.to_dict()
    own = {"zarr_format", "node_type", "attributes", ZARR_V3_CONSOLIDATED_METADATA_KEY}
    return {key: tuplify_json(value) for key, value in document.items() if key not in own}


def _write_group_extra_fields(
    group: zarr.Group, extra_fields: Mapping[str, object] | None
) -> zarr.Group:
    """
    Rewrite a stored group's metadata document with `extra_fields` added,
    returning the group as rewritten. `BaseGroupSpec._check_storable` has
    confirmed that zarr-python can store them.
    """
    if not extra_fields:
        return group
    from zarr.core.group import AsyncGroup, Group, GroupMetadata
    from zarr.core.sync import sync

    metadata = GroupMetadata.from_dict(
        {**group.metadata.to_dict(), **json.loads(json.dumps(extra_fields))}
    )
    async_group = AsyncGroup(metadata=metadata, store_path=group.store_path)
    sync(async_group._save_metadata())
    return Group(async_group)


def ensure_member_names_v3(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Ensure that every key of a members mapping is a valid Zarr V3 node name.

    The node-name rules of the Zarr V3 spec are judged by `zarr-metadata`'s
    `validate_node_name_v3`: a name holds no ``"/"``, is not composed of
    periods alone, and does not start with the reserved ``"__"`` prefix.
    The empty string names the root node, so it is not a valid *member*
    name and is rejected here as well.
    """
    for key in data:
        if key == "":
            msg = 'Invalid member name "": the empty string names the root node'
            raise ValueError(msg)
        problems = validate_node_name_v3(key)
        if problems:
            msg = "Invalid member name: " + "; ".join(problem.message for problem in problems)
            raise ValueError(msg)
    return data


class BaseGroupSpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 3 Group's own metadata, without members.

    This is the shape of a group in a flattened hierarchy (`to_flat` emits it
    for every group), and of a group loaded without traversing its children.
    `GroupSpec` subclasses this model and adds `members`.

    Attributes
    ----------
    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr, default = {}
        The user-defined attributes of this group. The default is the empty
        dict, matching the Zarr V3 spec, where the `attributes` key is
        optional; the default is validated against `TAttr`, so a model whose
        attributes type does not accept the empty dict still requires them.
    """

    node_type: Literal["group"] = "group"
    attributes: TAttr = Field(default={}, validate_default=True)

    @model_validator(mode="after")
    def validate_metadata_document(self, info: ValidationInfo) -> Self:
        """
        Validate this spec's extra fields as part of a Zarr V3 group metadata
        document, via `zarr-metadata`.

        A group document's only members beyond the fields this class declares
        are its extra fields, so only a spec holding some has anything to
        read. The scope is resolved as for `ArraySpec`: an extra field whose
        key a definition in the scope claims is judged by that definition
        (``CORE`` claims ``consolidated_metadata``, whose contents are kept as
        written and not read as members); any other must be an extension
        object with ``"must_understand": false``. On a class with
        `zarr_scope_strict` set, an extra field nothing in the scope claims is
        refused. Attributes and members are not part of this check.
        """
        if not self.extra_fields:
            return self
        scope = self._scope(info.context)
        document = {"zarr_format": 3, "node_type": "group", "attributes": {}, **self.extra_fields}
        try:
            ZarrV3GroupMetadataModel.from_json(document, context=scope)
        except MetadataValidationError as e:
            raise ValueError(format_problems(e, "group")) from e
        self._refuse_unjudged(GroupExtraFieldDefinition, scope)
        return self

    def update(self, **changes: Unpack[BaseGroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        Unlike `model_copy(update=...)`, the result is validated, so an
        invalid change is refused. Passing `extra_fields` replaces the extra fields as a whole.
        """
        return self._updated(changes)

    def _own_metadata(self) -> AnyBaseGroupSpec:
        """
        This group's own metadata, as a `BaseGroupSpec`: its attributes and
        extra fields, without members. Constructed without validation, since
        this spec was validated already, possibly in a scope that a bare
        `BaseGroupSpec` would not read its extra fields in.
        """
        return BaseGroupSpec.model_construct(
            attributes=self.attributes, extra_fields=self.extra_fields
        )

    def _check_storable(self) -> None:
        """
        Refuse, before anything is written, a group whose extra fields
        `to_zarr` cannot store: ``consolidated_metadata``, which records the
        members of a hierarchy as it was consolidated (writing it as held
        could record members other than the ones written; reading and writing
        it is left to a dedicated consolidated-metadata API), and any extra
        field on a zarr-python too old to store one.
        """
        extra = self.extra_fields
        if ZARR_V3_CONSOLIDATED_METADATA_KEY in extra:
            msg = (
                f"This group holds `{ZARR_V3_CONSOLIDATED_METADATA_KEY}`, which `to_zarr` does "
                "not write: it lists the members of the hierarchy as it was consolidated, "
                "which need not be the members written. Drop it before writing (pass "
                "`update` the `extra_fields` without it), and consolidate the written "
                "hierarchy with `zarr.consolidate_metadata`."
            )
            raise ValueError(msg)
        if extra:
            from zarr.core.group import GroupMetadata

            if not _stores_extra_fields(GroupMetadata):
                msg = (
                    f"This group holds extra fields ({', '.join(map(repr, extra))}), which "
                    f"zarr-python {version('zarr')} cannot store; zarr-python 3.4 and later can."
                )
                raise ValueError(msg)

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> Self:
        """
        Create a `BaseGroupSpec` from a zarr group, without reading its members.

        The group's extra fields are read with it, except
        ``consolidated_metadata``: it is zarr-python's record of the
        hierarchy below the group, not part of the group's own metadata.
        """
        return cls.model_validate(
            {"attributes": group.attrs.asdict(), **_stored_group_extra_fields(group)}
        )

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

        self._check_storable()
        extant_node = maybe_node(store, path, zarr_format=3)
        if isinstance(extant_node, zarr.Group):
            if not self.like(extant_node):
                if not overwrite:
                    raise ContainsGroupError(store, path)
            elif not overwrite:
                return extant_node
        elif isinstance(extant_node, zarr.Array) and not overwrite:
            raise ContainsArrayError(store, path)
        else:
            zarr.create_group(store=store, overwrite=overwrite, path=path, zarr_format=3)

        result = zarr.group(store=store, path=path, overwrite=overwrite, zarr_format=3)
        result.attrs.put(self.model_dump()["attributes"])
        return _write_group_extra_fields(result, self.extra_fields)

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
        to its own metadata first. A `zarr.Group` whose metadata does not
        validate as a spec (one holding an extra field the spec's rules refuse,
        say) is like no spec.
        """
        other_parsed: AnyBaseGroupSpec
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Group):
            try:
                other_parsed = BaseGroupSpec.from_zarr(other)
            except ValidationError:
                # a stored group that does not validate as a spec is like no spec
                return False
        elif isinstance(other, GroupSpec):
            other_parsed = other._own_metadata()
        else:
            other_parsed = cast("AnyBaseGroupSpec", other)

        return model_like(self, other_parsed, include=include, exclude=exclude)


class GroupSpec(BaseGroupSpec[TAttr], Generic[TAttr, TItem]):
    """
    A model of a Zarr Version 3 Group.

    Attributes
    ----------

    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr
        The user-defined attributes of this group.
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
        Validate member names as Zarr V3 node names.

        This runs as a model validator rather than field metadata so that a
        subclass narrowing the `members` annotation (for example to a closed
        `TypedDict` of members) cannot drop it.
        """
        ensure_member_names_v3(self.members)
        return self

    @override
    def update(self, **changes: Unpack[GroupSpecUpdate]) -> Self:  # type: ignore[override]
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so `attributes` and `members`
        change together atomically. Unlike `model_copy(update=...)`, the
        result is validated. Passing `extra_fields` replaces the extra fields as a whole.
        """
        return self._updated(changes)

    def resolve_path(self, path: str) -> AnyArraySpec | AnyGroupSpec:
        """
        Return the node at a `/`-separated path of member names relative to
        this group (`"a"`, `"a/b/c"`; leading and trailing `/` are ignored).
        Metadata conventions that reference nodes by relative path (OME-Zarr
        dataset paths, GeoZarr `grid_mapping` and multiscale assets) resolve
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
    def from_flat(cls, data: Mapping[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation.

        The flattened hierarchy is a
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
        ```py
        from pydantic_zarr.v3 import BaseGroupSpec, GroupSpec, ArraySpec
        import numpy as np
        flat = {'': BaseGroupSpec(attributes={'foo': 10})}
        GroupSpec.from_flat(flat)
        # GroupSpec(zarr_format=3, node_type='group', attributes={'foo': 10}, members={})
        flat = {
            '': BaseGroupSpec(attributes={'foo': 10}),
            '/a': ArraySpec.from_array(np.arange(10))}
        GroupSpec.from_flat(flat)
        # GroupSpec(
        #    zarr_format=3,
        #    node_type='group',
        #    attributes={'foo': 10},
        #    members={
        #        'a': ArraySpec(
        #                zarr_format=3,
        #                node_type='array',
        #                attributes={},
        #                shape=(10,),
        #                data_type='int64',
        #                chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10,)}},
        #                chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        #                fill_value=0,
        #                codecs=(),
        #                storage_transformers=(),
        #                dimension_names=None)})
        ```
        """
        from_flated = from_flat_group(data)
        return cls.model_validate(from_flated.model_dump())

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

        >>> from pydantic_zarr.v3 import GroupSpec
        >>> g1 = GroupSpec(attributes={'foo': 'bar'})
        >>> g1.to_flat()
        {'': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
        >>> g1.to_flat(root_path='baz')
        {'baz': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
        >>> GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}).to_flat()
        {'': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'}), '/g1': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
        """
        return to_flat(self, root_path=root_path)

    @classmethod
    @override
    def from_zarr(cls, group: zarr.Group) -> Self:
        """
        Create a GroupSpec from a zarr group. Subgroups and arrays contained in the zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .members attribute of the parent
        GroupSpec. This occurs recursively, so the entire zarr hierarchy below a given
        group can be represented as a GroupSpec. Reading the entire hierarchy may be slow
        for large hierarchies on high-latency storage. To read part of a hierarchy, use
        `from_zarr_flat`; to read a group without its members, use `BaseGroupSpec.from_zarr`.

        Parameters
        ----------
        group : zarr.Group
            The Zarr group to model.

        Returns
        -------
        An instance of GroupSpec that represents the structure of the zarr hierarchy.
        """
        try:
            import zarr
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        metadata = {"attributes": group.attrs.asdict(), **_stored_group_extra_fields(group)}
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

        return cls.model_validate({**metadata, "members": members})

    @override
    def to_zarr(
        self, store: Store, path: str, *, overwrite: bool = False, **kwargs: Any
    ) -> zarr.Group:
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
        try:
            import zarr
            from zarr.errors import ContainsArrayError, ContainsGroupError
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        spec_dict = self.model_dump(exclude={"members": True})
        attrs = spec_dict.pop("attributes")
        _check_storable_tree(self)
        extant_node = maybe_node(store, path, zarr_format=3)
        if isinstance(extant_node, zarr.Group):
            if not self.like(extant_node):
                if not overwrite:
                    """
                    msg = (
                        f"A group already exists at path {path}. "
                        "That group is structurally dissimilar to the group you are trying to store. "
                        "Call `to_zarr` with `overwrite=True` to overwrite that group."
                    )
                    """
                    # TODO: use the above message when we fix the ContainsGroupError in zarr python
                    # To accept a proper message
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
            # TODO: use the above message when we fix the ContainsArrayError in zarr python
            raise ContainsArrayError(store, path)
        else:
            zarr.create_group(store=store, overwrite=overwrite, path=path, zarr_format=3)

        result = zarr.group(store=store, path=path, overwrite=overwrite, zarr_format=3)
        result.attrs.put(attrs)
        result = _write_group_extra_fields(result, self.extra_fields)
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
        Compare a `GroupSpec` to another `GroupSpec` or a `zarr.Group`.

        With no `include`/`exclude` filters, the groups' own metadata is
        compared as plain fields and each member is compared by that member's
        own `like`, so member arrays compare by the meaning of their metadata
        documents (a fill value written `"NaN"` equals one written
        `"0x7fc00000"`), exactly as `ArraySpec.like` judges them directly.
        A `BaseGroupSpec` — a group considered without members — compares as
        that projection: by the group's own metadata alone, whatever this
        side's members hold. A `zarr.Group` whose hierarchy does not validate
        as a spec is like no spec.
        With filters, models are converted to `dict` via the `model_dump`
        method of `pydantic.BaseModel` and compared with the `==` operator.

        Parameters
        ----------
        other : GroupSpec | zarr.Group
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
        >>> from pydantic_zarr.v3 import GroupSpec
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
            return self._own_metadata().like(other, include=include, exclude=exclude)
        other_parsed: AnyGroupSpec
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Group):
            try:
                other_parsed = GroupSpec.from_zarr(other)
            except ValidationError:
                # a stored hierarchy that does not validate as a spec is like no spec
                return False
        else:
            other_parsed = other  # type: ignore[assignment]

        if include is None and exclude is None:
            # Compare the groups' own metadata as plain fields and each member
            # by that member's own `like`, so member arrays compare by the
            # meaning of their metadata documents, exactly as `ArraySpec.like`
            # judges them directly.
            if not model_like(self, other_parsed, exclude={"members"}):
                return False
            if self.members.keys() != other_parsed.members.keys():
                return False
            for name, member in self.members.items():
                other_member = other_parsed.members[name]
                if isinstance(member, ArraySpec):
                    if not isinstance(other_member, ArraySpec) or not member.like(other_member):
                        return False
                elif isinstance(other_member, ArraySpec) or not member.like(other_member):
                    return False
            return True

        return model_like(self, other_parsed, include=include, exclude=exclude)


def _check_storable_tree(node: AnyArraySpec | AnyBaseGroupSpec, path: str = "") -> None:
    """
    Refuse, before anything is written, a hierarchy holding a node whose
    extra fields `to_zarr` cannot store, naming the node's path. A group's
    `to_zarr` writes itself before its members, so checking only the group
    would leave a partial hierarchy behind when a member is refused.
    """
    try:
        node._check_storable()
    except ValueError as e:
        msg = f"Cannot write the node at {path or '/'!r}: {e}"
        raise ValueError(msg) from e
    if isinstance(node, GroupSpec):
        for name, member in node.members.items():
            _check_storable_tree(member, f"{path}/{name}")


@overload
def from_zarr(element: zarr.Array) -> AnyArraySpec: ...


@overload
def from_zarr(element: zarr.Group) -> AnyGroupSpec: ...


def from_zarr(element: zarr.Array | zarr.Group) -> AnyArraySpec | AnyGroupSpec:
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.
    A group is read with its entire hierarchy; to read part of one, use `from_zarr_flat`.

    Parameters
    ----------
    element : a zarr Array or zarr Group

    Returns
    -------
    An instance of GroupSpec or ArraySpec that represents the
    structure of the zarr group or array.
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
    overwrite: bool = False,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: AnyGroupSpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Group: ...


def to_zarr(
    spec: AnyArraySpec | AnyGroupSpec,
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


def from_flat(
    data: Mapping[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec],
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
    ```py
    from pydantic_zarr.v3 import from_flat, ArraySpec
    import numpy as np
    tree = {'': ArraySpec.from_array(np.arange(10))}
    from_flat(tree) # special case of a Zarr array at the root of the hierarchy
    # ArraySpec(zarr_format=3, node_type='array', attributes={}, shape=(10,), ...)
    tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    from_flat(tree) # note that an implicit Group is created
    # GroupSpec(
    #    zarr_format=3,
    #    node_type='group',
    #    attributes={},
    #    members={'foo': ArraySpec(zarr_format=3, node_type='array', ...)})
    ```
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
    ```py
    from pydantic_zarr.v3 import from_flat_group, ArraySpec
    import numpy as np
    tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    from_flat_group(tree) # note that an implicit Group is created
    # GroupSpec(
    #    zarr_format=3,
    #    node_type='group',
    #    attributes={},
    #    members={'foo': ArraySpec(zarr_format=3, node_type='array', ...)})
    ```
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
                    f"Value at '{key}' is not a v3 ArraySpec or GroupSpec (got {type(value)=})"
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

    return GroupSpec(
        members={**member_groups, **member_arrays},
        attributes=root_node.attributes,
        extra_fields=root_node.extra_fields,
    )


def auto_attributes(data: object) -> Mapping[str, object]:
    if hasattr(data, "attributes") and isinstance(data.attributes, Mapping):
        return data.attributes
    return {}


def auto_chunk_grid(data: object) -> AnyNamedConfig:
    if hasattr(data, "chunk_shape"):
        return {"name": "regular", "configuration": {"chunk_shape": tuple(data.chunk_shape)}}
    elif hasattr(data, "shape"):
        return {"name": "regular", "configuration": {"chunk_shape": tuple(data.shape)}}
    raise ValueError("Cannot get chunk grid from object without .shape or .chunk_shape attribute")


def auto_chunk_key_encoding(data: object) -> AnyNamedConfig:
    if hasattr(data, "metadata") and hasattr(data.metadata, "chunk_key_encoding"):
        return data.metadata.to_dict()["chunk_key_encoding"]
    return {"name": "default", "configuration": {"separator": "/"}}


def auto_fill_value(data: object) -> FillValue:
    if hasattr(data, "fill_value"):
        return data.fill_value
    elif hasattr(data, "dtype"):
        kind = np.dtype(data.dtype).kind
        if kind == "b":
            return False
        elif kind in ["i", "u"]:
            return 0
        elif kind == "f":
            return "NaN"
        elif kind == "c":
            return ("NaN", "NaN")
        elif kind in ["M", "m"]:
            # NaT, as the Zarr V3 spec writes it: the minimum int64 value.
            return int(np.iinfo(np.int64).min)
        else:
            raise ValueError(f"Cannot determine default fill value for data type {kind}")
    raise ValueError("Cannot determine default data type for object without shape attribute.")


def auto_codecs(data: object) -> tuple[CodecLike, ...]:
    """
    Automatically create a tuple of codecs from an arbitrary python object.
    """
    if hasattr(data, "codecs"):
        # todo: type check
        return tuple(data.codecs)
    if hasattr(data, "dtype") and np.dtype(data.dtype).itemsize > 1:
        # the zarr v3 spec requires the ``endian`` configuration for multi-byte data types
        return ({"name": "bytes", "configuration": {"endian": "little"}},)
    return ({"name": "bytes"},)


def auto_storage_transformers(data: object) -> tuple[AnyNamedConfig, ...]:
    if hasattr(data, "storage_transformers"):
        return tuple(data.storage_transformers)
    return ()


def auto_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if hasattr(data, "metadata") and hasattr(data.metadata, "dimension_names"):
        if data.metadata.dimension_names is None:
            return None
        else:
            return tuple(data.metadata.dimension_names)
    return None


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

    >>> from pydantic_zarr.v3 import to_flat, GroupSpec
    >>> g1 = GroupSpec(attributes={'foo': 'bar'})
    >>> to_flat(g1)
    {'': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
    >>> to_flat(g1, root_path='baz')
    {'baz': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
    >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
    {'': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'}), '/g1': BaseGroupSpec(zarr_format=3, node_type='group', attributes={'foo': 'bar'})}
    """
    result = {}
    model_copy: AnyArraySpec | AnyBaseGroupSpec
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        model_copy = node.model_copy(deep=True)._own_metadata()
        if isinstance(node, GroupSpec):
            for name, value in node.members.items():
                result.update(to_flat(value, f"{root_path}/{name}"))

    result[root_path] = model_copy
    # sort by increasing key length
    return dict(sorted(result.items(), key=lambda v: len(v[0])))


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
