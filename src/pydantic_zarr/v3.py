from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    NotRequired,
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
    BeforeValidator,
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    model_serializer,
    model_validator,
)
from typing_extensions import TypedDict, TypeIs, TypeVar
from zarr_metadata import (
    BOOL_DATA_TYPE_NAME,
    BYTES_CODEC_NAME,
    BYTES_DATA_TYPE_NAME,
    CAST_VALUE_CODEC_NAME,
    COMPLEX64_DATA_TYPE_NAME,
    COMPLEX128_DATA_TYPE_NAME,
    CRC32C_CODEC_NAME,
    DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR,
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    FIXED_LENGTH_UTF32_DATA_TYPE_NAME,
    FLOAT16_DATA_TYPE_NAME,
    FLOAT32_DATA_TYPE_NAME,
    FLOAT64_DATA_TYPE_NAME,
    INT8_DATA_TYPE_NAME,
    INT16_DATA_TYPE_NAME,
    INT32_DATA_TYPE_NAME,
    INT64_DATA_TYPE_NAME,
    NUMPY_DATETIME64_DATA_TYPE_NAME,
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    REGULAR_CHUNK_GRID_NAME,
    SCALE_OFFSET_CODEC_NAME,
    SHARDING_INDEXED_CODEC_NAME,
    STRING_DATA_TYPE_NAME,
    TRANSPOSE_CODEC_NAME,
    UINT8_DATA_TYPE_NAME,
    UINT16_DATA_TYPE_NAME,
    UINT32_DATA_TYPE_NAME,
    UINT64_DATA_TYPE_NAME,
    ZARR_V3_ARRAY_METADATA_STORE_KEY,
    ZARR_V3_GROUP_METADATA_STORE_KEY,
    DefaultChunkKeyEncodingMetadata,
    JSONValue,
    MetadataValidationError,
    RegularChunkGridMetadata,
    ShardingIndexLocation,
    V2ChunkKeyEncodingMetadata,
    ZarrV3ArrayMetadataJSON,
    ZarrV3DeclarationJSON,
    ZarrV3ExtensionField,
    ZarrV3GroupMetadataJSON,
    ZarrV3NamedConfigJSON,
    ZarrV3NodeType,
)
from zarr_metadata.model import ZarrV3ArrayMetadata as ZarrV3ArrayMetadataModel
from zarr_metadata.model import ZarrV3GroupMetadata as ZarrV3GroupMetadataModel
from zarr_metadata.model import validate_node_name_v3
from zarr_metadata.typed_json import check as check_json
from zarr_metadata.v3 import is_zarr_v3_node_type
from zarr_metadata.v3.data_type import (
    NULL_TERMINATED_BYTES_DATA_TYPE_NAME,
    VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME,
    ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME,
    raw_bytes_dtype_name,
)
from zarr_metadata.v3.definition import (
    ArrayExtraFieldDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    GroupExtraFieldDefinition,
    StorageTransformerDefinition,
    storage_of,
)

from pydantic_zarr._metadata import CONTEXT_KEY as CONTEXT_KEY
from pydantic_zarr._metadata import (
    CORE_AND_EXTENSIONS_CONTEXT,
    format_lines,
    format_problems,
    scope_from_context,
)
from pydantic_zarr.core import (
    ExtraFieldsBase,
    IncEx,
    StoredNodes,
    as_dtype_like,
    as_shape,
    canonicalize_fill_value,
    copy_members,
    ensure_members_mapping,
    ensure_multiple,
    flatten_extra_fields,
    guess_chunks,
    model_like,
    read_attribute,
    resolve_member_path,
    split_store_key,
    stored_array,
    stored_flat,
    stored_root,
    tuplify_json,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from zarr_metadata.definition import Definition
    from zarr_metadata.v3.chunk_key_encoding.default import (
        DefaultChunkKeyEncodingObject,
    )
    from zarr_metadata.v3.definition import ZarrV3Context


# The BaseModel arm is SerializeAsAny so that an attribute model held by a
# spec whose attributes type is not narrowed (a bare ArraySpec / GroupSpec)
# serializes with the model's own fields; annotated as plain BaseModel it
# would serialize through the base schema, silently dumping {}.
type TBaseAttr = Mapping[str, object] | SerializeAsAny[BaseModel]
# BaseGroupSpec is not a member type: a GroupSpec is a complete hierarchy,
# and a group whose members are unknown is never part of one.
type TBaseItem = Union["GroupSpec", "ArraySpec"]
# The widest members mapping: any names, each member a GroupSpec or an ArraySpec.
type TBaseMembers = Mapping[str, TBaseItem]

# Aliases for specs of any specialization. Since the spec parameters are
# covariant (specs are frozen, so a narrowed spec IS-A wider one), these are
# ordinary wide types, not gradual-typing escapes: a value typed with one
# keeps checked, TBaseAttr/TBaseMembers-typed reads. The names remain the public
# spelling for heterogeneous APIs (`resolve_path`, `from_flat`, `like`,
# `is_array_spec`) and are now equal to the bare class names.
type AnyArraySpec = "ArraySpec[TBaseAttr]"
type AnyGroupSpec = "GroupSpec[TBaseAttr, TBaseMembers]"
type AnyBaseGroupSpec = "BaseGroupSpec[TBaseAttr]"

# Covariant because the specs are frozen: TAttr and TMembers appear only in
# read positions on instances, so ArraySpec[CFAttrs] is substitutable for
# ArraySpec[TBaseAttr], and TBaseMembers serves as both TMembers' bound and its
# default (the bare, recursive spelling of TBaseItem inside it also makes
# pydantic validate a bare GroupSpec's nested members as specs all the way
# down; tests/typing_conformance.py guards the narrowed-parameterization
# contract).
#
# TMembers is the whole members mapping, not the type of one member, so that a
# group whose member names are part of its schema is one parameterization:
# GroupSpec[Attrs, Members] with Members a closed TypedDict. A group that only
# constrains its members' type is GroupSpec[Attrs, Mapping[str, MyArraySpec]].
# Constructing classmethods take TBaseAttr instead of TAttr, since a covariant
# parameter may not appear in argument positions; the class's real attribute
# type is still enforced, by validation.
TAttr = TypeVar("TAttr", bound=TBaseAttr, default=TBaseAttr, covariant=True)
TMembers = TypeVar("TMembers", bound=TBaseMembers, default=TBaseMembers, covariant=True)
TDefault = TypeVar("TDefault")

NodeType = ZarrV3NodeType

type ZarrV3StoredDocumentJSON = ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON
"""Any document a Zarr V3 hierarchy stores: a node's `zarr.json`."""

# The documents a store holds for a node, keyed by store key. The keys are not
# identifiers, hence the functional syntax. The array type is closed, and the
# group type admits only its members' documents beyond its own, so each is
# assignable to a `Mapping[str, ...]` of its value types.
ZarrV3ArrayStoredJSON = TypedDict(
    "ZarrV3ArrayStoredJSON", {"zarr.json": ZarrV3ArrayMetadataJSON}, closed=True
)
"""A Zarr V3 array's metadata as a store holds it: its `zarr.json` document."""
ZarrV3GroupStoredJSON = TypedDict(
    "ZarrV3GroupStoredJSON",
    {"zarr.json": ZarrV3GroupMetadataJSON},
    extra_items=ZarrV3StoredDocumentJSON,
)
"""
A Zarr V3 group's metadata as a store holds it: its own `zarr.json` document,
and each member's at its metadata key: the member's hierarchy path relative to
the group, with no leading `/`, then `/zarr.json` (`"a/b/zarr.json"` for the
member at `/a/b`).
"""

FillValue = JSONValue
"""
The JSON form of a fill value, as a spec holds it. Which values are valid depends on the array's
data type, and is judged by that data type's definition when the whole
metadata document is validated; at the field level any JSON value is a
fill value. `zarr-metadata` exports per-data-type fill value types
(``Float32FillValue`` and so on) for callers that want the narrow forms.
"""

FillValueLike = FillValue | np.generic | complex
"""
A fill value as an input takes it (`ArraySpec.create_default`,
`ArraySpec.replace`, `ArraySpec.from_array`): its JSON form, or a numpy scalar
or a Python complex number, which validation writes in JSON form (non-finite
floats as `"NaN"`, `"Infinity"` or `"-Infinity"`, complex numbers as pairs).
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


CodecLike = str | AnyNamedConfig | ZarrV3NamedConfigJSON
"""
A codec declaration as an input takes it: an extension name, or a named
configuration object. `zarr_metadata`'s `ZarrV3NamedConfigJSON`, whose members
are read-only, keeps every typed codec object (`BytesCodecObject`,
`ZstdCodecObject`, ...) assignable, which the mutable `AnyNamedConfig` cannot,
since TypedDict keys are invariant. A spec holds every codec as an object
(`AnyNamedConfig`), since validation writes a bare name as ``{"name": ...}``.
"""

ChunkGridLike = RegularChunkGridMetadata | AnyNamedConfig | ZarrV3NamedConfigJSON
"""
A chunk grid declaration as an input takes it (`ArraySpec.create_default`,
`ArraySpec.replace`, `ArraySpec.from_array`): the regular grid the core spec
defines, typed exactly, or any other grid as a named configuration, judged by
its definition in the scope when the document is validated. The exact member
and the read-only `ZarrV3NamedConfigJSON` keep typed grid objects (such as
`zarr_metadata`'s `RegularChunkGridObject`) assignable, since TypedDict keys
are invariant. A spec holds the grid as an `AnyNamedConfig`.
"""

ChunkKeyEncodingLike = (
    DefaultChunkKeyEncodingMetadata
    | V2ChunkKeyEncodingMetadata
    | AnyNamedConfig
    | ZarrV3NamedConfigJSON
    | str
)
"""
A chunk key encoding declaration as an input takes it: the two the core spec
defines, typed exactly (as an object or a bare name), or any other as a named
configuration (typed objects through the read-only `ZarrV3NamedConfigJSON`) or a
bare name, judged by its definition in the scope when the document is
validated. A spec holds the encoding as an `AnyNamedConfig`.
"""


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
        operation (``replace``, ``create_default``, ``like``, ...),
        which re-validate as the class. An explicit validation context
        still wins, and ``None`` (the default) means ``zarr-metadata``'s
        ``CORE_AND_EXTENSIONS_CONTEXT``. A group's members are validated as their
        own classes, so a scoped tree narrows its member types to
        scope-bearing classes (an explicit context, which pydantic carries
        into nested validation, reaches the members either way).
    zarr_scope_strict: ClassVar[bool]
        When true, validation additionally refuses every extension-point
        name — data type, chunk grid, chunk key encoding, codec (a sharding
        codec's inner and index pipelines included) and storage transformer
        — that nothing in the operative scope claims, where the default is
        to leave such names unjudged. Together with `zarr_scope` this makes
        a class a conformance contract: ``zarr_scope = CORE_CONTEXT`` plus strict
        accepts exactly the documents built from names the core spec
        defines. The variable-length data types and their codecs are
        registered extensions, not core, so a strict core class refuses
        them. Enforced by
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
    (``CORE_CONTEXT`` claims a group's ``consolidated_metadata``). Any other extra
    field is one the reader does not recognize, which the spec admits only
    as an object that says ``"must_understand": false``, kept as written;
    every other unknown key is refused, which also catches a misspelled
    field name. Accepted extra fields are
    kept in the `extra_fields` mapping (see `ExtraFieldsBase`) and written
    back as top-level members wherever the document is written:
    ``model_dump``, ``model_dump_json``, ``replace``, ``like`` and
    ``to_stored_metadata_json``.
    """

    zarr_format: Literal[3] = 3
    zarr_scope: ClassVar[ZarrV3Context | None] = None
    zarr_scope_strict: ClassVar[bool] = False

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

    @classmethod
    def _scope(cls, context: object) -> ZarrV3Context | None:
        """
        The scope a validation of this class reads metadata in: the validation
        context's scope when it carries one, else this class's `zarr_scope`.
        `None` means the readers' default (``CORE_AND_EXTENSIONS_CONTEXT``).
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
        Refuse what a whole-document read in `scope` (``CORE_AND_EXTENSIONS_CONTEXT``
        when `None`) leaves unjudged and the spec or this class does not
        allow: the extra fields the spec's rule for fields a reader does not
        recognize refuses, and, on a class with `zarr_scope_strict` set, every
        extra field and extension-point name (`unclaimed_extension_points`)
        nothing in `scope` claims. Extra fields a definition in `scope` claims
        are judged by that definition when the whole document is read, not
        here.
        """
        extra = self.extra_fields
        scope_actual = CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
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
    name, or to the named configuration object the Zarr V3 form of some data
    types requires, spelled as zarr-python writes them: ``datetime64`` /
    ``timedelta64`` become ``numpy.datetime64`` / ``numpy.timedelta64``, a
    fixed-length unicode type (``<U5``) ``fixed_length_utf32``, a fixed-length
    bytes type (``S3``) ``null_terminated_bytes``, and numpy's variable-length
    ``StringDType`` ``string``. An unstructured void type (``V4``) becomes
    ``raw_bytes`` when `scope` claims that name (zarr-python's scope does), and
    the core spec's raw type (``r32``) otherwise. A declaration
    already in Zarr V3 form — a data type name or a named configuration
    mapping — needs no parsing and passes through unchanged: like every other
    field, it is judged against the definitions in `scope` (``zarr-metadata``'s
    ``CORE_AND_EXTENSIONS_CONTEXT`` when `scope` is `None`) when the whole metadata
    document is validated. One guard runs here instead of there: a string
    that names no data type in the scope but that numpy reads as a data type
    (``"i8"``, ``"float"``, ``">f4"``) is ambiguous — far more likely a numpy
    spelling than an unknown extension — and is refused, with the Zarr V3
    spelling suggested when one exists. The spellings zarr-python has stored
    for the variable-length data types are recognized in any scope, so
    documents zarr-python wrote pass through: ``"str"``, which no scope claims,
    and ``"variable_length_bytes"``, which only ``ZARR_PYTHON_CONTEXT`` claims.
    """
    if isinstance(dtype, Mapping):
        return dtype
    if isinstance(dtype, str):
        if dtype in _VARIABLE_LENGTH_DATA_TYPES:
            return dtype
        scope_actual = CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
        if scope_actual.claimant(DataTypeDefinition, dtype) is not None:
            return dtype
        try:
            as_numpy = np.dtype(dtype)
        except TypeError:
            # not a numpy spelling either: an unknown extension data type
            # name, passed through for document validation to judge in scope
            return dtype
        try:
            suggestion = parse_dtype_v3(as_numpy, scope=scope)
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
    # a zero-length string or bytes type (numpy's bare `str`, `"U"`, `"S"`) has
    # no Zarr V3 form, and neither has a structured or subarray void type
    if isinstance(parsed, np.dtypes.StrDType) and parsed.itemsize > 0:
        return {
            "name": FIXED_LENGTH_UTF32_DATA_TYPE_NAME,
            "configuration": {"length_bytes": parsed.itemsize},
        }
    if isinstance(parsed, np.dtypes.BytesDType) and parsed.itemsize > 0:
        return {
            "name": NULL_TERMINATED_BYTES_DATA_TYPE_NAME,
            "configuration": {"length_bytes": parsed.itemsize},
        }
    if isinstance(parsed, np.dtypes.StringDType):
        return STRING_DATA_TYPE_NAME
    if (
        isinstance(parsed, np.dtypes.VoidDType)
        and parsed.fields is None
        and parsed.subdtype is None
        and parsed.itemsize > 0
    ):
        scope_actual = CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
        if (
            scope_actual.claimant(DataTypeDefinition, ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME)
            is not None
        ):
            return {
                "name": ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME,
                "configuration": {"length_bytes": parsed.itemsize},
            }
        return raw_bytes_dtype_name(f"r{parsed.itemsize * 8}")
    if parsed.kind in ("M", "m"):
        # numpy time dtypes take the named-configuration form in Zarr V3. These are
        # handled outside the match statement because numpy's stubs reject the
        # DateTime64DType / TimeDelta64DType classes in class patterns.
        unit, scale_factor = np.datetime_data(
            cast("np.dtype[np.datetime64 | np.timedelta64]", parsed)
        )
        name = (
            NUMPY_DATETIME64_DATA_TYPE_NAME
            if parsed.kind == "M"
            else NUMPY_TIMEDELTA64_DATA_TYPE_NAME
        )
        return {"name": name, "configuration": {"unit": unit, "scale_factor": scale_factor}}
    else:
        match parsed:
            case np.dtypes.BoolDType():
                return BOOL_DATA_TYPE_NAME
            case np.dtypes.Int8DType():
                return INT8_DATA_TYPE_NAME
            case np.dtypes.Int16DType():
                return INT16_DATA_TYPE_NAME
            case np.dtypes.Int32DType():
                return INT32_DATA_TYPE_NAME
            case np.dtypes.Int64DType():
                return INT64_DATA_TYPE_NAME
            case np.dtypes.UInt8DType():
                return UINT8_DATA_TYPE_NAME
            case np.dtypes.UInt16DType():
                return UINT16_DATA_TYPE_NAME
            case np.dtypes.UInt32DType():
                return UINT32_DATA_TYPE_NAME
            case np.dtypes.UInt64DType():
                return UINT64_DATA_TYPE_NAME
            case np.dtypes.Float16DType():
                return FLOAT16_DATA_TYPE_NAME
            case np.dtypes.Float32DType():
                return FLOAT32_DATA_TYPE_NAME
            case np.dtypes.Float64DType():
                return FLOAT64_DATA_TYPE_NAME
            case np.dtypes.Complex64DType():
                return COMPLEX64_DATA_TYPE_NAME
            case np.dtypes.Complex128DType():
                return COMPLEX128_DATA_TYPE_NAME
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")


_FIXED_LENGTH_NUMPY_CODES: Final = {
    FIXED_LENGTH_UTF32_DATA_TYPE_NAME: ("<U", 4),
    NULL_TERMINATED_BYTES_DATA_TYPE_NAME: ("S", 1),
    ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME: ("V", 1),
}
"""The numpy type code and code-unit size of each fixed-length data type
`parse_dtype_v3` writes as a named configuration with a `length_bytes`."""


def _numpy_dtype_from_v3(data_type: str | Mapping[str, object]) -> np.dtype[np.generic] | None:
    """
    Best-effort inverse of `parse_dtype_v3`: the numpy data type equivalent to
    a Zarr V3 data type declaration, or `None` when there is none (an unknown
    extension data type).
    """
    if isinstance(data_type, str):
        # the core spec's raw types (`r8`, `r16`, ...) are numpy's void types
        digits = data_type[1:]
        if data_type[:1] == "r" and digits.isascii() and digits.isdigit():
            bits = int(digits)
            return np.dtype(f"V{bits // 8}") if bits > 0 and bits % 8 == 0 else None
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
    if isinstance(name, str) and name in _FIXED_LENGTH_NUMPY_CODES and isinstance(config, Mapping):
        code, unit_bytes = _FIXED_LENGTH_NUMPY_CODES[name]
        length = config.get("length_bytes")
        if isinstance(length, int) and length > 0 and length % unit_bytes == 0:
            return np.dtype(f"{code}{length // unit_bytes}")
        return None
    if name in (
        NUMPY_DATETIME64_DATA_TYPE_NAME,
        NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    ) and isinstance(config, Mapping):
        unit = config.get("unit")
        scale = config.get("scale_factor", 1)
        code = "M8" if name == NUMPY_DATETIME64_DATA_TYPE_NAME else "m8"
        try:
            return np.dtype(f"{code}[{unit}]" if scale == 1 else f"{code}[{scale}{unit}]")
        except TypeError:
            return None
    return None


_SINGLE_BYTE_DATA_TYPES = ("bool", "int8", "uint8")


def _is_single_byte(data_type: object, scope: ZarrV3Context) -> bool:
    """
    Whether the values of `data_type` are stored in single bytes, where the
    `bytes` codec's endian does not apply, as the data type's definition in
    `scope` declares it (`zarr-metadata`'s `storage_of`): the single-byte
    numbers, `null_terminated_bytes`, `raw_bytes` and `r8`, among others. A
    data type the scope does not know, or derives no default document for, is
    not known to be single-byte.
    """
    if data_type in _SINGLE_BYTE_DATA_TYPES:
        return True
    try:
        key = json.dumps(data_type, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return _is_single_byte_cached(scope, key)


@functools.lru_cache(maxsize=256)
def _is_single_byte_cached(scope: ZarrV3Context, data_type_json: str) -> bool:
    try:
        model = ZarrV3ArrayMetadataModel.create_default(
            context=scope, data_type=json.loads(data_type_json)
        )
    except (ValueError, TypeError):
        return False
    return storage_of(model.data_type) == "single_byte"


# The spellings of the variable-length data types. "string" and "bytes" are the
# names zarr-extensions registers; "variable_length_bytes" is the name
# zarr-python gives variable-length bytes, and "str" is the spelling
# zarr-python 3.0 stored for the string data type (renamed "string" in 3.1), so
# documents it wrote still hold them. Membership exempts a name from the
# ambiguous-numpy-spelling guard in `parse_dtype_v3` (numpy reads "str" as a
# zero-length unicode type), and gives `create_default(chunk_grid="auto")` an
# element size to guess chunks from. Their fill values and codecs are not
# derived here: `zarr-metadata` derives them for the names a scope claims.
_VARIABLE_LENGTH_DATA_TYPES: frozenset[str] = frozenset(
    {STRING_DATA_TYPE_NAME, "str", BYTES_DATA_TYPE_NAME, VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME}
)
# chunk guessing needs an element size; variable-length elements have none,
# so use the size of a pointer, as numpy object arrays do
_VARIABLE_LENGTH_NOMINAL_ITEM_SIZE = 8

_DEFAULT_CHUNK_KEY_ENCODING: DefaultChunkKeyEncodingObject = {
    "name": DEFAULT_CHUNK_KEY_ENCODING_NAME,
    "configuration": {"separator": DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR},
}
"""
The chunk key encoding `create_default` gives an array: the `default` encoding,
spelled with its `/` separator as zarr-python writes it, so a spec built here
compares equal to one read from a store zarr-python wrote (specs compare
fields literally, and `zarr-metadata` writes the same encoding as
``{"name": "default"}``).
"""


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

    def check(kind: type[Definition], label: str, name: object, loc: str) -> None:
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
            if codec.get("name") == SHARDING_INDEXED_CODEC_NAME and isinstance(config, Mapping):
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


type ExtraFieldKind = type[ArrayExtraFieldDefinition | GroupExtraFieldDefinition]


def _refused_extra_fields(
    extra_fields: Mapping[str, object], kind: ExtraFieldKind, scope: ZarrV3Context
) -> list[str]:
    """
    The extra fields of a node document that the Zarr V3 spec's rules for
    additional fields refuse in `scope`, each as a ``<loc>: <message>`` line.

    An extra field whose key a definition in `scope` claims is one the reader
    recognizes: that definition judges it when the whole document is read, so
    it is skipped here. Every other extra field is one the reader does not
    recognize, which the spec admits only when it explicitly waives
    understanding: "An implementation MUST fail to open Zarr groups or arrays
    if any metadata fields are present which (a) the implementation does not
    recognize and (b) are not explicitly set to ``"must_understand": false``."
    A waived field is an object that says ``"must_understand": false``, kept
    as written, whatever else it holds: the reading `zarr-metadata` gives the
    rule, and the one its strategies draw.
    """
    problems: list[str] = []
    for key, value in extra_fields.items():
        if scope.claimant(kind, key) is not None:
            continue
        if not isinstance(value, Mapping):
            problems.append(
                f"{key}: not a field of this spec, and not an extra field the spec permits: "
                'an extra field is an object that says "must_understand": false, '
                f"got {value!r}"
            )
            continue
        if value.get("must_understand") is not False:
            problems.append(
                f"{key}: no {kind.label} in the scope recognizes this field, and it does not "
                'set "must_understand": false, so the spec requires a reader to refuse it'
            )
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


def canonical_envelope(declaration: object, *, bare_name: bool = False) -> object:
    """
    A declaration in the canonical envelope spelling, as `zarr-metadata`
    writes it: a bare name becomes ``{"name": ...}``, an explicit
    ``"must_understand": true`` is dropped (leaving it out means true), an
    empty ``configuration`` is dropped, and the JSON arrays of a
    configuration are tuples (`tuplify_json`), as a spec holds every other
    sequence, so a document read from JSON compares equal to one built in
    Python. With `bare_name`, for data types, an object left with nothing but
    a name becomes that bare name, as core data types have been written since
    Zarr v3.0. Anything else passes through for validation to refuse.
    """
    if isinstance(declaration, str):
        return declaration if bare_name else {"name": declaration}
    if not isinstance(declaration, Mapping):
        return declaration
    result = {
        key: tuplify_json(value) if key == "configuration" else value
        for key, value in declaration.items()
        if not (key == "must_understand" and value is True)
        and not (key == "configuration" and isinstance(value, Mapping) and not value)
    }
    if bare_name and set(result) == {"name"} and isinstance(result["name"], str):
        return result["name"]
    return result


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
    tuple[AnyNamedConfig, ...],
    BeforeValidator(ensure_multiple),
    BeforeValidator(reject_bare_codec),
]
"""
The codec pipeline a spec holds: one or more codec objects. Bare codec names
are written as objects before the field is validated (see
`ArraySpec.canonicalize_declarations`).
"""


class ArraySpecCreate(TypedDict, total=False):
    """
    The fields `ArraySpec.create_default` takes as keyword arguments: every
    `ArraySpec` field except `chunk_grid`, which it takes on its own, since it
    also accepts `"auto"`.
    """

    shape: Sequence[int]
    data_type: npt.DTypeLike | Mapping[str, object]
    attributes: TBaseAttr
    chunk_key_encoding: ChunkKeyEncodingLike
    fill_value: FillValueLike
    codecs: Sequence[CodecLike]
    storage_transformers: Sequence[str | AnyNamedConfig]
    dimension_names: Sequence[str | None] | None
    extra_fields: Mapping[str, ZarrV3ExtensionField]


class ArraySpecUpdate(ArraySpecCreate, total=False, closed=True):
    """The fields an `ArraySpec.replace` call may replace."""

    chunk_grid: ChunkGridLike


class BaseGroupSpecUpdate(TypedDict, total=False, closed=True):
    """The fields a `BaseGroupSpec.replace` call may replace."""

    attributes: TBaseAttr
    extra_fields: Mapping[str, ZarrV3ExtensionField]


class GroupSpecUpdate(TypedDict, total=False, closed=True):
    """The fields a `GroupSpec.replace` call may replace."""

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
        The chunk grid declaration: any chunk grid the scope claims, judged by
        its definition when the document is validated (the core spec defines
        the regular grid; `zarr-metadata` also defines the rectilinear one).
    chunk_key_encoding: NamedConfig
        The chunk key encoding declaration: any chunk key encoding the scope
        claims, judged the same way.
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
    chunk_grid: AnyNamedConfig
    chunk_key_encoding: AnyNamedConfig
    fill_value: Annotated[FillValue, BeforeValidator(canonicalize_fill_value)]
    codecs: CodecTuple
    storage_transformers: tuple[AnyNamedConfig, ...] = ()
    dimension_names: tuple[str | None, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def canonicalize_declarations(cls, data: object, info: ValidationInfo) -> object:
        """
        Read the document's declarations in every envelope spelling the spec
        allows, and normalize them to the canonical one, so a spec constructed
        from any spelling compares equal to one constructed from another and
        to what a store read gives back.

        The envelope rules follow `zarr-metadata`'s canonical form: a bare
        name string is the object ``{"name": ...}``, except that a data type
        with nothing to configure is written as its bare name; an explicit
        ``"must_understand": true`` is dropped, since leaving it out means
        true; an empty ``configuration`` mapping is dropped. A numpy
        `data_type` is converted to its Zarr V3 form via `parse_dtype_v3`, in
        the scope this validation reads in (a model validator rather than
        field metadata so the class's `zarr_scope` can serve as the scope
        when the validation context carries none).

        One rule of zarr-python's is kept as well: the `bytes` codec's
        `endian` configuration, a no-op for a single-byte data type, is
        dropped when the chunk it receives is of one, as the data type's
        definition in the scope declares its storage. That chunk's data type
        is the array's as changed by the array->array codecs before `bytes`
        in its pipeline (`cast_value` sets it to its configured `data_type`,
        `transpose` and `scale_offset` keep it, and after any other codec it is
        unknown, so the endian is kept), so a cast to a multi-byte type keeps
        the endian the cast chunk needs. The rule reaches inside a sharding
        codec's inner pipeline, which continues the data type its position in
        the outer pipeline hands it; its index codecs, which encode the
        uint64 index, keep their endianness. (`zarr-metadata`'s canonical
        document model does not equate the endian spellings; `==` on specs
        compares fields literally.)
        """
        if not isinstance(data, Mapping):
            return data
        document = dict(data)
        for field in ("chunk_grid", "chunk_key_encoding"):
            if field in document:
                document[field] = canonical_envelope(document[field])
        transformers = document.get("storage_transformers")
        if isinstance(transformers, Sequence) and not isinstance(transformers, str):
            document["storage_transformers"] = tuple(
                canonical_envelope(transformer) for transformer in transformers
            )
        scope = cls._scope(info.context)
        if "data_type" in document:
            document["data_type"] = parse_dtype_v3(
                cast(
                    "npt.DTypeLike | Mapping[str, object]",
                    canonical_envelope(document["data_type"], bare_name=True),
                ),
                scope=scope,
            )
        if "codecs" not in document:
            return document
        codecs = document["codecs"]
        if isinstance(codecs, (str, Mapping)) or not isinstance(codecs, Sequence):
            # a lone codec is refused by the field validator; leave it alone
            return document

        def parse_data_type(declaration: object) -> object:
            try:
                return parse_dtype_v3(
                    cast(
                        "npt.DTypeLike | Mapping[str, object]",
                        canonical_envelope(declaration, bare_name=True),
                    )
                )
            except (ValueError, TypeError):
                return None

        def normalize(codec: object, data_type: object) -> tuple[object, object]:
            """
            Normalize one codec, given the data type of the chunk it receives;
            return it with the data type of the chunk it passes on.
            """
            codec = canonical_envelope(codec)
            if not isinstance(codec, Mapping):
                return codec, None
            name = codec.get("name")
            config = codec.get("configuration")
            if name == CAST_VALUE_CODEC_NAME:
                cast_to = config.get("data_type") if isinstance(config, Mapping) else None
                return codec, None if cast_to is None else parse_data_type(cast_to)
            if (
                name == BYTES_CODEC_NAME
                and isinstance(config, Mapping)
                and set(config) == {"endian"}
                and data_type is not None
                and _is_single_byte(data_type, scope_actual)
            ):
                return {"name": BYTES_CODEC_NAME}, data_type
            if name == SHARDING_INDEXED_CODEC_NAME and isinstance(config, Mapping):
                new_config = dict(config)
                inner = config.get("codecs")
                if isinstance(inner, Sequence) and not isinstance(inner, str):
                    new_config["codecs"] = normalize_pipeline(inner, data_type)
                index = config.get("index_codecs")
                if isinstance(index, Sequence) and not isinstance(index, str):
                    new_config["index_codecs"] = normalize_pipeline(index, UINT64_DATA_TYPE_NAME)
                return {**codec, "configuration": new_config}, data_type
            # an array -> array codec this walk does not know may change the data
            # type, so what follows it gets an unknown one (and keeps its endian),
            # as zarr-metadata's pipeline reader does not guess either
            return codec, data_type if name in _DATA_TYPE_KEEPING_CODECS else None

        def normalize_pipeline(pipeline: Sequence[object], data_type: object) -> tuple[object, ...]:
            normalized: list[object] = []
            for codec in pipeline:
                codec, data_type = normalize(codec, data_type)
                normalized.append(codec)
            return tuple(normalized)

        data_type = parse_data_type(document["data_type"]) if "data_type" in document else None
        scope_actual = CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
        document["codecs"] = normalize_pipeline(codecs, data_type)
        return document

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
        class's `zarr_scope`, then to `zarr_metadata`'s `CORE_AND_EXTENSIONS_CONTEXT`.

        Extra fields (members the spec does not define) are judged in the
        same scope: one whose key a definition in the scope claims, by that
        definition; any other, by the spec's rule for fields a reader does
        not recognize (an object that says ``"must_understand": false``, kept
        as written).

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
                self.model_dump(), CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
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

    # No return annotation, as on `ExtraFieldsBase`: pydantic would take it as
    # the serialization schema in place of the fields' schema.
    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
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

    def to_metadata_json(self) -> ZarrV3ArrayMetadataJSON:
        """
        This array's Zarr V3 metadata document: the content of its `zarr.json`.

        Unlike `model_dump`, this takes no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV3ArrayMetadataJSON`. The document
        is what `model_dump()` returns: `dimension_names` is omitted when it is
        `None`, an attributes model is dumped by alias, and `extra_fields` are
        written as top-level members.
        """
        return cast("ZarrV3ArrayMetadataJSON", self.model_dump())

    def to_stored_metadata_json(self) -> ZarrV3ArrayStoredJSON:
        """
        This array's Zarr V3 metadata as a store holds it: a mapping from its
        one store key, `zarr.json`, to `to_metadata_json()`.
        `from_stored_metadata_json` reads it back.
        """
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: self.to_metadata_json()}

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read an array from its metadata as a store holds it: a mapping from
        store key to parsed JSON document, as `to_stored_metadata_json`
        returns it. Raises `ValueError` when `data` holds no array's
        `zarr.json` at the root, or holds documents of any other node, and
        `ValidationError` when the document is not a valid array.
        """
        return cls.model_validate(stored_array(_stored_nodes(data)))

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.generic],
        *,
        # TBaseAttr, not TAttr: covariant parameters may not appear in
        # argument positions. Validation enforces the class's attribute type.
        attributes: Literal["auto"] | TBaseAttr = "auto",
        chunk_grid: Literal["auto"] | ChunkGridLike = "auto",
        chunk_key_encoding: Literal["auto"] | ChunkKeyEncodingLike = "auto",
        fill_value: Literal["auto"] | FillValueLike = "auto",
        codecs: Literal["auto"] | Sequence[CodecLike] = "auto",
        storage_transformers: Literal["auto"] | Sequence[str | AnyNamedConfig] = "auto",
        dimension_names: Literal["auto"] | Sequence[str | None] = "auto",
    ) -> Self:
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array :
            object that conforms to the numpy array API: chunk grid, chunk key
            encoding, fill value, codecs and attributes are probed from
            same-named attributes, and a fill value or codecs the object lacks
            (a numpy array has neither) take the defaults `zarr-metadata`
            derives for its data type in this class's scope, as in
            `create_default`. Keyword arguments other than `"auto"` override
            what the array provides. To model a stored array, pass its stored
            metadata document to `from_stored_metadata_json` instead.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """

        if attributes == "auto":
            attributes_actual = cast("TAttr", auto_attributes(array))
        else:
            # the constructor validates this against the class's TAttr
            attributes_actual = cast("TAttr", attributes)

        if chunk_grid == "auto":
            chunk_grid_actual = auto_chunk_grid(array)
        else:
            chunk_grid_actual = chunk_grid

        chunk_key_actual: ChunkKeyEncodingLike
        if chunk_key_encoding == "auto":
            chunk_key_actual = _DEFAULT_CHUNK_KEY_ENCODING
        else:
            chunk_key_actual = chunk_key_encoding

        storage_transformers_actual: Sequence[str | AnyNamedConfig]
        if storage_transformers == "auto":
            storage_transformers_actual = auto_storage_transformers(array)
        else:
            storage_transformers_actual = storage_transformers

        dimension_names_actual: Sequence[str | None] | None
        if dimension_names == "auto":
            dimension_names_actual = auto_dimension_names(array)
        else:
            dimension_names_actual = dimension_names

        document: dict[str, object] = {
            "shape": array.shape,
            "data_type": parse_dtype_v3(array.dtype, scope=cls.zarr_scope),
            "chunk_grid": chunk_grid_actual,
            "attributes": attributes_actual,
            "chunk_key_encoding": chunk_key_actual,
            "storage_transformers": tuple(storage_transformers_actual),
            "dimension_names": dimension_names_actual,
        }
        # a fill value or codecs the array carries are taken from it; an array
        # without them (a numpy array) gets the defaults zarr-metadata derives
        # for its data type in this class's scope, as `create_default` does
        if fill_value != "auto":
            document["fill_value"] = fill_value
        elif hasattr(array, "fill_value"):
            document["fill_value"] = read_attribute(array, "fill_value")
        if codecs != "auto":
            document["codecs"] = tuple(codecs)
        elif hasattr(array, "codecs"):
            document["codecs"] = auto_codecs(array)
        scope = CORE_AND_EXTENSIONS_CONTEXT if cls.zarr_scope is None else cls.zarr_scope
        derived = _derived_array_members(document, scope)
        # the inputs above are of the types an input takes, which validation
        # writes as the types a spec holds; `model_validate` is that door
        return cls.model_validate({**derived, **document})

    @classmethod
    def create_default(
        cls,
        *,
        chunk_grid: ChunkGridLike | Literal["auto"] | None = None,
        **fields: Unpack[ArraySpecCreate],
    ) -> Self:
        """
        Create an `ArraySpec` with `fields` set and every other field set to a
        default, the counterpart of `zarr-metadata`'s
        `ZarrV3ArrayMetadata.create_default`, which derives the defaults.

        Each field not given takes `zarr-metadata`'s default, derived from the
        fields given where it depends on them: a data type brings its default
        fill value (zero, `False`, the empty string, `"NaT"`) and its
        array -> bytes codec (`vlen-utf8` for `string`, `vlen-bytes` for
        `bytes`, `bytes` otherwise), and a shape brings one regular chunk
        covering the array. With nothing given, the array is a scalar `uint8`.
        Two choices are pydantic-zarr's: the chunk key encoding is the
        `default` one spelled as zarr-python writes it (with its `/`
        separator), and `chunk_grid="auto"` guesses a chunk shape from the
        array's size in bytes, the heuristic zarr-python applies when no chunks
        are given. A numpy data type is converted to its Zarr V3 form first.

        On a subclass, a field of the array document that the class declares
        a default for counts as given: a class that pins
        `data_type: Literal["uint16"] = "uint16"` or
        `fill_value: Literal["NaN"] = "NaN"` keeps that value, and the fields
        not given are derived from it. Fields a subclass adds take their own
        defaults, and so does a field whose default factory reads other fields;
        nothing is derived from such a field, so when it is the shape or the
        data type, the fields that depend on it have to be given, and
        `chunk_grid="auto"`, which guesses from both, is refused.
        A field the class requires and the
        data cannot provide (an attributes model with required keys, say) has
        to be passed, as it does to the constructor. A subclass whose
        construction needs more than defaults (attributes computed from the
        shape, say) can override this method; to stay a compatible override,
        keep the keyword-only signature and pass the computed fields on to
        `super().create_default(...)`.

        Everything is validated in one pass, in this class's scope, and a
        problem in one field is reported as itself: only the shape, the data
        type and a given fill value feed the derivation, and when they do not
        fit together, what each derives alone is still derived. A data type
        nothing in the scope derives a fill value for (an extension data type,
        or a spelling only another scope claims) needs `fill_value` given; its
        codecs are then derived. A fill value left to a default factory that
        reads other fields is not given in this sense, so for such a data type
        pass `fill_value` or give `codecs`.
        A data type name that is not a Zarr V3 name but that numpy reads
        as some other type (`"i8"`, `"float"`) is refused as a likely mistake,
        with the V3 spelling suggested.

        Examples
        --------
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> ArraySpec.create_default(shape=(10, 10), data_type="uint16", dimension_names=("y", "x"))
        ArraySpec(zarr_format=3, node_type='array', attributes={}, shape=(10, 10), data_type='uint16', chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10, 10)}}, chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}}, fill_value=0, codecs=({'name': 'bytes', 'configuration': {'endian': 'little'}},), storage_transformers=(), dimension_names=('y', 'x'))
        """
        # A field of the array document that the class declares a default for
        # counts as given, so the fields not given are derived from it rather
        # than from zarr-metadata's generic defaults. Fields a subclass adds are
        # left to pydantic, and so is a default whose factory reads other fields:
        # it runs at validation, and nothing derived replaces it.
        document: dict[str, object] = {}
        factory_fields: set[str] = set()
        for name in _ARRAY_DOCUMENT_FIELDS:
            info = cls.model_fields[name]
            if name in fields or info.is_required():
                continue
            if info.default_factory_takes_validated_data:
                factory_fields.add(name)
                continue
            document[name] = info.get_default(call_default_factory=True)
        document.update(fields)
        if "data_type" in document:
            document["data_type"] = parse_dtype_v3(
                cast("npt.DTypeLike | Mapping[str, object]", document["data_type"]),
                scope=cls.zarr_scope,
            )
        if "chunk_key_encoding" not in factory_fields:
            document.setdefault("chunk_key_encoding", _DEFAULT_CHUNK_KEY_ENCODING)
        if chunk_grid == "auto":
            unguessable = sorted(factory_fields & {"shape", "data_type"})
            if unguessable:
                msg = (
                    'chunk_grid="auto" guesses a chunk shape from the shape and the data '
                    f"type, but {cls.__name__} leaves these to a default factory, with no "
                    f"value yet: {', '.join(unguessable)}. Give each, or give chunk_grid."
                )
                raise ValueError(msg)
            document["chunk_grid"] = _guessed_chunk_grid(
                tuple(cast("Sequence[int]", document.get("shape", ()))),
                document.get("data_type", "uint8"),
            )
        elif chunk_grid is not None:
            document["chunk_grid"] = chunk_grid
        scope = CORE_AND_EXTENSIONS_CONTEXT if cls.zarr_scope is None else cls.zarr_scope
        # a source left to its factory has no value yet, so nothing is derived
        # from it: what depends on it is the caller's to give
        undecided = set(factory_fields)
        if "data_type" in factory_fields:
            undecided |= _DERIVED_FROM_DATA_TYPE
        if "shape" in factory_fields:
            undecided |= _DERIVED_FROM_SHAPE
        derived = {
            name: value
            for name, value in _derived_array_members(document, scope).items()
            if name not in undecided
        }
        return cls.model_validate({**derived, **document})

    def replace(self, **changes: Unpack[ArraySpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so correlated fields (for an
        array, a new `data_type` with a matching `fill_value`, or a new
        `shape` with a matching `chunk_grid`) change together atomically.
        Unlike `model_copy(update=...)`, the result is validated. Passing
        `extra_fields` replaces the extra fields as a whole.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

    def with_sharding_codec(
        self,
        subchunk_shape: Sequence[int],
        *,
        index_location: ShardingIndexLocation = "end",
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
            "name": SHARDING_INDEXED_CODEC_NAME,
            "configuration": {
                "chunk_shape": tuple(subchunk_shape),
                "codecs": self.codecs,
                "index_codecs": (
                    {"name": BYTES_CODEC_NAME, "configuration": {"endian": "little"}},
                    {"name": CRC32C_CODEC_NAME},
                ),
                "index_location": index_location,
            },
        }
        return self.replace(codecs=(sharding_codec,))

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
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> x = ArraySpec.create_default(shape=(10, 10), data_type="uint8", attributes={'foo': 10})
        >>> print(x.like(x)) # it is like itself.
        True
        >>> y = ArraySpec.create_default(shape=(10, 10), data_type="uint8", attributes={'foo': 11})
        >>> print(x.like(y)) # x and y are the same, other than their attributes
        False
        >>> print(x.like(y, exclude={'attributes'}))
        True
        """
        if include is None and exclude is None:
            # Compare the metadata documents by meaning rather than spelling:
            # `zarr-metadata` canonicalizes what it interprets, so a fill value
            # written "NaN" equals one written "0x7fc00000", and a codec equals
            # itself with a no-op configuration key dropped. Attributes are not
            # interpreted, so they are compared separately, as plain fields.
            return self._canonical_document() == other._canonical_document() and model_like(
                self, other, include={"attributes"}
            )

        return model_like(self, other, include=include, exclude=exclude)


def ensure_member_names_v3(data: Mapping[str, object]) -> Mapping[str, object]:
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
        (``CORE_CONTEXT`` claims ``consolidated_metadata``, whose contents are kept as
        written and not read as members); any other must be an object that
        says ``"must_understand": false``, kept as written. On a class with
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

    def replace(self, **changes: Unpack[BaseGroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        Unlike `model_copy(update=...)`, the result is validated, so an
        invalid change is refused. Passing `extra_fields` replaces the extra fields as a whole.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

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

    def to_metadata_json(self) -> ZarrV3GroupMetadataJSON:
        """
        This group's Zarr V3 metadata document: the content of its `zarr.json`.

        Unlike `model_dump`, this takes no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV3GroupMetadataJSON`. Its
        `extra_fields` (such as `consolidated_metadata`) are written as
        top-level members. A group's members are not part of its metadata
        document, so a `GroupSpec`'s `members` are left out.
        """
        return cast("ZarrV3GroupMetadataJSON", self.model_dump(exclude={"members"}))

    def to_stored_metadata_json(self) -> ZarrV3GroupStoredJSON:
        """
        This group's Zarr V3 metadata as a store holds it: a mapping from each
        store key to the JSON document stored there. A `BaseGroupSpec` holds
        no members, so the mapping has one key, `zarr.json`, mapped to
        `to_metadata_json()`; `GroupSpec` adds each member's documents.
        """
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: self.to_metadata_json()}

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a group's own metadata from a stored layout: a mapping from store
        key to parsed JSON document, as `to_stored_metadata_json` returns it.
        Only the root's `zarr.json` is read; members' documents are left
        unread. Raises `ValueError` when `data`
        holds no group's `zarr.json` at the root, and `ValidationError` when
        the document is not a valid group.
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
        other_parsed = other._own_metadata() if isinstance(other, GroupSpec) else other
        return model_like(self, other_parsed, include=include, exclude=exclude)


class GroupSpec(BaseGroupSpec[TAttr], Generic[TAttr, TMembers]):
    """
    A model of a Zarr Version 3 Group.

    The second type parameter is the type of the whole `members` mapping. Leave it
    out for a group of any members, give a `Mapping` to constrain every member's
    type (`GroupSpec[Attrs, Mapping[str, MyArraySpec]]`), or give a closed
    `TypedDict` to fix the member names (`GroupSpec[Attrs, MyMembers]`).

    Attributes
    ----------

    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr
        The user-defined attributes of this group.
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
        Validate member names as Zarr V3 node names.

        This runs as a model validator rather than field metadata so that a
        subclass narrowing the `members` annotation (for example to a closed
        `TypedDict` of members) cannot drop it.
        """
        ensure_members_mapping(self.members)
        ensure_member_names_v3(self.members)
        return self

    @override
    def replace(self, **changes: Unpack[GroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes land in one validation pass, so `attributes` and `members`
        change together atomically. Unlike `model_copy(update=...)`, the
        result is validated. Passing `extra_fields` replaces the extra fields as a whole.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

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

    @override
    def to_stored_metadata_json(self) -> ZarrV3GroupStoredJSON:
        """
        This hierarchy's Zarr V3 metadata as a store holds it: a mapping from
        each store key to the JSON document stored there. That is this
        group's `zarr.json`, and each member's at any depth, at its metadata
        key (`"a/zarr.json"`, `"a/b/zarr.json"`). A key is the member's path
        as `to_flat` spells it (`"/a/b"`), with the leading `/` stripped and
        `/zarr.json` appended, as the Zarr V3 spec forms metadata keys from
        hierarchy paths (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L382-L388).
        These are the metadata documents zarr-python writes, so a consumer can
        write the hierarchy by writing each document, as JSON, at its key
        under the hierarchy's root. `from_stored_metadata_json` reads it back.
        """
        members: dict[str, ZarrV3StoredDocumentJSON] = {
            f"{path.removeprefix('/')}/{ZARR_V3_GROUP_METADATA_STORE_KEY}": node.to_metadata_json()
            for path, node in self.to_flat().items()
            if path
        }
        # each member's document is of the type's `extra_items`; the cast is for
        # checkers that do not yet type writes to `extra_items` (pyrefly)
        return cast("ZarrV3GroupStoredJSON", {**super().to_stored_metadata_json(), **members})

    @override
    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a hierarchy from its metadata as a store holds it: a mapping from
        store key to parsed JSON document, as `to_stored_metadata_json`
        returns it. Each node's `zarr.json` is read, and the nodes are
        assembled with `from_flat`. A layout cannot tell a group whose members
        were not read from one with none, so every group is read as a
        `GroupSpec`, with the members the layout holds.

        Raises `ValueError` when `data` holds no group's `zarr.json` at the
        root, a key with a leading `/` or an empty segment, a key other than a
        `zarr.json`, or a node whose parent is not a group in `data` (a store
        holds no implicit groups), and `ValidationError` when a document is
        not a valid node (one whose `node_type` is neither `"array"` nor
        `"group"` is validated as a group).
        """
        return cls.from_flat(stored_flat(_stored_nodes(data), ArraySpec, BaseGroupSpec))

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

    @override
    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare a `GroupSpec` to another `GroupSpec` or a `BaseGroupSpec`.

        With no `include`/`exclude` filters, the groups' own metadata is
        compared as plain fields and each member is compared by that member's
        own `like`, so member arrays compare by the meaning of their metadata
        documents (a fill value written `"NaN"` equals one written
        `"0x7fc00000"`), exactly as `ArraySpec.like` judges them directly.
        A `BaseGroupSpec` — a group considered without members — compares as
        that projection: by the group's own metadata alone, whatever this
        side's members hold.
        With filters, models are converted to `dict` via the `model_dump`
        method of `pydantic.BaseModel` and compared with the `==` operator.

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
        >>> from pydantic_zarr.v3 import ArraySpec, GroupSpec
        >>> foo = ArraySpec.create_default(shape=(10,), data_type="int64")
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
            return self._own_metadata().like(other, include=include, exclude=exclude)

        if include is None and exclude is None:
            # Compare the groups' own metadata as plain fields and each member
            # by that member's own `like`, so member arrays compare by the
            # meaning of their metadata documents, exactly as `ArraySpec.like`
            # judges them directly.
            if not model_like(self, other, exclude={"members"}):
                return False
            if self.members.keys() != other.members.keys():
                return False
            for name, member in self.members.items():
                other_member = other.members[name]
                if isinstance(member, ArraySpec):
                    if not isinstance(other_member, ArraySpec) or not member.like(other_member):
                        return False
                elif isinstance(other_member, ArraySpec) or not member.like(other_member):
                    return False
            return True

        return model_like(self, other, include=include, exclude=exclude)


def _stored_nodes(data: Mapping[str, object]) -> StoredNodes:
    """
    Read a stored layout, a mapping from store key to parsed JSON document,
    as each node's type and metadata document, keyed by the node's key prefix
    (see `split_store_key`): `""` for the root, `"a/b"` for the node at
    `/a/b`. The type is the document's `node_type` when that is
    `"array"` or `"group"`, and `None` otherwise. Raises `ValueError` naming a
    malformed key, or a key other than a node's `zarr.json`.
    """
    nodes: dict[str, tuple[ZarrV3NodeType | None, object]] = {}
    for key, document in data.items():
        path, name = split_store_key(key)
        if name != ZARR_V3_GROUP_METADATA_STORE_KEY:
            msg = (
                f"Unrecognized store key {key!r}: a Zarr V3 node's metadata is stored at "
                f"{ZARR_V3_GROUP_METADATA_STORE_KEY!r}."
            )
            raise ValueError(msg)
        node_type = document.get("node_type") if isinstance(document, Mapping) else None
        nodes[path] = (
            node_type if is_zarr_v3_node_type(node_type) else None,
            document,
        )
    return nodes


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


def _is_named_config(value: object) -> TypeIs[AnyNamedConfig]:
    """
    Whether `value` has the shape of `AnyNamedConfig`: a mapping with a string
    `name` and, when present, a mapping `configuration`.
    """
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("configuration", {}), Mapping)
    )


def _as_named_config(value: object) -> AnyNamedConfig:
    """
    `value`, a named configuration read from a duck-typed array, unchanged.

    Raises
    ------
    TypeError
        If `value` does not have the shape of `AnyNamedConfig`.
    """
    if _is_named_config(value):
        return value
    msg = f"Expected a named configuration ({{'name': ..., 'configuration': ...}}), got {value!r}."
    raise TypeError(msg)


def _as_tuple(value: object) -> tuple[object, ...]:
    """
    `value`, a sequence read from a duck-typed array, as a tuple.

    Raises
    ------
    TypeError
        If `value` is not an iterable, or is a string or a mapping.
    """
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        msg = f"Expected an iterable, got {value!r}."
        raise TypeError(msg)
    return tuple(value)


def auto_attributes(data: object) -> Mapping[str, object]:
    if hasattr(data, "attributes"):
        attributes = read_attribute(data, "attributes")
        if isinstance(attributes, Mapping):
            return attributes
    return {}


def auto_chunk_grid(data: object) -> AnyNamedConfig:
    if hasattr(data, "chunk_shape"):
        chunk_shape = as_shape(read_attribute(data, "chunk_shape"))
        return {"name": REGULAR_CHUNK_GRID_NAME, "configuration": {"chunk_shape": chunk_shape}}
    elif hasattr(data, "shape"):
        chunk_shape = as_shape(read_attribute(data, "shape"))
        return {"name": REGULAR_CHUNK_GRID_NAME, "configuration": {"chunk_shape": chunk_shape}}
    raise ValueError("Cannot get chunk grid from object without .shape or .chunk_shape attribute")


def auto_chunk_key_encoding(data: object) -> AnyNamedConfig:
    if hasattr(data, "metadata"):
        metadata = read_attribute(data, "metadata")
        if hasattr(metadata, "chunk_key_encoding"):
            # zarr-python's array metadata, whose `to_dict` gives the JSON form
            to_dict = read_attribute(metadata, "to_dict")
            document = to_dict() if callable(to_dict) else None
            if not isinstance(document, Mapping):
                msg = f"Cannot read the chunk key encoding of {metadata!r}."
                raise TypeError(msg)
            return _as_named_config(document["chunk_key_encoding"])
    return {
        "name": DEFAULT_CHUNK_KEY_ENCODING_NAME,
        "configuration": {"separator": DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR},
    }


def _checked_declaration(declaration: str | Mapping[str, object]) -> ZarrV3DeclarationJSON:
    """
    `declaration` as a Zarr V3 declaration: a name, or a named configuration
    with no problem (an unknown key included).

    Raises
    ------
    ValueError
        If `declaration` is a mapping that is not a named configuration.
    """
    if isinstance(declaration, str):
        return declaration
    checked, problems = check_json(declaration, ZarrV3NamedConfigJSON)
    if checked is None or problems:
        msg = f"{declaration!r} is not a named configuration: " + "; ".join(
            problem.message for problem in problems
        )
        raise ValueError(msg)
    return checked


class _DerivingMembers(TypedDict, total=False, closed=True):
    """The members of an array document `zarr-metadata` derives other members from."""

    shape: tuple[int, ...]
    data_type: ZarrV3DeclarationJSON
    fill_value: JSONValue


def dtype_default_fill_value(
    dtype: npt.DTypeLike | Mapping[str, object], *, scope: ZarrV3Context | None = None
) -> FillValue:
    """
    The fill value an array of data type `dtype` gets when none is given: the
    default `zarr-metadata` derives for it in `scope` (its default scope when
    `None`), the value zarr-python uses: zero, `False`, the empty string, null
    bytes, or `"NaT"` for a time data type. A numpy data type is converted to
    its Zarr V3 form first.

    Raises
    ------
    ValueError
        If nothing in the scope derives a fill value for the data type (an
        extension data type, say).
    """
    parsed = parse_dtype_v3(dtype, scope=scope)
    context = CORE_AND_EXTENSIONS_CONTEXT if scope is None else scope
    try:
        document = ZarrV3ArrayMetadataModel.create_default(
            context=context, data_type=_checked_declaration(parsed)
        ).to_json()
    except (ValueError, TypeError) as e:
        msg = f"Cannot determine a default fill value for data type {parsed!r}: pass one."
        raise ValueError(msg) from e
    return document["fill_value"]


def auto_fill_value(data: object) -> FillValue:
    """
    The fill value of `data`, when it has one, or else the default
    `zarr-metadata` derives for its `dtype` (see `dtype_default_fill_value`).
    """
    if hasattr(data, "fill_value"):
        # validation of the spec judges the value against its data type
        return cast("FillValue", read_attribute(data, "fill_value"))
    if hasattr(data, "dtype"):
        return dtype_default_fill_value(np.dtype(as_dtype_like(read_attribute(data, "dtype"))))
    raise ValueError("Cannot determine a default fill value for an object without a dtype.")


_DATA_TYPE_KEEPING_CODECS = frozenset({TRANSPOSE_CODEC_NAME, SCALE_OFFSET_CODEC_NAME})
"""
The array -> array codecs that pass on a chunk of the data type they receive
(the specs say so for both). `cast_value` sets the data type it passes on; any
other codec ahead of the array -> bytes codec leaves it unknown.
"""

_ARRAY_DOCUMENT_FIELDS: tuple[str, ...] = (
    "shape",
    "data_type",
    "chunk_grid",
    "chunk_key_encoding",
    "fill_value",
    "codecs",
    "storage_transformers",
    "dimension_names",
)
"""The `ArraySpec` fields that are members of the array's metadata document, other than its attributes."""

_DERIVED_FROM_DATA_TYPE = frozenset({"data_type", "fill_value", "codecs"})
_DERIVED_FROM_SHAPE = frozenset({"shape", "chunk_grid", "chunk_key_encoding"})


def _derived_array_members(
    document: Mapping[str, object], scope: ZarrV3Context
) -> dict[str, object]:
    """
    The members of an array document that `zarr-metadata` derives for the
    members `document` gives, in `scope`, without the members `document` gives.

    Only the members the others depend on are handed over: the shape, the
    data type, and the fill value when one is given (so a data type with no
    default fill value still brings its codecs). Every other member given is
    left for validation to judge, so a problem in it is reported as itself.
    When `zarr-metadata` refuses the sources together, what each derives
    alone is still derived: the data type's fill value and codecs, and the
    shape's chunk grid. A source that is itself refused derives nothing, and
    validation reports it.
    """

    def derive(given: Mapping[str, object], takes: frozenset[str]) -> dict[str, object] | None:
        checked, problems = check_json(given, _DerivingMembers)
        if checked is None or problems:
            return None
        try:
            derived = ZarrV3ArrayMetadataModel.create_default(context=scope, **checked).to_json()
        except (ValueError, TypeError):
            return None
        return {
            name: value for name, value in derived.items() if name in takes and name not in document
        }

    shape = {"shape": document["shape"]} if "shape" in document else {}
    data_type = {"data_type": document["data_type"]} if "data_type" in document else {}
    fill_value = (
        {"fill_value": canonicalize_fill_value(document["fill_value"])}
        if "fill_value" in document
        else {}
    )
    everything = _DERIVED_FROM_DATA_TYPE | _DERIVED_FROM_SHAPE
    for given in ({**shape, **data_type, **fill_value}, {**shape, **data_type}):
        together = derive(given, everything)
        if together is not None:
            return together
    found: dict[str, object] = {}
    for given in ({**data_type, **fill_value}, data_type):
        from_data_type = derive(given, _DERIVED_FROM_DATA_TYPE)
        if from_data_type is not None:
            found.update(from_data_type)
            break
    from_shape = derive(shape, _DERIVED_FROM_SHAPE)
    if from_shape is not None:
        found.update(from_shape)
    elif "chunk_grid" not in document and isinstance(document.get("shape"), Sequence):
        # a shape zarr-metadata refuses still gets a grid of its rank, so that
        # validation reaches the shape and reports it
        lengths = cast("Sequence[object]", document["shape"])
        if all(isinstance(n, int) for n in lengths):
            chunk_shape = tuple(max(cast("int", n), 1) for n in lengths)
            found["chunk_grid"] = {
                "name": REGULAR_CHUNK_GRID_NAME,
                "configuration": {"chunk_shape": chunk_shape},
            }
    return found


def _guessed_chunk_grid(shape: tuple[int, ...], data_type: object) -> RegularChunkGridMetadata:
    """
    A regular chunk grid for an array of `shape` and `data_type`, its chunk
    shape guessed from the array's size in bytes as zarr-python guesses it.
    A data type with no knowable element size (an extension data type) gets
    one chunk covering the array.
    """
    np_dtype = (
        _numpy_dtype_from_v3(cast("str | Mapping[str, object]", data_type))
        if isinstance(data_type, (str, Mapping))
        else None
    )
    if np_dtype is not None:
        item_size: int | None = np_dtype.itemsize
    elif isinstance(data_type, str) and data_type in _VARIABLE_LENGTH_DATA_TYPES:
        item_size = _VARIABLE_LENGTH_NOMINAL_ITEM_SIZE
    else:
        item_size = None
    chunk_shape = (
        guess_chunks(shape, item_size) if item_size is not None else tuple(max(n, 1) for n in shape)
    )
    return {"name": REGULAR_CHUNK_GRID_NAME, "configuration": {"chunk_shape": chunk_shape}}


def auto_codecs(data: object) -> tuple[CodecLike, ...]:
    """
    The codecs of `data`, when it has them, or else the codecs `zarr-metadata`
    derives for its `dtype` in its default scope (those `from_array` gives a
    numpy array): a `bytes` codec, with an explicit endian for a multi-byte
    type, or `vlen-utf8` for numpy's variable-length strings.
    """
    if hasattr(data, "codecs"):
        return tuple(
            codec if isinstance(codec, str) else _as_named_config(codec)
            for codec in _as_tuple(read_attribute(data, "codecs"))
        )
    dtype = (
        np.dtype(as_dtype_like(read_attribute(data, "dtype"))) if hasattr(data, "dtype") else None
    )
    if dtype is not None:
        # the codecs a spec of this data type holds once validated (which drops
        # the no-op endian of a single-byte type)
        try:
            spec = ArraySpec.create_default(data_type=parse_dtype_v3(dtype))
        except ValueError:
            pass
        else:
            return tuple(spec.codecs)
    if dtype is not None and dtype.itemsize > 1:
        # the zarr v3 spec requires the ``endian`` configuration for multi-byte data types
        return ({"name": BYTES_CODEC_NAME, "configuration": {"endian": "little"}},)
    return ({"name": BYTES_CODEC_NAME},)


def auto_storage_transformers(data: object) -> tuple[AnyNamedConfig, ...]:
    if hasattr(data, "storage_transformers"):
        return tuple(
            _as_named_config(transformer)
            for transformer in _as_tuple(read_attribute(data, "storage_transformers"))
        )
    return ()


def _as_dimension_name(value: object) -> str | None:
    """
    `value`, a dimension name read from a duck-typed array, unchanged.

    Raises
    ------
    TypeError
        If `value` is not a string or `None`.
    """
    if value is None or isinstance(value, str):
        return value
    msg = f"Expected a dimension name (a string or None), got {value!r}."
    raise TypeError(msg)


def auto_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if hasattr(data, "metadata"):
        metadata = read_attribute(data, "metadata")
        if hasattr(metadata, "dimension_names"):
            names = read_attribute(metadata, "dimension_names")
            if names is None:
                return None
            return tuple(_as_dimension_name(name) for name in _as_tuple(names))
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
