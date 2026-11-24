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
from typing_extensions import TypedDict, TypeIs, TypeVar, deprecated
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
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
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
    LIKE_DEPRECATION,
    DocumentReader,
    ExtraFieldsBase,
    IncEx,
    NodeView,
    PatchOperation,
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
    node_patch,
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


# The BaseModel arm is SerializeAsAny so that an attributes model in a spec
# with an unnarrowed attributes type (a bare ArraySpec or GroupSpec) is
# serialized with the model's own fields. As a plain BaseModel it would be
# serialized through the base schema, which silently dumps {}.
type TBaseAttr = Mapping[str, object] | SerializeAsAny[BaseModel]
# BaseGroupSpec is not a member type. A GroupSpec is a complete hierarchy,
# and a group with unknown members is never part of one.
type TBaseItem = Union["GroupSpec", "ArraySpec"]
# The widest members mapping: any names, each member a GroupSpec or an ArraySpec.
type TBaseMembers = Mapping[str, TBaseItem]

# Aliases for specs of any specialization. The spec type parameters are
# covariant (specs are frozen, so a narrowed spec is a subtype of a wider one).
# These aliases are therefore ordinary wide types, not gradual-typing escapes:
# reads from a value typed with one are still checked as TBaseAttr or
# TBaseMembers. The public signatures of heterogeneous APIs (`resolve_path`,
# `from_flat`, `diff`, `is_array_spec`) use these names. They are equal to the
# bare class names.
type AnyArraySpec = "ArraySpec[TBaseAttr]"
type AnyGroupSpec = "GroupSpec[TBaseAttr, TBaseMembers]"
type AnyBaseGroupSpec = "BaseGroupSpec[TBaseAttr]"

# TAttr and TMembers are covariant because the specs are frozen. They appear
# only in read positions on instances, so ArraySpec[CFAttrs] can be used where
# ArraySpec[TBaseAttr] is expected. TBaseMembers is both the bound and the
# default of TMembers. Because TBaseItem refers to the bare GroupSpec, pydantic
# also validates a bare GroupSpec's nested members as specs at every depth.
# tests/typing_conformance.py checks the contract for narrowed parameters.
#
# TMembers is the type of the whole members mapping, not of one member. This
# lets one parameterization fix the member names as part of the schema:
# GroupSpec[Attrs, Members], with Members a closed TypedDict. A group that only
# constrains the type of its members is GroupSpec[Attrs, Mapping[str, MyArraySpec]].
# Constructing classmethods accept TBaseAttr instead of TAttr, because a
# covariant parameter may not appear in argument positions. Validation still
# enforces the class's real attributes type.
TAttr = TypeVar("TAttr", bound=TBaseAttr, default=TBaseAttr, covariant=True)
TMembers = TypeVar("TMembers", bound=TBaseMembers, default=TBaseMembers, covariant=True)
TDefault = TypeVar("TDefault")

NodeType = ZarrV3NodeType

type ZarrV3StoredDocumentJSON = ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON
"""Any document a Zarr V3 hierarchy stores: a node's `zarr.json`."""

# The documents a store contains for a node, keyed by store key. The keys are
# not identifiers, so these use the functional syntax. The array type is
# closed. The group type allows only its members' documents in addition to its
# own. So each type is assignable to a `Mapping[str, ...]` of its value types.
ZarrV3ArrayStoredJSON = TypedDict(
    "ZarrV3ArrayStoredJSON", {"zarr.json": ZarrV3ArrayMetadataJSON}, closed=True
)
"""A Zarr V3 array's metadata as it is stored: its `zarr.json` document."""
ZarrV3GroupStoredJSON = TypedDict(
    "ZarrV3GroupStoredJSON",
    {"zarr.json": ZarrV3GroupMetadataJSON},
    extra_items=ZarrV3StoredDocumentJSON,
)
"""
A Zarr V3 group's metadata as it is stored: the group's own `zarr.json`
document, plus each member's document at its metadata key. A member's metadata
key is its hierarchy path relative to the group, with no leading `/`, followed
by `/zarr.json`. For example, the key for the member at `/a/b` is
`"a/b/zarr.json"`.
"""

FillValue = JSONValue
"""
The JSON form of a fill value, as stored in a spec. Any JSON value is accepted
at the field level. Which values are valid depends on the array's data type.
That data type's definition validates the fill value when the whole metadata
document is validated. `zarr-metadata` exports a fill value type for each data
type (``Float32FillValue`` and so on) for callers that want the narrow forms.
"""

FillValueLike = FillValue | np.generic | complex
"""
A fill value accepted as input (`ArraySpec.create_default`,
`ArraySpec.replace`, `ArraySpec.from_array`). This is its JSON form, a numpy
scalar, or a Python complex number. Validation converts the value to JSON
form: non-finite floats become `"NaN"`, `"Infinity"` or `"-Infinity"`, and
complex numbers become pairs.
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

    `zarr_metadata.ZarrV3NamedConfigJSON` is the canonical type for this shape.
    This local copy exists because that type marks its members `ReadOnly`, and
    pydantic emits an advisory warning about that at every schema build (each
    time a model is parametrized or subclassed).
    """


CodecLike = str | AnyNamedConfig | ZarrV3NamedConfigJSON
"""
A codec declaration accepted as input: an extension name, or a named
configuration object. TypedDict keys are invariant, so typed codec objects
(`BytesCodecObject`, `ZstdCodecObject`, ...) are not assignable to the mutable
`AnyNamedConfig`. They are assignable to `zarr_metadata`'s
`ZarrV3NamedConfigJSON`, whose members are read-only, so that type is included.
A spec stores every codec as an object (`AnyNamedConfig`), because validation
converts a bare name to ``{"name": ...}``.
"""

ChunkGridLike = RegularChunkGridMetadata | AnyNamedConfig | ZarrV3NamedConfigJSON
"""
A chunk grid declaration accepted as input (`ArraySpec.create_default`,
`ArraySpec.replace`, `ArraySpec.from_array`). This is either the regular grid
that the core spec defines, typed exactly, or any other grid as a named
configuration. The grid's definition in the scope validates it when the
document is validated. TypedDict keys are invariant, so the exact regular-grid
type and the read-only `ZarrV3NamedConfigJSON` are included to keep typed grid
objects (such as `zarr_metadata`'s `RegularChunkGridObject`) assignable. A spec
stores the grid as an `AnyNamedConfig`.
"""

ChunkKeyEncodingLike = (
    DefaultChunkKeyEncodingMetadata
    | V2ChunkKeyEncodingMetadata
    | AnyNamedConfig
    | ZarrV3NamedConfigJSON
    | str
)
"""
A chunk key encoding declaration accepted as input. This is one of the two
encodings that the core spec defines, typed exactly (as an object or a bare
name), or any other encoding as a named configuration or a bare name. Typed
encoding objects are accepted through the read-only `ZarrV3NamedConfigJSON`.
The encoding's definition in the scope validates it when the document is
validated. A spec stores the encoding as an `AnyNamedConfig`.
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
        context does not provide one. This makes the scope a property of the
        model rather than of each ``model_validate`` call. A subclass that
        sets it keeps its scope through direct construction and through every
        derived operation (``replace``, ``create_default``, ...),
        because those re-validate as the class. `diff` reads each node's
        document in its class's scope too. An explicit validation context
        takes precedence. ``None`` (the default) means
        ``zarr-metadata``'s ``CORE_AND_EXTENSIONS_CONTEXT``. A group's members
        are validated as their own classes, so a scoped tree should narrow its
        member types to classes that set a scope. An explicit context reaches
        the members either way, because pydantic passes it to nested
        validation.
    zarr_scope_strict: ClassVar[bool]
        When true, validation also rejects every extension-point name that
        nothing in the active scope claims (that is, recognizes). The
        extension points are the data type, chunk grid, chunk key encoding,
        codecs (including a sharding codec's inner and index pipelines) and
        storage transformers. By default such names are not validated.
        Strict validation also rejects every extra field whose key nothing
        in the scope claims. Together with `zarr_scope`, this makes a class a
        conformance contract. For example, a strict class with
        ``zarr_scope = CORE_CONTEXT`` accepts exactly the documents built
        from names that the core spec defines. The variable-length data
        types and their codecs are registered extensions, not core, so a
        strict core class rejects them. `ArraySpec`'s document validation
        enforces this setting. A group document has no extension points, and
        its members are validated as their own classes.

    Notes
    -----
    A metadata document may contain members that the Zarr V3 spec does not
    define (its "additional fields"). A spec accepts the ones that the Zarr V3
    spec permits, and validates them in the same scope as the rest of the
    document:

    - If a definition in the scope claims the extra field's key, that
      definition validates the field. For example, ``CORE_CONTEXT`` claims a
      group's ``consolidated_metadata``.
    - Any other extra field is one the reader does not recognize. The Zarr V3
      spec allows it only as an object with ``"must_understand": false``,
      which is kept as written.
    - Every other unknown key is rejected. This also catches a misspelled
      field name.

    Accepted extra fields are stored in the `extra_fields` mapping (see
    `ExtraFieldsBase`). They are written back as top-level members wherever
    the document is written: ``model_dump``, ``model_dump_json``,
    ``replace``, ``diff`` and ``to_stored_metadata_json``.
    """

    zarr_format: Literal[3] = 3
    zarr_scope: ClassVar[ZarrV3Context | None] = None
    zarr_scope_strict: ClassVar[bool] = False

    def __replace__(self, **changes: object) -> Self:
        """
        Support `copy.replace(spec, **changes)` (Python 3.13 and later).

        This calls `self.replace`, so the changes are validated, and a
        subclass that overrides `replace` is respected. Pydantic's
        `BaseModel.__replace__` copies without validating.
        """
        # `Any`: each subclass's `replace` accepts its own TypedDict of changes,
        # which no signature declared here can name. `replace` validates `changes`.
        return cast("Any", self).replace(**changes)

    @classmethod
    def _scope(cls, context: object) -> ZarrV3Context | None:
        """
        Return the scope to validate metadata in for this class.

        This is the validation context's scope if it has one, and this class's
        `zarr_scope` otherwise. `None` means the readers' default
        (``CORE_AND_EXTENSIONS_CONTEXT``).
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
        Reject what whole-document validation leaves unchecked but the spec or
        this class does not allow.

        Validation happens in `scope` (``CORE_AND_EXTENSIONS_CONTEXT`` when
        `None`). This raises `ValueError` for:

        - extra fields that break the spec's rule for fields a reader does not
          recognize;
        - on a class with `zarr_scope_strict` set, every extra field and every
          extension-point name (`unclaimed_extension_points`) that nothing in
          `scope` claims.

        Extra fields that a definition in `scope` claims are not checked here.
        That definition validates them when the whole document is read.
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

    Only numpy data types are converted. A numpy data type maps to its Zarr V3
    name, or to a named configuration object for the data types whose Zarr V3
    form requires one. The names match what zarr-python writes:

    - ``datetime64`` and ``timedelta64`` become ``numpy.datetime64`` and
      ``numpy.timedelta64``.
    - A fixed-length unicode type (``<U5``) becomes ``fixed_length_utf32``.
    - A fixed-length bytes type (``S3``) becomes ``null_terminated_bytes``.
    - numpy's variable-length ``StringDType`` becomes ``string``.
    - An unstructured void type (``V4``) becomes ``raw_bytes`` if `scope`
      claims that name (zarr-python's scope does), and the core spec's raw
      type (``r32``) otherwise.

    A declaration already in Zarr V3 form (a data type name or a named
    configuration mapping) is returned unchanged. Like every other field, it
    is validated against the definitions in `scope` when the whole metadata
    document is validated. If `scope` is `None`, that is ``zarr-metadata``'s
    ``CORE_AND_EXTENSIONS_CONTEXT``.

    One check runs here instead of at document validation. A string that names
    no data type in the scope, but that numpy parses as a data type (``"i8"``,
    ``"float"``, ``">f4"``), is ambiguous. It is far more likely a numpy name
    than an unknown extension, so this raises `ValueError`, and suggests the
    Zarr V3 name if one exists.

    The names zarr-python has stored for the variable-length data types are
    accepted in any scope, so documents that zarr-python wrote pass through.
    These are ``"str"``, which no scope claims, and
    ``"variable_length_bytes"``, which only ``ZARR_PYTHON_CONTEXT`` claims.
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
            # Not a numpy name either, so treat it as an unknown extension data
            # type. Document validation checks it in the scope.
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
        # numpy time dtypes use the named-configuration form in Zarr V3. They are
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
"""The numpy type code and code-unit size of each fixed-length data type that
`parse_dtype_v3` writes as a named configuration with `length_bytes`."""


def _numpy_dtype_from_v3(data_type: str | Mapping[str, object]) -> np.dtype[np.generic] | None:
    """
    Return the numpy data type equivalent to a Zarr V3 data type declaration.

    This is a best-effort inverse of `parse_dtype_v3`. It returns `None` if
    there is no equivalent (for example, for an unknown extension data type).
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
            # A name is an equivalent only if its numpy data type maps back to
            # the same Zarr V3 name. np.dtype accepts some names that mean
            # something else entirely (for example, the V3 variable-length "bytes").
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
    Return whether the values of `data_type` are stored in single bytes.

    The `bytes` codec's endian does not apply to such data types. The answer
    comes from the data type's definition in `scope` (`zarr-metadata`'s
    `storage_of`). Single-byte data types include the single-byte numbers,
    `null_terminated_bytes`, `raw_bytes` and `r8`. This returns `False` for a
    data type that the scope does not know, or that it cannot derive a default
    document for.
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


# The names of the variable-length data types. "string" and "bytes" are the
# names that zarr-extensions registers. "variable_length_bytes" is zarr-python's
# name for variable-length bytes. "str" is the name zarr-python 3.0 stored for
# the string data type (renamed "string" in 3.1). Documents that zarr-python
# wrote may still contain these names.
#
# A name in this set skips the ambiguous-numpy-name check in `parse_dtype_v3`
# (numpy parses "str" as a zero-length unicode type). It also gets a nominal
# element size, so `create_default(chunk_grid="auto")` can guess chunks. Fill
# values and codecs for these data types are not derived here. `zarr-metadata`
# derives them for the names that a scope claims.
_VARIABLE_LENGTH_DATA_TYPES: frozenset[str] = frozenset(
    {STRING_DATA_TYPE_NAME, "str", BYTES_DATA_TYPE_NAME, VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME}
)
# Chunk guessing needs an element size. Variable-length elements have none,
# so use the size of a pointer, as numpy object arrays do.
_VARIABLE_LENGTH_NOMINAL_ITEM_SIZE = 8

_DEFAULT_CHUNK_KEY_ENCODING: DefaultChunkKeyEncodingObject = {
    "name": DEFAULT_CHUNK_KEY_ENCODING_NAME,
    "configuration": {"separator": DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR},
}
"""
The chunk key encoding that `create_default` gives an array. This is the
`default` encoding with its `/` separator written out, as zarr-python writes
it. So a spec built here compares equal to one read from a store that
zarr-python wrote. Specs compare fields literally, and `zarr-metadata` writes
the same encoding as ``{"name": "default"}``.
"""


DTypeLike = str | AnyNamedConfig
"""
A data type declaration in Zarr V3 form (name or named configuration).
`ArraySpec` also accepts numpy data types as input. It converts them with
`parse_dtype_v3` in a model validator, so the conversion can use the scope
from the class's `zarr_scope` as well as from the validation context.
"""


def _unclaimed_extension_point_names(
    document: Mapping[str, object], scope: ZarrV3Context
) -> list[str]:
    """
    Return the extension-point names in a dumped array metadata document that
    nothing in `scope` claims.

    Each name is returned as a ``<loc>: <message>`` line. Document validation
    does not check these names, but a strict class rejects them. The codec
    walk includes a sharding codec's inner and index pipelines.
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
    Return the extra fields of a node document that the Zarr V3 spec's rules
    for additional fields reject in `scope`.

    Each field is returned as a ``<loc>: <message>`` line.

    If a definition in `scope` claims an extra field's key, the reader
    recognizes that field. That definition validates it when the whole document
    is read, so it is skipped here. The reader does not recognize any other
    extra field. The spec allows such a field only if it explicitly waives
    understanding: "An implementation MUST fail to open Zarr groups or arrays
    if any metadata fields are present which (a) the implementation does not
    recognize and (b) are not explicitly set to ``"must_understand": false``."
    A waived field is an object with ``"must_understand": false``. It is kept
    as written, whatever else it contains. This is how `zarr-metadata`
    interprets the rule, and what its strategies generate.
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
    Return the extra fields of a node document whose key nothing in `scope`
    claims.

    Each field is returned as a ``<loc>: <message>`` line. A reader in `scope`
    does not recognize these fields. A strict class rejects them, even if they
    set ``"must_understand": false``.
    """
    return [
        f"{key}: no {kind.label} in the scope claims {key!r}"
        for key in extra_fields
        if scope.claimant(kind, key) is None
    ]


def canonical_envelope(declaration: object, *, bare_name: bool = False) -> object:
    """
    Return a declaration in the canonical envelope form that `zarr-metadata`
    writes.

    The rules are:

    - A bare name becomes ``{"name": ...}``.
    - An explicit ``"must_understand": true`` is dropped, because leaving it
      out means true.
    - An empty ``configuration`` is dropped.
    - JSON arrays in a configuration become tuples (`tuplify_json`), as a spec
      stores every other sequence. So a document read from JSON compares equal
      to one built in Python.
    - With `bare_name` (for data types), an object that contains only a name
      becomes that bare name. Core data types have been written this way since
      Zarr v3.0.

    Any other value is returned unchanged, for validation to reject.
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
    Reject a single codec where a pipeline is expected.

    Iterating a lone mapping or string would validate its keys or characters
    as codec names.
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
The codec pipeline stored in a spec: one or more codec objects. Bare codec
names are converted to objects before the field is validated (see
`ArraySpec.canonicalize_declarations`).
"""


class ArraySpecCreate(TypedDict, total=False):
    """
    The fields that `ArraySpec.create_default` accepts as keyword arguments.

    This is every `ArraySpec` field except `chunk_grid`. `create_default`
    accepts `chunk_grid` as a separate parameter, because it also accepts
    `"auto"`.
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
        empty dict, because the `attributes` key is optional in the Zarr V3
        spec. The default is validated against `TAttr`, so a model whose
        attributes type does not accept the empty dict still requires them.
    shape: Sequence[int]
        The shape of this array.
    data_type: DTypeLike
        The data type of this array: a data type name, or a named
        configuration for a data type that takes parameters.
    chunk_grid: NamedConfig
        The chunk grid declaration. This can be any chunk grid that the scope
        claims. The grid's definition validates it when the document is
        validated. The core spec defines the regular grid, and `zarr-metadata`
        also defines the rectilinear one.
    chunk_key_encoding: NamedConfig
        The chunk key encoding declaration. This can be any chunk key encoding
        that the scope claims, and it is validated the same way.
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
        Normalize the document's declarations to the canonical envelope form.

        The spec allows several envelope forms for each declaration. This
        validator accepts all of them and converts them to one canonical form.
        So specs built from different forms compare equal to each other, and to
        the spec read back from a store.

        The envelope rules follow `zarr-metadata`'s canonical form:

        - A bare name string becomes the object ``{"name": ...}``. The
          exception is a data type with nothing to configure, which is written
          as its bare name.
        - An explicit ``"must_understand": true`` is dropped, because leaving
          it out means true.
        - An empty ``configuration`` mapping is dropped.

        A numpy `data_type` is converted to its Zarr V3 form with
        `parse_dtype_v3`, in the scope of this validation. This is a model
        validator rather than field metadata, so the class's `zarr_scope` can
        provide the scope when the validation context has none.

        This validator also applies one of zarr-python's rules. The `bytes`
        codec's `endian` configuration has no effect on a single-byte data
        type. So it is dropped when the chunk that the `bytes` codec receives
        has a single-byte data type, according to the data type's definition
        in the scope. The chunk's data type is the array's data type, as
        changed by the array -> array codecs before `bytes` in the pipeline:

        - `cast_value` sets it to its configured `data_type`. So after a cast
          to a multi-byte type, the endian is kept.
        - `transpose` and `scale_offset` keep it.
        - After any other codec it is unknown, so the endian is kept.

        The rule also applies inside a sharding codec's inner pipeline. The
        inner pipeline starts with the data type that the sharding codec
        receives at its position in the outer pipeline. The sharding codec's
        index codecs encode the uint64 index, so they keep their endian.
        This rule is needed because `zarr-metadata`'s canonical document model
        does not treat the two endian forms as equal, and `==` on specs
        compares fields literally.
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
            # the field validator rejects a lone codec, so leave it alone
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
            Normalize one codec, given the data type of the chunk it receives.

            Return the codec and the data type of the chunk it passes on.
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
            # An array -> array codec that this walk does not know may change the
            # data type. So the codecs after it get an unknown data type and keep
            # their endian. zarr-metadata's pipeline reader does not guess either.
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

        The document built from this spec's fields is validated in a scope.
        The fill value is checked against the data type, the chunk grid
        against the shape, and the codecs as a pipeline. Each extension point
        is validated by the definition that claims its name. A name that
        nothing in the scope claims is not checked, so unknown extensions
        pass. A claimed name with an invalid configuration is rejected.

        The scope comes from the pydantic validation context, as a
        `zarr_metadata` `ZarrV3Context`, either directly or under the key
        `pydantic_zarr.v3.CONTEXT_KEY`. If the context has no scope, this
        class's `zarr_scope` is used. If that is `None` too, the scope is
        `zarr_metadata`'s `CORE_AND_EXTENSIONS_CONTEXT`.

        Extra fields (members the spec does not define) are validated in the
        same scope. If a definition in the scope claims an extra field's key,
        that definition validates it. Any other extra field must follow the
        spec's rule for fields a reader does not recognize: it must be an
        object with ``"must_understand": false``, and it is kept as written.

        On a class with `zarr_scope_strict` set, names that the scope does not
        claim are rejected. Every extension-point name and extra-field key in
        the document must be claimed by the active scope.

        The `attributes` field is not part of this check. Only this class's
        `TAttr` parameter governs its type and contents.
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
        Return this spec, without attributes, as a validated `zarr-metadata` document model.

        With no `context`, the document is validated in the class's
        `zarr_scope`, or in the readers' default scope if that is `None`. So a
        call without arguments, such as `like`'s canonical comparison, uses the
        class's own scope.

        Raises `ValueError` listing every problem if the document is invalid.
        """
        if context is None:
            context = type(self).zarr_scope
        document = self.model_dump()
        document["attributes"] = {}
        try:
            return ZarrV3ArrayMetadataModel.from_json(document, context=context)
        except MetadataValidationError as e:
            raise ValueError(format_problems(e)) from e

    # No return annotation, as on `ExtraFieldsBase`. Pydantic would use it as
    # the serialization schema instead of the fields' schema.
    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
        """
        Omit `dimension_names` when it is `None`.

        The Zarr V3 spec allows the key to be absent, but not null. This is a
        model serializer rather than a `model_dump` override, so the omission
        also applies to arrays nested as members in a group's dump, and to JSON
        serialization. Like every spec's serializer, it writes the extra fields
        as top-level members.
        """
        document = flatten_extra_fields(handler(self))
        if document.get("dimension_names") is None:
            document.pop("dimension_names", None)
        return document

    def to_metadata_json(self) -> ZarrV3ArrayMetadataJSON:
        """
        Return this array's Zarr V3 metadata document: the content of its `zarr.json`.

        Unlike `model_dump`, this accepts no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV3ArrayMetadataJSON`. The document
        is what `model_dump()` returns. `dimension_names` is omitted when it is
        `None`, an attributes model is dumped by alias, and `extra_fields` are
        written as top-level members.
        """
        return cast("ZarrV3ArrayMetadataJSON", self.model_dump())

    def to_stored_metadata_json(self) -> ZarrV3ArrayStoredJSON:
        """
        Return this array's Zarr V3 metadata as it is stored.

        The result maps the array's one store key, `zarr.json`, to
        `to_metadata_json()`. `from_stored_metadata_json` reads it back.
        """
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: self.to_metadata_json()}

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read an array from its stored metadata.

        `data` maps store keys to parsed JSON documents, in the form that
        `to_stored_metadata_json` returns. Raises `ValueError` if `data`
        contains no array `zarr.json` at the root, or contains documents for
        any other node. Raises `ValidationError` if the document is not a
        valid array.
        """
        return cls.model_validate(stored_array(_stored_nodes(data)))

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.generic],
        *,
        # TBaseAttr, not TAttr, because covariant parameters may not appear in
        # argument positions. Validation enforces the class's attributes type.
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
            object that conforms to the numpy array API. The shape and data
            type come from the array. With `"auto"`, the other fields are
            read from the array where it provides them:

            - the chunk grid is a regular grid of `chunk_shape`, or of `shape`
              if the array has no `chunk_shape`
            - the attributes, fill value, codecs and storage transformers are
              read from attributes of the same name
            - the dimension names are read from `metadata.dimension_names`
            - the chunk key encoding is always the `default` encoding with a
              `/` separator

            If the object has no fill value or codecs (a numpy array has
            neither), they get the defaults that `zarr-metadata` derives for
            its data type in this class's scope, as in `create_default`.
            Keyword arguments other than `"auto"` override what the array
            provides. To model a stored array, pass its stored metadata
            document to `from_stored_metadata_json` instead.

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
        # Use the array's fill value and codecs if it has them. An array without
        # them (a numpy array) gets the defaults that zarr-metadata derives for
        # its data type in this class's scope, as in `create_default`.
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
        # The values above have the input types. `model_validate` converts
        # them to the types that a spec stores.
        return cls.model_validate({**derived, **document})

    @classmethod
    def create_default(
        cls,
        *,
        chunk_grid: ChunkGridLike | Literal["auto"] | None = None,
        **fields: Unpack[ArraySpecCreate],
    ) -> Self:
        """
        Create an `ArraySpec` with `fields` set and every other field set to a default.

        This is the counterpart of `zarr-metadata`'s
        `ZarrV3ArrayMetadata.create_default`, which derives the defaults.

        Each field you do not pass gets `zarr-metadata`'s default. Where that
        default depends on other fields, it is derived from the fields you
        pass:

        - The data type determines the default fill value (zero, `False`, the
          empty string, `"NaT"`) and the array -> bytes codec (`vlen-utf8` for
          `string`, `vlen-bytes` for `bytes`, and `bytes` otherwise).
        - The shape determines the chunk grid: one regular chunk that covers
          the array.
        - If you pass nothing, the array is a scalar `uint8`.

        Two defaults are pydantic-zarr's own choices:

        - The chunk key encoding is the `default` encoding with its `/`
          separator, as zarr-python writes it.
        - `chunk_grid="auto"` guesses a chunk shape from the array's size in
          bytes. zarr-python uses the same heuristic when no chunks are given.

        A numpy data type is converted to its Zarr V3 form first.

        On a subclass, a field of the array document that the class declares a
        default for is treated as if you passed it. For example, a class that
        pins `data_type: Literal["uint16"] = "uint16"` or
        `fill_value: Literal["NaN"] = "NaN"` keeps that value, and the other
        fields are derived from it. Two kinds of field get their own defaults
        instead: fields that a subclass adds, and fields whose default factory
        reads other fields. Nothing is derived from those fields. If such a
        field is the shape or the data type, you have to pass the fields that
        depend on it. In that case `chunk_grid="auto"`, which needs both the
        shape and the data type, raises `ValueError`.

        As with the constructor, you have to pass any field that the class
        requires and that cannot be derived (for example, an attributes model
        with required keys). A subclass that needs more than defaults (for
        example, attributes computed from the shape) can override this method.
        To keep the override compatible, keep the keyword-only signature and
        pass the computed fields to `super().create_default(...)`.

        All fields are validated together, in this class's scope. Only the
        shape, the data type and an explicit `fill_value` are used to derive
        other fields. If they conflict, each still derives what it can on its
        own, so the error points at the field that caused it.

        If nothing in the scope can derive a fill value for the data type (for
        example, an extension data type, or a name that only another scope
        claims), pass `fill_value`. The codecs are then derived. A fill value
        from a default factory that reads other fields does not count as
        passed. So for such a data type, pass `fill_value` or `codecs`.

        A data type name that is not a Zarr V3 name, but that numpy parses as
        some other type (`"i8"`, `"float"`), is likely a mistake. It raises
        `ValueError`, which suggests the Zarr V3 name.

        Examples
        --------
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> ArraySpec.create_default(shape=(10, 10), data_type="uint16", dimension_names=("y", "x"))
        ArraySpec(zarr_format=3, node_type='array', attributes={}, shape=(10, 10), data_type='uint16', chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10, 10)}}, chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}}, fill_value=0, codecs=({'name': 'bytes', 'configuration': {'endian': 'little'}},), storage_transformers=(), dimension_names=('y', 'x'))
        """
        # Treat a class-declared default of an array document field as passed,
        # so the other fields are derived from it rather than from
        # zarr-metadata's generic defaults. Leave fields that a subclass adds to
        # pydantic. Also leave a default whose factory reads other fields: it
        # runs at validation, and no derived value replaces it.
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
        # A field left to its factory has no value yet, so nothing is derived
        # from it. The caller has to pass the fields that depend on it.
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

        All changes are validated together. So related fields can change at
        the same time: for example, a new `data_type` with a matching
        `fill_value`, or a new `shape` with a matching `chunk_grid`. Unlike
        `model_copy(update=...)`, this validates the result. Passing
        `extra_fields` replaces all of the extra fields.

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
        Return a copy of this spec sharded with the `sharding_indexed` codec.

        The existing chunk shape becomes the shard shape. The existing codec
        pipeline moves inside the shards as the subchunk pipeline. The shard
        index uses the default encoding: little-endian `bytes` plus `crc32c`,
        at `index_location`. Document validation checks that the subchunk shape
        evenly divides the chunk shape.

        On a spec that is already sharded, this nests the shards (the sharding
        codec is recursive). Document validation then also checks that the new
        subchunk shape is divisible by the existing inner chunk shape.
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

    def diff(self, other: ArraySpec, *, semantic: bool = False) -> list[PatchOperation]:
        """
        Return the JSON Patch that turns this array's document into `other`'s.

        The patch follows RFC 6902 (https://www.rfc-editor.org/rfc/rfc6902).
        It turns the JSON document `self.model_dump()` into
        `other.model_dump()`. Each path is a JSON Pointer into that document,
        such as
        `"/fill_value"`. So the patch is empty exactly when the two
        specs have the same document. To apply the patch with a JSON Patch
        library, apply it to the document as JSON, with lists for tuples:
        `json.loads(json.dumps(self.model_dump()))`.

        With `semantic=True`, only the operations that change what the
        metadata means are returned, so the result is empty exactly when the
        two specs mean the same. A document that `zarr-metadata` cannot read
        in its class's `zarr_scope` cannot be judged, so every change to it
        is returned. `zarr-metadata` reads both documents to
        judge this. For example, a float32 fill value written `"NaN"` means
        the same as one written `"0x7fc00000"`. A change to attributes always
        changes the meaning, because attributes are not interpreted.

        Parameters
        ----------
        other : ArraySpec
            The Zarr V3 array spec to compare with.
        semantic : bool, default False
            Whether to return only the operations that change the meaning.

        Returns
        -------
        list[PatchOperation]
            The operations, in the order they apply.

        Raises
        ------
        TypeError
            If `other` is not a Zarr V3 node spec.
        ValueError
            If `other` is a group.

        Examples
        --------
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> a = ArraySpec.create_default(shape=(2,), data_type="float32", fill_value="NaN")
        >>> b = a.replace(fill_value="0x7fc00000", attributes={"units": "m"})
        >>> a.diff(b)
        [{'op': 'add', 'path': '/attributes/units', 'value': 'm'}, {'op': 'replace', 'path': '/fill_value', 'value': '0x7fc00000'}]
        >>> a.diff(b, semantic=True)
        [{'op': 'add', 'path': '/attributes/units', 'value': 'm'}]
        """
        return node_patch(self, _checked_node(other), _view, semantic=semantic)

    @deprecated(LIKE_DEPRECATION, category=DeprecationWarning)
    def like(
        self,
        other: ArraySpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Return whether this array spec matches `other`.

        Deprecated: use `diff`. `not a.diff(b)` checks that the two specs have
        the same document, and `not a.diff(b, semantic=True)` checks that they
        mean the same. To compare only some fields, filter the operations by
        their `path`.

        With no `include` or `exclude`, the metadata documents are compared by
        meaning, and the attributes as plain values. With `include` or
        `exclude`, the selected fields of `model_dump` are compared as plain
        values.
        """
        return self._like(other, include=include, exclude=exclude)

    def _like(
        self,
        other: ArraySpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """Make the comparison that `like` makes."""
        if include is None and exclude is None:
            # Compare the metadata documents by meaning rather than by form.
            # `zarr-metadata` canonicalizes the values it interprets. So a fill
            # value written "NaN" equals one written "0x7fc00000", and a codec
            # equals the same codec without a no-op configuration key.
            # Attributes are not interpreted, so they are compared separately,
            # as plain fields.
            return self._canonical_document() == other._canonical_document() and model_like(
                self, other, include={"attributes"}
            )

        return model_like(self, other, include=include, exclude=exclude)


def ensure_member_names_v3(data: Mapping[str, object]) -> Mapping[str, object]:
    """
    Ensure that every key of a members mapping is a valid Zarr V3 node name.

    `zarr-metadata`'s `validate_node_name_v3` checks the node-name rules of the
    Zarr V3 spec. A name must not contain ``"/"``, must not consist only of
    periods, and must not start with the reserved ``"__"`` prefix. The empty
    string names the root node, so it is not a valid *member* name, and this
    function rejects it as well.
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


class _GroupNode(NodeSpec, Generic[TAttr]):
    """
    What every Zarr V3 group spec has: the fields of a group's own metadata
    document, and its checks.

    `BaseGroupSpec` and `GroupSpec` both build on this class, but neither is a
    kind of the other. A `BaseGroupSpec` is one group's own metadata, and a
    `GroupSpec` is a hierarchy.
    """

    node_type: Literal["group"] = "group"
    attributes: TAttr = Field(default={}, validate_default=True)

    @model_validator(mode="after")
    def validate_metadata_document(self, info: ValidationInfo) -> Self:
        """
        Validate this spec's extra fields as part of a Zarr V3 group metadata
        document, via `zarr-metadata`.

        Apart from the fields this class declares, a group document contains
        only extra fields. So this check only runs on a spec that has extra
        fields. The scope is chosen as for `ArraySpec`.

        - If a definition in the scope claims an extra field's key, that
          definition validates the field. For example, ``CORE_CONTEXT`` claims
          ``consolidated_metadata``. Its contents are kept as written, and are
          not read as members.
        - Any other extra field must be an object with
          ``"must_understand": false``, and it is kept as written.

        On a class with `zarr_scope_strict` set, an extra field that nothing in
        the scope claims is rejected. Attributes and members are not part of
        this check.
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

    def _own_metadata(self) -> AnyBaseGroupSpec:
        """
        Return this group's own metadata as a `BaseGroupSpec`.

        The result contains the attributes and extra fields, without members.
        It is constructed without validation, because this spec is already
        validated. It may have been validated in a scope that a bare
        `BaseGroupSpec` would not use for its extra fields.
        """
        return BaseGroupSpec.model_construct(
            attributes=self.attributes, extra_fields=self.extra_fields
        )

    def to_metadata_json(self) -> ZarrV3GroupMetadataJSON:
        """
        Return this group's Zarr V3 metadata document: the content of its `zarr.json`.

        Unlike `model_dump`, this accepts no options, so its result has one
        static type, `zarr_metadata`'s `ZarrV3GroupMetadataJSON`. Its
        `extra_fields` (such as `consolidated_metadata`) are written as
        top-level members. A group's members are not part of its metadata
        document, so a `GroupSpec`'s `members` are left out.
        """
        return cast("ZarrV3GroupMetadataJSON", self.model_dump(exclude={"members"}))

    def _own_stored_metadata_json(self) -> ZarrV3GroupStoredJSON:
        """Return this group's own `zarr.json`, as `to_stored_metadata_json` writes it."""
        return {ZARR_V3_GROUP_METADATA_STORE_KEY: self.to_metadata_json()}


class BaseGroupSpec(_GroupNode[TAttr], Generic[TAttr]):
    """
    A model of a Zarr Version 3 Group's own metadata, without members.

    This is the form of a group in a flattened hierarchy (`to_flat` returns it
    for every group), and of a group loaded without traversing its children.
    It is not a kind of `GroupSpec`, and a `GroupSpec` is not a kind of
    it: a `GroupSpec` is a hierarchy. `GroupSpec.own_metadata` returns a
    hierarchy's root as a `BaseGroupSpec`.

    Attributes
    ----------
    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr, default = {}
        The user-defined attributes of this group. The default is the empty
        dict, because the `attributes` key is optional in the Zarr V3 spec.
        The default is validated against `TAttr`, so a model whose attributes
        type does not accept the empty dict still requires them.
    """

    def replace(self, **changes: Unpack[BaseGroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        Unlike `model_copy(update=...)`, this validates the result, so an
        invalid change raises a validation error. Passing `extra_fields`
        replaces all of the extra fields.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

    def to_stored_metadata_json(self) -> ZarrV3GroupStoredJSON:
        """
        Return this group's Zarr V3 metadata as it is stored.

        The result maps each store key to the JSON document stored there. A
        `BaseGroupSpec` has no members, so the mapping has one key,
        `zarr.json`, mapped to `to_metadata_json()`.
        """
        return self._own_stored_metadata_json()

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a group's own metadata from a stored layout.

        `data` maps store keys to parsed JSON documents, in the form that
        `to_stored_metadata_json` returns. Only the root's `zarr.json` is
        read. Members' documents are ignored. Raises `ValueError` if `data`
        contains no group `zarr.json` at the root. Raises `ValidationError`
        if the document is not a valid group.
        """
        own = {key: document for key, document in data.items() if "/" not in key}
        return cls.model_validate(stored_root(_stored_nodes(own), "group"))

    def diff(self, other: AnyBaseGroupSpec, *, semantic: bool = False) -> list[PatchOperation]:
        """
        Return the JSON Patch that turns this group's own metadata into `other`'s.

        The patch follows RFC 6902 (https://www.rfc-editor.org/rfc/rfc6902).
        It turns the JSON document `self.model_dump()` into
        `other.model_dump()`. Each path is a JSON Pointer into that document,
        such as `"/attributes/title"`. So the patch is empty exactly when the two specs
        have the same document. To apply the patch with a JSON Patch library,
        apply it to the document as JSON, with lists for tuples:
        `json.loads(json.dumps(self.model_dump()))`.

        With `semantic=True`, only the operations that change what the
        metadata means are returned, so the result is empty exactly when the
        two specs mean the same. `zarr-metadata` reads both documents, in their classes'
        `zarr_scope`, to judge this. A document it cannot read in that scope
        cannot be judged, so every change to it is returned. A change to attributes
        always changes the meaning, because attributes are not interpreted.

        Parameters
        ----------
        other : BaseGroupSpec
            The Zarr V3 group's own metadata to compare with.
        semantic : bool, default False
            Whether to return only the operations that change the meaning.

        Returns
        -------
        list[PatchOperation]
            The operations, in the order they apply.

        Raises
        ------
        TypeError
            If `other` is not a Zarr V3 node spec.
        ValueError
            If `other` is an array or a `GroupSpec`. A `GroupSpec` is a
            hierarchy, not one group's own metadata. To compare a hierarchy's
            root with this spec, diff `group.own_metadata()` with it.

        Examples
        --------
        >>> from pydantic_zarr.v3 import BaseGroupSpec
        >>> old = BaseGroupSpec(attributes={"title": "run 1"})
        >>> old.diff(BaseGroupSpec(attributes={"title": "run 2"}))
        [{'op': 'replace', 'path': '/attributes/title', 'value': 'run 2'}]
        """
        return node_patch(self, _checked_node(other), _view, semantic=semantic)

    @deprecated(LIKE_DEPRECATION, category=DeprecationWarning)
    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Return whether this group spec matches `other`.

        Deprecated: use `diff`. `not a.diff(b)` checks that the two specs have
        the same document, and `not a.diff(b, semantic=True)` checks that they
        mean the same. To compare only some fields, filter the operations by
        their `path`.

        Only the groups' own metadata is compared. If `other` is a `GroupSpec`,
        only its own metadata is used. The fields of `model_dump` selected by
        `include` and `exclude` are compared as plain values.
        """
        return self._like(other, include=include, exclude=exclude)

    def _like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """Make the comparison that `like` makes."""
        other_parsed = other._own_metadata() if isinstance(other, GroupSpec) else other
        return model_like(self, other_parsed, include=include, exclude=exclude)


class GroupSpec(_GroupNode[TAttr], Generic[TAttr, TMembers]):
    """
    A model of a Zarr Version 3 Group.

    The second type parameter is the type of the whole `members` mapping:

    - Leave it out for a group with any members.
    - Pass a `Mapping` to constrain the type of every member
      (`GroupSpec[Attrs, Mapping[str, MyArraySpec]]`).
    - Pass a closed `TypedDict` to fix the member names
      (`GroupSpec[Attrs, MyMembers]`).

    Attributes
    ----------

    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: TAttr
        The user-defined attributes of this group.
    members: TMembers, default = {}
        The members of this group: a mapping from member name to member, of
        type `TMembers`. With the default `TMembers`, every member is validated
        as an `ArraySpec` or a `GroupSpec`. With `Any`, the values are not
        validated. The spec stores its own `dict` copy of the mapping you pass.
        A `GroupSpec` is a complete hierarchy, so an empty mapping means the
        group has no members. A group with unknown members is a
        `BaseGroupSpec`, which has no `members` field and is never a member.
        The default is validated against `TMembers`, so a group whose members
        type requires a member still requires `members`.
    """

    # The default is cast so that a type checker does not infer TMembers from it.
    # An unparameterized GroupSpec() then has the default members type.
    members: TMembers = Field(default=cast("TMembers", {}), validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def _copy_members(cls, data: object) -> object:
        # This is a model validator, not a field validator, because pyrefly and
        # ty widen the `__init__` parameter of a field with a before validator to `Any`.
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

    def replace(self, **changes: Unpack[GroupSpecUpdate]) -> Self:
        """
        Return a revalidated copy of this model with `changes` applied.

        All changes are validated together, so `attributes` and `members` can
        change at the same time. Unlike `model_copy(update=...)`, this
        validates the result. Passing `extra_fields` replaces all of the extra
        fields.

        `copy.replace` (Python 3.13 and later) calls this method too (see
        `NodeSpec.__replace__`), so it validates as well.
        """
        return self._replaced(changes)

    def own_metadata(self) -> BaseGroupSpec[TAttr]:
        """
        Return this hierarchy's root as a `BaseGroupSpec`: the group's own
        metadata, without members.

        The result is not validated again, because this spec is already
        valid. It is a plain `BaseGroupSpec`, so it does not keep the class of
        a subclass of `GroupSpec`. To read it as a subclass of
        `BaseGroupSpec`, validate its `model_dump()` with that class.
        """
        return cast("BaseGroupSpec[TAttr]", self._own_metadata())

    def resolve_path(self, path: str) -> AnyArraySpec | AnyGroupSpec:
        """
        Return the node at a `/`-separated path of member names relative to this group.

        Examples of paths are `"a"` and `"a/b/c"`. Leading and trailing `/` are
        ignored. Use this method to resolve the relative node paths that
        metadata conventions use, such as OME-Zarr dataset paths, and GeoZarr
        `grid_mapping` and multiscale assets.

        Raises
        ------
        ValueError
            If the path contains no member names, or an empty one (`"a//b"`).
        KeyError
            If a segment fails: either a member name does not exist, or a
            segment descends into a node that is not a group. The error names
            the failing segment.
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
        Return the node at `path`, or `default` if the path does not resolve.

        This is `resolve_path`, except that it returns `default` instead of
        raising `KeyError` (for a missing member, or a non-group along the
        path). A malformed path still raises `ValueError`. This form suits
        pydantic validators, which turn `ValueError` into a validation error
        but let `KeyError` escape.
        """
        try:
            return self.resolve_path(path)
        except KeyError:
            return default

    def to_stored_metadata_json(self) -> ZarrV3GroupStoredJSON:
        """
        Return this hierarchy's Zarr V3 metadata as it is stored.

        The result maps each store key to the JSON document stored there. It
        contains this group's `zarr.json`, and the `zarr.json` of each member
        at any depth, at the member's metadata key (`"a/zarr.json"`,
        `"a/b/zarr.json"`). A key is the member's path in `to_flat` form
        (`"/a/b"`), with the leading `/` stripped and `/zarr.json` appended.
        This is how the Zarr V3 spec forms metadata keys from hierarchy paths
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L382-L388).
        These are the metadata documents that zarr-python writes. So to write
        the hierarchy, write each document as JSON at its key under the
        hierarchy's root. `from_stored_metadata_json` reads it back.
        """
        members: dict[str, ZarrV3StoredDocumentJSON] = {
            f"{path.removeprefix('/')}/{ZARR_V3_GROUP_METADATA_STORE_KEY}": node.to_metadata_json()
            for path, node in self.to_flat().items()
            if path
        }
        # Each member's document has the type's `extra_items` type. The cast is
        # for type checkers that do not yet type writes to `extra_items` (pyrefly).
        return cast("ZarrV3GroupStoredJSON", {**self._own_stored_metadata_json(), **members})

    @classmethod
    def from_stored_metadata_json(cls, data: Mapping[str, object]) -> Self:
        """
        Read a hierarchy from its stored metadata.

        `data` maps store keys to parsed JSON documents, in the form that
        `to_stored_metadata_json` returns. Each node's `zarr.json` is read,
        and the nodes are assembled with `from_flat`. A stored layout cannot
        distinguish a group whose members were not read from a group with no
        members. So every group is read as a `GroupSpec`, with the members
        that the layout contains.

        Raises `ValueError` if `data` contains any of the following:

        - no group `zarr.json` at the root;
        - a key with a leading `/` or an empty segment;
        - a key other than a `zarr.json`;
        - a node whose parent is not a group in `data` (a store contains no
          implicit groups).

        Raises `ValidationError` if a document is not a valid node. A document
        whose `node_type` is neither `"array"` nor `"group"` is validated as a
        group.
        """
        return cls.from_flat(stored_flat(_stored_nodes(data), ArraySpec, BaseGroupSpec))

    @classmethod
    def from_flat(cls, data: Mapping[str, AnyArraySpec | AnyGroupSpec | AnyBaseGroupSpec]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation.

        The flattened hierarchy is a `dict`. Its keys must be valid paths. Its
        values must be `ArraySpec` or `BaseGroupSpec` instances, as `to_flat`
        returns them. For a `GroupSpec` value, only its own metadata is used.
        Its members are ignored, because a group's members are the entries
        under its path.

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
        #                codecs=({'name': 'bytes', 'configuration': {'endian': 'little'}},),
        #                storage_transformers=(),
        #                dimension_names=None)})
        ```
        """
        from_flated = from_flat_group(data)
        return cls.model_validate(from_flated.model_dump())

    def to_flat(self, root_path: str = "") -> dict[str, AnyArraySpec | AnyBaseGroupSpec]:
        """
        Flatten this `GroupSpec`.

        This method returns a `dict` with string keys and values that are `BaseGroupSpec` or
        `ArraySpec`.

        The resulting `dict` contains this group's own metadata, as a `BaseGroupSpec`, under
        the key `root_path`. It also contains the flattened form of each member. Each member is
        under a key made by joining `root_path` and the member's name with a `/` character, and
        so on recursively for each sub-member. The entries are sorted by key length.

        Parameters
        ----------
        root_path : `str`, default = ''
            The root path. The keys in `self.members` will be
            made relative to `root_path` when used as keys in the result dictionary.

        Returns
        -------
        Dict[str, ArraySpec | BaseGroupSpec]
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

    def diff(self, other: AnyGroupSpec, *, semantic: bool = False) -> list[PatchOperation]:
        """
        Return the JSON Patch that turns this hierarchy's document into `other`'s.

        The patch follows RFC 6902 (https://www.rfc-editor.org/rfc/rfc6902).
        It turns the JSON document `self.model_dump()` into
        `other.model_dump()`. Each path is a JSON Pointer into that document,
        such as `"/members/x/fill_value"`. So the patch is empty exactly when the two specs
        have the same document. To apply the patch with a JSON Patch library,
        apply it to the document as JSON, with lists for tuples:
        `json.loads(json.dumps(self.model_dump()))`.

        The patch is built node by node. A member that only one side has is
        added or removed whole. So is a member that is a different kind on
        each side, such as an array on one side and a group on the other.
        Everything else is compared key by key, as
        `pydantic_zarr.core.json_patch` does.

        With `semantic=True`, only the operations that change what the
        metadata means are returned, so the result is empty exactly when the
        two specs mean the same. `zarr-metadata` reads each node's document, in its class's
        `zarr_scope`, to judge this. A document it cannot read in that scope
        cannot be judged, so every change to it is returned. For example, a
        float32 fill value written `"NaN"` means the same as one written
        `"0x7fc00000"`. A change to attributes
        always changes the meaning, because attributes are not interpreted.
        So does adding, removing or replacing a member.

        Parameters
        ----------
        other : GroupSpec
            The Zarr V3 hierarchy to compare with.
        semantic : bool, default False
            Whether to return only the operations that change the meaning.

        Returns
        -------
        list[PatchOperation]
            The operations, in the order they apply.

        Raises
        ------
        TypeError
            If `other` is not a Zarr V3 node spec.
        ValueError
            If `other` is an array or a `BaseGroupSpec`. A `BaseGroupSpec` is
            one group's own metadata, and says nothing about its members. To
            compare this hierarchy's root with it, diff `self.own_metadata()`
            with it.

        Examples
        --------
        >>> from pydantic_zarr.v3 import ArraySpec, GroupSpec
        >>> a = ArraySpec.create_default(shape=(2,), data_type="float32", fill_value="NaN")
        >>> b = a.replace(fill_value="0x7fc00000")
        >>> old = GroupSpec(members={"x": a, "y": a})
        >>> new = GroupSpec(members={"x": b}, attributes={"title": "run 1"})
        >>> old.diff(new)
        [{'op': 'add', 'path': '/attributes/title', 'value': 'run 1'}, {'op': 'replace', 'path': '/members/x/fill_value', 'value': '0x7fc00000'}, {'op': 'remove', 'path': '/members/y'}]
        >>> old.diff(new, semantic=True)
        [{'op': 'add', 'path': '/attributes/title', 'value': 'run 1'}, {'op': 'remove', 'path': '/members/y'}]
        """
        return node_patch(self, _checked_node(other), _view, semantic=semantic)

    @deprecated(LIKE_DEPRECATION, category=DeprecationWarning)
    def like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Return whether this group spec matches `other`.

        Deprecated: use `diff`. `not a.diff(b)` checks that the two specs have
        the same document, and `not a.diff(b, semantic=True)` checks that they
        mean the same. To compare only some fields, filter the operations by
        their `path`.

        If `other` is a `BaseGroupSpec` (a group without members), only the
        groups' own metadata is compared. Otherwise, with no `include` or
        `exclude`, the groups' own metadata is compared as plain values, and
        each member in the same way as these specs, so member arrays are
        compared by meaning. With `include` or `exclude`, the selected fields
        of `model_dump` are compared as plain values.
        """
        return self._like(other, include=include, exclude=exclude)

    def _like(
        self,
        other: AnyBaseGroupSpec | AnyGroupSpec,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """Make the comparison that `like` makes."""
        if not isinstance(other, GroupSpec):
            # A group without members is compared by its own metadata. Reduce
            # this group to its own metadata too, or the comparison would
            # always fail on the members field.
            return self._own_metadata()._like(other, include=include, exclude=exclude)

        if include is None and exclude is None:
            # Compare the groups' own metadata as plain fields, and each member
            # with that member's own `like`. So member arrays are compared by
            # the meaning of their metadata documents, as in `ArraySpec.like`.
            if not model_like(self, other, exclude={"members"}):
                return False
            if self.members.keys() != other.members.keys():
                return False
            for name, member in self.members.items():
                other_member = other.members[name]
                if isinstance(member, ArraySpec):
                    if not isinstance(other_member, ArraySpec) or not member._like(other_member):
                        return False
                elif isinstance(other_member, ArraySpec) or not member._like(other_member):
                    return False
            return True

        return model_like(self, other, include=include, exclude=exclude)


def _metadata_reader(
    model: type[ZarrV3ArrayMetadataModel | ZarrV3GroupMetadataModel],
    context: ZarrV3Context | None,
) -> DocumentReader:
    """Return a reader of documents as `model`, in `context`, for `diff`."""

    def read(document: Mapping[str, object]) -> object:
        try:
            return model.from_json(document, context=context)
        except MetadataValidationError:
            return None

    return read


def _checked_node(node: object) -> NodeSpec:
    """Return `node` if it is a Zarr V3 node spec. Raise `TypeError` otherwise."""
    if isinstance(node, ArraySpec | BaseGroupSpec | GroupSpec):
        return node
    kind = f"{type(node).__module__}.{type(node).__qualname__}"
    msg = f"Expected a Zarr V3 ArraySpec, GroupSpec or BaseGroupSpec, got {kind}"
    raise TypeError(msg)


def _view(node: NodeSpec) -> NodeView[NodeSpec]:
    """
    Describe a node for `diff`.

    A node's metadata is read in its class's `zarr_scope`.
    """
    if isinstance(node, ArraySpec):
        array_reader = _metadata_reader(ZarrV3ArrayMetadataModel, type(node).zarr_scope)
        return NodeView("array", array_reader, None)
    group = cast("AnyBaseGroupSpec | AnyGroupSpec", _checked_node(node))
    group_reader = _metadata_reader(ZarrV3GroupMetadataModel, type(group).zarr_scope)
    members = group.members if isinstance(group, GroupSpec) else None
    return NodeView("group", group_reader, members)


def _stored_nodes(data: Mapping[str, object]) -> StoredNodes:
    """
    Return each node's type and metadata document from a stored layout.

    `data` maps store keys to parsed JSON documents. The result is keyed by
    each node's key prefix (see `split_store_key`): `""` for the root, and
    `"a/b"` for the node at `/a/b`. The type is the document's `node_type` if
    that is `"array"` or `"group"`, and `None` otherwise. Raises `ValueError`
    for a malformed key, or for a key other than a node's `zarr.json`. The
    error names the key.
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
        and values that are either `BaseGroupSpec` or `ArraySpec` instances. For a `GroupSpec`
        value, only its own metadata is used, and its members are ignored.

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
    data : Dict[str, ArraySpec | GroupSpec | BaseGroupSpec]
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
            elif isinstance(value, BaseGroupSpec | GroupSpec):
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
    Return whether `value` has the shape of `AnyNamedConfig`.

    That is, a mapping with a string `name` and, if present, a mapping
    `configuration`.
    """
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("configuration", {}), Mapping)
    )


def _as_named_config(value: object) -> AnyNamedConfig:
    """
    Return `value`, a named configuration read from a duck-typed array, unchanged.

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
    Return `value`, a sequence read from a duck-typed array, as a tuple.

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
            # zarr-python's array metadata, whose `to_dict` returns the JSON form
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
    Return `declaration` as a Zarr V3 declaration.

    A valid declaration is a name, or a named configuration with no problems.
    An unknown key counts as a problem.

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
    """The members of an array document that `zarr-metadata` derives other members from."""

    shape: tuple[int, ...]
    data_type: ZarrV3DeclarationJSON
    fill_value: JSONValue


def dtype_default_fill_value(
    dtype: npt.DTypeLike | Mapping[str, object], *, scope: ZarrV3Context | None = None
) -> FillValue:
    """
    Return the default fill value for an array of data type `dtype`.

    This is the default that `zarr-metadata` derives for the data type in
    `scope` (or in its default scope, if `scope` is `None`). It is the same
    value zarr-python uses: zero, `False`, the empty string, null bytes, or
    `"NaT"` for a time data type. A numpy data type is converted to its Zarr
    V3 form first.

    Raises
    ------
    ValueError
        If nothing in the scope can derive a fill value for the data type (for
        example, an extension data type).
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
    Return the fill value of `data`.

    If `data` has no fill value, return the default that `zarr-metadata`
    derives for its `dtype` (see `dtype_default_fill_value`).
    """
    if hasattr(data, "fill_value"):
        # spec validation checks the value against the data type
        return cast("FillValue", read_attribute(data, "fill_value"))
    if hasattr(data, "dtype"):
        return dtype_default_fill_value(np.dtype(as_dtype_like(read_attribute(data, "dtype"))))
    raise ValueError("Cannot determine a default fill value for an object without a dtype.")


_DATA_TYPE_KEEPING_CODECS = frozenset({TRANSPOSE_CODEC_NAME, SCALE_OFFSET_CODEC_NAME})
"""
The array -> array codecs whose output chunk has the same data type as their
input chunk (both codec specs say so). `cast_value` sets the data type of its
output. After any other codec before the array -> bytes codec, the data type is
unknown.
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
    Return the array document members that `zarr-metadata` derives from `document`.

    The members are derived in `scope`. Members already in `document` are
    left out of the result.

    Only the members that others depend on are passed to `zarr-metadata`: the
    shape, the data type, and the fill value if `document` has one. Passing the
    fill value means a data type with no default fill value still gets its
    codecs. Every other member in `document` is left for validation to check,
    so an error in it points at that member.

    If `zarr-metadata` rejects these members together, each one still
    derives what it can on its own: the data type derives the fill value and
    codecs, and the shape derives the chunk grid. A member that is itself
    invalid derives nothing, and validation reports it.
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
        # A shape that zarr-metadata rejects still gets a grid of its rank, so
        # that validation reaches the shape and reports it.
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
    Return a regular chunk grid for an array of `shape` and `data_type`.

    The chunk shape is guessed from the array's size in bytes, as zarr-python
    guesses it. If the data type's element size is unknown (for example, an
    extension data type), the grid has one chunk that covers the array.
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
    Return the codecs of `data`.

    If `data` has no codecs, return the codecs that `zarr-metadata` derives
    for its `dtype` in its default scope. These are the codecs that
    `from_array` gives a numpy array: a `bytes` codec, with an explicit endian
    for a multi-byte type, or `vlen-utf8` for numpy's variable-length strings.
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
        # The codecs of a validated spec with this data type. Validation drops
        # the no-op endian of a single-byte type.
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
    Return `value`, a dimension name read from a duck-typed array, unchanged.

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

    Converts a `GroupSpec` or `ArraySpec` and a string into a `dict` with string keys and
    values that are `BaseGroupSpec` or `ArraySpec`.

    If the input is an `ArraySpec`, this function returns `{root_path: copy}`, where `copy` is
    a deep copy of the input.

    If the input is a `GroupSpec`, the resulting `dict` contains the input's own metadata, as a
    `BaseGroupSpec`, under the key `root_path`. It also contains the result of calling `to_flat`
    on each element of `node.members`. Each is under a key made by joining `root_path` and the
    member's name with a `/` character, and so on recursively for each sub-member. The entries
    are sorted by key length.

    Parameters
    ----------
    node : `GroupSpec` | `ArraySpec`
        The node to flatten.
    root_path : `str`, default = ''
        The root path. If `node` is a `GroupSpec`, then the keys in `node.members` will be
        made relative to `root_path` when used as keys in the result dictionary.

    Returns
    -------
    Dict[str, ArraySpec | BaseGroupSpec]
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


def normalize_group_metadata(
    document: JSONValue, *, normalize_null_consolidated_metadata: bool = True
) -> JSONValue:
    """
    Repair a stored Zarr V3 group metadata document that a writer got wrong.

    Call this before validation. It returns the repaired document. If the
    document needs no repair, or the value is not a Zarr V3 group document
    (for example, an array document), it is returned unchanged (the same
    object).

    pydantic-zarr's readers validate a document against the spec, so they
    reject a document that breaks the spec. Some writers break it in known,
    harmless ways. This function undoes those mistakes. Each repair runs only
    if its keyword enables it:

    normalize_null_consolidated_metadata :
        zarr-python 3.0 writes ``"consolidated_metadata": null`` into every
        group's ``zarr.json`` to say the group has no consolidated metadata.
        The spec requires this member to be an object when present, so the
        readers reject the null. When `True` (the default), a null
        ``consolidated_metadata`` in a group document is dropped, since its
        absence means the same thing.

    The input is not modified. Apply this function to each document of a
    stored-documents mapping before `from_stored_metadata_json`.

    Examples
    --------
    >>> from pydantic_zarr.v3 import GroupSpec, normalize_group_metadata
    >>> stored = {"zarr_format": 3, "node_type": "group", "consolidated_metadata": None}
    >>> normalize_group_metadata(stored)
    {'zarr_format': 3, 'node_type': 'group'}
    >>> GroupSpec.from_stored_metadata_json({"zarr.json": normalize_group_metadata(stored)}).members
    {}
    """
    if (
        normalize_null_consolidated_metadata
        and isinstance(document, Mapping)
        and document.get("zarr_format") == 3
        and document.get("node_type") == "group"
        and ZARR_V3_CONSOLIDATED_METADATA_KEY in document
        and document[ZARR_V3_CONSOLIDATED_METADATA_KEY] is None
    ):
        return {
            key: value
            for key, value in document.items()
            if key != ZARR_V3_CONSOLIDATED_METADATA_KEY
        }
    return document


def is_array_spec(node: object) -> TypeIs[AnyArraySpec]:
    """Return whether `node` is an `ArraySpec`, narrowing the type in both branches."""
    return isinstance(node, ArraySpec)


def is_group_spec(node: object) -> TypeIs[AnyGroupSpec]:
    """
    Return whether `node` is a `GroupSpec` (a group with members).

    This narrows the type in both branches. It returns `False` for a
    `BaseGroupSpec` that is not a `GroupSpec`, so use it to tell the two
    apart in a flattened hierarchy.
    """
    return isinstance(node, GroupSpec)


def is_base_group_spec(node: object) -> TypeIs[AnyBaseGroupSpec]:
    """
    Return whether `node` is a `BaseGroupSpec`: a group's own metadata,
    without members.

    This narrows the type in both branches. It returns `False` for a
    `GroupSpec`, which is a hierarchy, not one group's own metadata.
    """
    return isinstance(node, BaseGroupSpec)
