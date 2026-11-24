from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from functools import cache
from types import MappingProxyType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    Protocol,
    Self,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
    override,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
import typing_extensions
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    GetJsonSchemaHandler,
    ModelWrapValidatorHandler,
    PlainSerializer,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    Tag,
    TypeAdapter,
    WrapSerializer,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import NoDefault, is_typeddict

# pydantic resolves the `extra_fields` annotation at runtime
from zarr_metadata import JSONValue  # noqa: TC002

if TYPE_CHECKING:
    # Both pydantic import paths below re-export pydantic_core's sentinel.
    # Unlike them, this import resolves statically on every supported pydantic
    # version. `as MISSING` marks the re-export as explicit for strict checkers.
    from pydantic_core import MISSING as MISSING
else:
    try:
        # the MISSING sentinel is stable from pydantic 2.14
        from pydantic import MISSING
    except ImportError:
        # On pydantic 2.12 and 2.13 the sentinel is in the experimental module.
        # It is the same object, with the same behavior. It is re-exported here
        # so that users of AttributesBase have one import that works across the
        # supported pydantic range.
        from pydantic.experimental.missing_sentinel import MISSING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic.fields import FieldInfo
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema, SchemaSerializer

# The selection that pydantic's `model_dump` accepts as `include` or `exclude`
# (`pydantic.main.IncEx`), plus `None` for "no selection".
type _IncExSelection = (
    set[int]
    | set[str]
    | Mapping[int, _IncExSelection | bool]
    | Mapping[str, _IncExSelection | bool]
)
type IncEx = _IncExSelection | None

T = TypeVar("T")


@overload
def tuplify_json(obj: Mapping) -> Mapping: ...


@overload
def tuplify_json(obj: list) -> tuple: ...


def tuplify_json(obj: object) -> object:
    """
    Recursively converts lists within a Python object to tuples.
    """
    if isinstance(obj, list):
        return tuple(tuplify_json(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: tuplify_json(v) for k, v in obj.items()}
    else:
        return obj


@cache
def _attributes_serializer(spec: type[BaseModel]) -> SchemaSerializer:
    """
    Return a serializer for the `attributes` field of `spec`.

    The serializer is built from the field's annotation alone. It serializes
    values the same way the field does, but without the field serializer that
    calls it. Cached per class.
    """
    return TypeAdapter(spec.model_fields["attributes"].rebuild_annotation()).serializer


def serialize_attributes(
    spec: type[BaseModel],
    value: object,
    handler: SerializerFunctionWrapHandler,
    info: SerializationInfo,
) -> object:
    """
    Serialize the attributes of a node of class `spec`, with aliases by default.

    Aliases are used unless the dump passes `by_alias=False`. The aliases of
    an attribute model name stored metadata keys that are not Python
    identifiers (`_FillValue`, `image-label`, `proj:code`). Without this
    default, a hierarchy would be written with the Python field names. A
    spec's own fields declare no aliases, so the default only affects the
    attributes: an attribute model, or the models in an attributes mapping.

    The default is applied in the serializer, not in `model_dump`, so that
    every way of dumping a spec uses it: `model_dump`, `model_dump_json`, a
    `TypeAdapter`, and a spec contained in another model. The dump's other
    settings carry over. This includes the part of `include` and `exclude`
    that selects within the attributes.

    An attribute model is serialized with its own schema, as `SerializeAsAny`
    would do. Any other value is serialized as the `attributes` field
    declares.
    """
    if info.by_alias is not None:
        return handler(value)
    serializer = (
        value.__pydantic_serializer__
        if isinstance(value, BaseModel)
        else _attributes_serializer(spec)
    )
    # The casts bridge a gap in pydantic-core's stubs. They type the selection
    # that a dump passes on (`set[int | str] | dict[int | str, ...]`) wider than
    # the selection that `to_python` accepts (`set[int] | set[str] | Mapping[...]`).
    # At runtime both are the same value.
    return serializer.to_python(
        value,
        mode=info.mode,
        by_alias=True,
        include=cast("IncEx", info.include),
        exclude=cast("IncEx", info.exclude),
        context=info.context,
        exclude_unset=info.exclude_unset,
        exclude_defaults=info.exclude_defaults,
        exclude_none=info.exclude_none,
        exclude_computed_fields=info.exclude_computed_fields,
        round_trip=info.round_trip,
        serialize_as_any=info.serialize_as_any,
    )


class StrictBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # No return annotation: pydantic uses a wrap serializer's return annotation
    # as the field's serialization schema, which would replace the attributes' own.
    @field_serializer("attributes", mode="wrap", check_fields=False)
    def _serialize_attributes(  # noqa: ANN202
        self, value: object, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Dump an attribute model with its aliases by default. See `serialize_attributes`."""
        return serialize_attributes(type(self), value, handler, info)


EXTRA_FIELDS = "extra_fields"
"""The name of the field where `ExtraFieldsBase` stores a document's extra fields."""


def flatten_extra_fields(document: dict[str, object]) -> dict[str, object]:
    """
    Move the entries of a dumped spec's `extra_fields` to the document's top level.

    The metadata document stores extra fields as top-level members.
    """
    extra = document.pop(EXTRA_FIELDS, None)
    if isinstance(extra, Mapping):
        document.update(extra)
    return document


class ExtraFieldsBase(StrictBase):
    """
    A base class for specs of metadata documents with undeclared members.

    Members of a document that the spec class does not declare are called
    "extra fields". They are collected, by key, into the `extra_fields`
    mapping. `model_dump` and `model_dump_json` write them back as top-level
    members, so the document is written as it was read.

    Extra fields are stored in a field of their own rather than as pydantic
    extras (`extra="allow"`). This is because pydantic applies a model's
    `extra` setting to every TypedDict the model contains, which would open
    the closed TypedDicts of field and attribute types.

    Each subclass decides which extra fields a document may contain. Even if
    a subclass does not validate extra fields, `extra_fields` rejects a key
    that names a declared field.

    Attributes
    ----------
    extra_fields: Mapping[str, JSONValue], default = {}
        The members of the document that this class does not declare, by key.
        JSON arrays are stored as tuples.
    """

    extra_fields: Mapping[str, Annotated[JSONValue, BeforeValidator(tuplify_json)]] = Field(
        default={}
    )

    @model_validator(mode="before")
    @classmethod
    def _collect_extra_fields(cls, data: object) -> object:
        """
        Collect the members of an input document that this class does not
        declare into `extra_fields`.

        They are merged with any entries passed in `extra_fields` directly. A
        key passed both ways raises `ValueError`.
        """
        if not isinstance(data, Mapping):
            return data
        declared = cls.model_fields
        extra = {key: value for key, value in data.items() if key not in declared}
        if not extra:
            return data
        given = data.get(EXTRA_FIELDS, {})
        if not isinstance(given, Mapping):
            # field validation reports the malformed mapping
            return data
        twice = sorted(set(extra) & set(given))
        if twice:
            msg = f"Extra fields given both as members and in `{EXTRA_FIELDS}`: {twice}"
            raise ValueError(msg)
        return {
            **{key: value for key, value in data.items() if key in declared},
            EXTRA_FIELDS: {**given, **extra},
        }

    @field_validator(EXTRA_FIELDS, mode="after")
    @classmethod
    def _refuse_declared_keys(cls, value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
        """Reject an extra field whose name is a field that this class declares."""
        declared = sorted(set(value) & set(cls.model_fields))
        if declared:
            msg = f"{declared} name fields of this spec, so they are not extra fields"
            raise ValueError(msg)
        return value

    # No return annotation: pydantic uses a wrap serializer's return annotation
    # as the serialization schema. That would replace the fields' schema in
    # `model_json_schema(mode="serialization")` with an open object.
    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
        """Write the extra fields as top-level members of the dumped document."""
        return flatten_extra_fields(handler(self))

    @override
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler, /
    ) -> JsonSchemaValue:
        """
        Describe the document, not the model.

        A document stores its extra fields as top-level members and has no
        `extra_fields` member. So the schema replaces the `extra_fields`
        property with `additionalProperties`, which gives the type of an extra
        field's value. It also rejects any member named `extra_fields`.
        Declared fields are properties, so `additionalProperties` does not
        apply to them. This is the same in both modes, matching what
        validation accepts and what `model_dump` writes.
        """
        json_schema = handler(core_schema)
        definition = handler.resolve_ref_schema(json_schema)
        container = definition.get("properties", {}).pop(EXTRA_FIELDS, None)
        if container is not None:
            definition["additionalProperties"] = container["additionalProperties"]
            definition["propertyNames"] = {"not": {"const": EXTRA_FIELDS}}
        return json_schema

    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        """Show `extra_fields` last, and only if the document has any."""
        extra: object = None
        for name, value in super().__repr_args__():
            if name == EXTRA_FIELDS:
                extra = value
                continue
            yield name, value
        if extra:
            yield EXTRA_FIELDS, extra

    def _replaced(self, changes: Mapping[str, object]) -> Self:
        """
        Return a revalidated copy of this spec with `changes` applied.

        This is the shared implementation of each subclass's `replace`. The
        extra fields are passed as the `extra_fields` mapping, so a change to
        `extra_fields` replaces all of them.
        """
        document = self.model_dump(exclude={EXTRA_FIELDS})
        return type(self).model_validate({**document, EXTRA_FIELDS: self.extra_fields, **changes})


@cache
def _declared_keys(model: type[BaseModel]) -> frozenset[str]:
    """
    Return the input keys that pydantic validates into the fields of `model`.

    These are each field's alias or validation alias, and the field's name if
    pydantic accepts names. Cached per class, since every validation calls
    this.
    """
    config = model.model_config
    by_name = bool(config.get("validate_by_name") or config.get("populate_by_name"))
    by_alias = config.get("validate_by_alias", True)
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        alias = field.validation_alias if field.validation_alias is not None else field.alias
        if alias is None or by_name:
            keys.add(name)
        if alias is None or not by_alias:
            continue
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        for choice in choices:
            first = choice if isinstance(choice, str) else choice.path[0]
            if isinstance(first, str):
                keys.add(first)
    return frozenset(keys)


_EXTRA_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _dumps_extra_field(key: str, value: object, info: SerializationInfo) -> bool:
    """
    Return whether the dump described by `info` writes the extra field `key`.

    The rules are the same as for a declared field. If `include` is passed,
    it must name the key. Neither `exclude` nor `exclude_none` may drop the
    whole key.
    """
    include, exclude = info.include, info.exclude
    if include is not None and key not in include:
        return False
    # An `exclude` set drops the key. An `exclude` mapping drops the whole key
    # only if its value is `True` (or `...`).
    excluded = (
        exclude is not None
        and key in exclude
        and (isinstance(exclude, set) or exclude[key] in (True, ...))
    )
    return not excluded and not (info.exclude_none and value is None)


class _OpenModel(BaseModel):
    """
    A model that keeps the input keys it does not declare ("extra fields").

    It is for documents whose keys are an open namespace: attributes, and the
    experimental v3 specs.

    The extra fields are available through `extra_fields` and dumped back as
    top-level keys. A dump's `include`, `exclude` and `exclude_none` select
    them in the same way as declared fields.

    Extra fields are stored apart from the fields, rather than as pydantic
    extras (`extra="allow"`). This is because pydantic applies a model's
    `extra` setting to every TypedDict in the model that has no configuration
    of its own. `allow` would open plain TypedDicts, and warn for closed ones.
    Extra fields are not stored in a field either. So no input key names
    their container, and a frozen model hashes by its declared fields alone.

    A subclass that sets `extra` to `"forbid"` or `"ignore"` keeps no extra
    fields. Setting `"allow"` raises `TypeError` when the subclass is
    defined, since it would affect the TypedDicts.
    """

    _extra_fields: dict[str, object] = PrivateAttr(default_factory=dict)

    @override
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("extra") == "allow":
            msg = (
                f"{cls.__name__} keeps the keys it does not declare without "
                "`extra='allow'`, which would also open the TypedDicts it holds: "
                "leave `extra` unset"
            )
            raise TypeError(msg)

    @property
    def extra_fields(self) -> Mapping[str, object]:
        """The input keys that this model does not declare, with their values unchanged."""
        return MappingProxyType(self._extra_fields)

    @model_validator(mode="wrap")
    @classmethod
    def _collect_extra_fields(cls, data: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Validate the declared keys of an input document, and keep the others."""
        if not isinstance(data, Mapping) or "extra" in cls.model_config:
            return handler(data)
        declared = _declared_keys(cls)
        extra = {key: value for key, value in data.items() if key not in declared}
        named = sorted(set(extra) & set(cls.model_fields))
        if named:
            msg = f"{named} name fields of this model, so they are not extra fields"
            raise ValueError(msg)
        model = handler({key: value for key, value in data.items() if key in declared})
        model._extra_fields = extra
        return model

    # No return annotation, as on `ExtraFieldsBase`: pydantic uses a wrap
    # serializer's return annotation as the serialization schema. That would
    # replace the fields' schema in `model_json_schema(mode="serialization")`
    # with an open object.
    @model_serializer(mode="wrap")
    def _serialize_document(  # noqa: ANN202
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Write the extra fields that the dump selects as top-level keys of the document."""
        extra = {
            key: value
            for key, value in self._extra_fields.items()
            if _dumps_extra_field(key, value, info)
        }
        return {
            **handler(self),
            **_EXTRA_FIELDS_ADAPTER.dump_python(
                extra,
                mode="json" if info.mode_is_json() else "python",
                by_alias=info.by_alias,
                round_trip=info.round_trip,
            ),
        }

    @override
    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        """
        Copy this model as pydantic does.

        A key of `update` that names no field is set as an extra field.
        """
        changes = {} if update is None else update
        fields = type(self).model_fields
        copied = super().model_copy(
            update={key: value for key, value in changes.items() if key in fields}, deep=deep
        )
        copied._extra_fields = {
            **copied._extra_fields,
            **{key: value for key, value in changes.items() if key not in fields},
        }
        return copied

    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        """Show `extra_fields` last, and only if the model has any."""
        yield from super().__repr_args__()
        if self._extra_fields:
            yield EXTRA_FIELDS, self._extra_fields


class AttributesBase(_OpenModel):
    """
    A base class for models of stored attributes ("spec-style" attributes).

    Models of the attribute conventions found in stored metadata (CF, OME,
    STAC, ...) need two behaviors that plain pydantic models lack by default.
    This class provides both:

    - Keys that are not Python identifiers (`_FillValue`, `image-label`,
      `proj:code`) are declared as field aliases. Serialization writes the
      alias, and validation accepts either the alias or the field name.
    - Optional keys are declared with the [`MISSING`
      sentinel](https://docs.pydantic.dev/latest/concepts/experimental/#missing-sentinel),
      for example `units: str | MISSING = MISSING`. The sentinel is
      pydantic's, re-exported as `pydantic_zarr.core.MISSING` so that one
      import works across the supported pydantic range. A field left at
      `MISSING` is excluded from serialization, so "absent key" is a value of
      its own rather than a serializer policy. The annotation can then match
      the spec exactly. Where `null` is meaningful, include `None`
      (`str | None | MISSING`); an assigned `None` keeps its key, written as
      `null`. Where a key is present or absent but never null (most specs
      where "optional keys are omitted"), leave `None` out; an explicit
      `None` is then a validation error.

    Unknown keys are preserved and round-trip, so the model is open by
    default. Attributes read from a store keep the keys this model does not
    declare. They are available through the read-only `extra_fields` mapping
    and dumped back as top-level keys. A dump's `include` and `exclude` name
    them by key.

    For a closed model, set `extra="forbid"` in a subclass's `model_config`.
    `extra="ignore"` drops unknown keys instead. `extra="allow"` raises
    `TypeError`.

    Unknown keys are kept apart from pydantic's extras, so that this model's
    openness does not affect the TypedDicts it contains. Each nested
    TypedDict follows its own configuration, as it would anywhere else: a
    plain one drops unknown keys, a closed one rejects them, and one with
    `extra_items` keeps them.

    `model_copy(update=...)` sets a key that names no field as an extra
    field. Assigning an extra field as an attribute is an error.

    Subclasses can override these behaviors through `model_config`, except
    for alias serialization. A spec dumps its whole tree with
    `by_alias=True`, so attributes dumped through a spec use aliases
    regardless of this model's own config. To turn aliases off, pass
    `by_alias=False` to the spec's dump.

    Attributes
    ----------
    extra_fields: Mapping[str, object]
        The keys this model does not declare, with their values unchanged.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


def get_attribute(attributes: object, key: str, default: object = None) -> object:
    """
    Read one key from a node's attributes, in either form.

    A `Mapping` is read by key. A pydantic model is read through the field
    whose name or alias matches the key, or else from its extra (undeclared)
    fields. Returns `default` if the key is absent. For a model, the key is
    absent if it would not appear in the model's serialization. So a field
    left at the `MISSING` sentinel returns the default, just as the mapping
    form of the same document would. A cross-member validator on a group
    whose attributes type allows both forms needs this behavior.
    """
    if isinstance(attributes, Mapping):
        return attributes.get(key, default)
    if isinstance(attributes, BaseModel):
        for name, field in type(attributes).model_fields.items():
            aliases: list[str] = [name]
            for alias in (field.alias, field.serialization_alias, field.validation_alias):
                if isinstance(alias, str):
                    aliases.append(alias)
                elif isinstance(alias, AliasChoices):
                    aliases.extend(choice for choice in alias.choices if isinstance(choice, str))
            if key not in aliases:
                continue
            value = getattr(attributes, name)
            if value is MISSING:
                return default
            return value
        extra = (
            attributes.extra_fields
            if isinstance(attributes, _OpenModel)
            else attributes.__pydantic_extra__
        )
        if extra is not None and key in extra:
            return extra[key]
    return default


@runtime_checkable
class _HasMembers(Protocol):
    """
    The structural type that path traversal requires of a group: a `members` mapping.

    The v2 and v3 `GroupSpec` both have it. `ArraySpec` and `BaseGroupSpec`
    do not, which is how traversal tells a group with members apart from
    every other node.
    """

    @property
    def members(self) -> Mapping[str, object]: ...


def resolve_member_path(node: object, path: str) -> object:
    """
    Return the node at a `/`-separated path of member names relative to `node`.

    Examples of paths are `"a"` and `"a/b/c"`. Leading and trailing `/` are
    ignored. This is the shared implementation of the v2 and v3
    `GroupSpec.resolve_path`. Their docstrings document the public behavior.

    Raises
    ------
    ValueError
        If the path contains no member names, or an empty one (`"a//b"`).
    KeyError
        If a segment names a member that does not exist, or descends into a
        node that is not a group. The message names the segment that failed.
    """
    stripped = path.strip("/")
    if stripped == "":
        msg = f"Invalid path {path!r}: it holds no member names."
        raise ValueError(msg)
    resolved: list[str] = []
    for segment in stripped.split("/"):
        where = "/".join(resolved) if resolved else "this group"
        if segment == "":
            msg = f"Invalid path {path!r}: empty member name."
            raise ValueError(msg)
        if not isinstance(node, _HasMembers):
            msg = f"Cannot resolve {segment!r}: {where} is not a group with members."
            raise KeyError(msg)
        if segment not in node.members:
            available = ", ".join(repr(name) for name in sorted(node.members)) or "(none)"
            msg = f"No member named {segment!r} in {where}. Members: {available}."
            raise KeyError(msg)
        node = node.members[segment]
        resolved.append(segment)
    return node


def read_attribute(data: object, name: str) -> object:
    """
    Return `data.<name>`, typed as `object`.

    The `auto_*` readers of the v2 and v3 `from_array` probe a duck-typed
    array (numpy, dask, zarr, a spec) with `hasattr`. They read what they find
    through this function. They then check the value before returning it as
    the type they promise.
    """
    return getattr(data, name)


def as_shape(value: object) -> tuple[int, ...]:
    """
    Convert `value` to a tuple of ints.

    `value` is an array's shape or chunk shape, read from a duck-typed array.
    Each element is converted with `operator.index`, which accepts both
    Python and numpy integers.

    Raises
    ------
    TypeError
        If `value` is not an iterable of integers.
    """
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        msg = f"Expected an iterable of integers, got {value!r}."
        raise TypeError(msg)
    return tuple(operator.index(n) for n in value)


def as_dtype_like(value: object) -> npt.DTypeLike:
    """
    Return `value` unchanged, typed as numpy's `DTypeLike`.

    `value` is the `dtype` read from a duck-typed array. `DTypeLike` cannot be
    checked with `isinstance`. Every consumer of the result (`np.dtype` and
    the `parse_dtype_*` functions built on it) validates the value. They
    raise `TypeError` if numpy cannot interpret it as a data type, so no
    check is needed here.
    """
    return cast("npt.DTypeLike", value)


def guess_chunks(shape: tuple[int, ...], typesize: int) -> tuple[int, ...]:
    """
    Vendored from zarr-python v2.

    Guess an appropriate chunk layout for an array, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """

    CHUNK_BASE = 256 * 1024  # Multiplier by which chunks are adjusted
    CHUNK_MIN = 128 * 1024  # Soft lower limit (128k)
    CHUNK_MAX = 64 * 1024 * 1024  # Hard upper limit

    ndims = len(shape)
    # require chunks to have non-zero length for all dimensions
    chunks = np.maximum(np.array(shape, dtype="=f8"), 1)

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.prod(chunks) * typesize
    target_size = CHUNK_BASE * (2 ** np.log10(dset_size / (1024.0 * 1024)))

    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        # 2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.prod(chunks) * typesize

        if (
            chunk_bytes < target_size or abs(chunk_bytes - target_size) / target_size < 0.5
        ) and chunk_bytes < CHUNK_MAX:
            break

        if np.prod(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        chunks[idx % ndims] = math.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)


def parse_dtype_v2(value: npt.DTypeLike) -> str | list[tuple[object, ...]]:
    """
    Convert the input to a NumPy dtype and either return the ``str`` attribute of that
    object or, if the dtype is a structured dtype, return the fields of that dtype as a list
    of tuples.

    Parameters
    ----------
    value : npt.DTypeLike
        A value that can be converted to a NumPy dtype.

    Returns
    -------

    A Zarr V2-compatible encoding of the dtype.

    References
    ----------
    See the [Zarr V2 specification](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding)
    for more details on this encoding of data types.
    """
    # Assume that a non-string sequence is the Zarr V2 JSON form of a structured dtype.
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [tuple(v) for v in value]
    else:
        np_dtype = np.dtype(value)
        if np_dtype.fields is not None:
            # This is a structured dtype, which must be converted to a list of tuples. Note that
            # this function recurses, because a structured dtype is parametrized by other dtypes.
            return [(k, parse_dtype_v2(v[0])) for k, v in np_dtype.fields.items()]
        else:
            return np_dtype.str


def canonicalize_fill_value(fill_value: object) -> object:
    """
    Normalize a fill value to the form the Zarr specs use in JSON.

    - Numpy scalars become the equivalent Python scalars.
    - Complex numbers become ``(real, imaginary)`` pairs.
    - Non-finite floats become the strings that both the Zarr V2 and V3 specs
      require (``"NaN"``, ``"Infinity"``, ``"-Infinity"``), because a
      metadata document can only contain finite JSON numbers.
    - Sequences (complex pairs, raw byte values) are normalized element-wise.
    - Everything else passes through unchanged.
    """
    if isinstance(fill_value, np.generic):
        fill_value = fill_value.item()
    if isinstance(fill_value, complex):
        return (
            canonicalize_fill_value(fill_value.real),
            canonicalize_fill_value(fill_value.imag),
        )
    if isinstance(fill_value, float) and not math.isfinite(fill_value):
        if math.isnan(fill_value):
            return "NaN"
        return "Infinity" if fill_value > 0 else "-Infinity"
    if isinstance(fill_value, (tuple, list)):
        return tuple(canonicalize_fill_value(item) for item in fill_value)
    return fill_value


def ensure_member_name(data: object) -> str:
    """
    Return the input if it is a valid name for a member of a Zarr group.

    Raises `ValueError` if the name contains `/`, or is `""`, `"."` or
    `".."`. Raises `TypeError` if the input is not a string.
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


def ensure_key_no_path(data: T) -> T:
    if isinstance(data, Mapping):
        for key in data:
            ensure_member_name(key)
    return data


def copy_members(members: object) -> object:
    """
    Return a `dict` copy of `members` if it is a mapping, or `members` unchanged.

    `GroupSpec` runs this before it validates `members`. Under
    `GroupSpec[Any, Any]`, pydantic keeps the object it receives. Without the
    copy, a frozen spec would share the caller's mapping, and later changes to
    that mapping would skip validation. A mapping type that pydantic cannot
    serialize, such as `MappingProxyType`, would also make `model_dump_json`
    fail. A value that is not a mapping is left for validation to reject.
    """
    return dict(members) if isinstance(members, Mapping) else members


def restore_member_order(members_field: FieldInfo, given: object, validated: object) -> object:
    """
    Return a group's validated members with their keys in the order of `given`,
    the members the field received.

    pydantic builds a `TypedDict` with its keys in the order the class
    declares them, followed by any extra items. This puts the members back in
    the order they were given. It does so only when every part of the members
    type is one whose order pydantic changes by declaration alone: a
    `TypedDict`, generic or not, a `dict` or `Mapping`, or a union, type alias,
    type variable or `Annotated` form built from these. Any other members
    type, such as one with validators of its own, may set an order on purpose,
    so its value is returned as it is. So is a value that is not a plain
    `dict`.
    """
    if type(validated) is not dict or not isinstance(given, Mapping):
        return validated
    order = [key for key in given if key in validated]
    if list(validated)[: len(order)] == order:
        return validated
    if not all(map(_keeps_order, members_field.metadata)) or not _reorders_by_declaration_only(
        members_field.annotation
    ):
        return validated
    # the given keys first, then any others in their validated order
    return {**{key: validated[key] for key in order}, **validated}


# `Annotated` metadata that changes how a value is serialized, or which arm of
# a union validates it, but never the order of the validated mapping's keys
_ORDER_KEEPING_METADATA = (PlainSerializer, WrapSerializer, Discriminator, Tag)

_TYPE_ALIAS_TYPES = (TypeAliasType, typing_extensions.TypeAliasType)


def _keeps_order(metadata: object) -> bool:
    # pydantic metadata that changes validation (`AfterValidator` and the
    # others) does so through this hook; plain metadata such as a docstring,
    # a `Field(...)` or an `annotated_types` constraint has no such hook
    return not hasattr(metadata, "__get_pydantic_core_schema__") or isinstance(
        metadata, _ORDER_KEEPING_METADATA
    )


def _reorders_by_declaration_only(annotation: object) -> bool:
    """
    Return whether pydantic can change the key order of a value of
    `annotation` only by building a `TypedDict` in declaration order.

    This looks through type aliases, `Annotated`, unions and type variables.
    It does not look into type arguments such as the value type of
    `dict[str, X]`, because validators there act on single members.
    """
    if isinstance(annotation, _TYPE_ALIAS_TYPES):
        return _reorders_by_declaration_only(annotation.__value__)
    if isinstance(annotation, TypeVar):
        default = getattr(annotation, "__default__", NoDefault)
        candidates = [annotation.__bound__, *annotation.__constraints__]
        if default is not NoDefault:
            candidates.append(default)
        return all(
            _reorders_by_declaration_only(candidate)
            for candidate in candidates
            if candidate is not None
        )
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        return all(map(_keeps_order, metadata)) and _reorders_by_declaration_only(inner)
    if origin is Union or origin is UnionType:
        return all(map(_reorders_by_declaration_only, get_args(annotation)))
    plain = origin or annotation
    return is_typeddict(plain) or plain in (dict, Mapping, MutableMapping)


def ensure_members_mapping(members: object) -> None:
    """
    Raise `ValueError` unless `members` is a mapping with string keys.

    `GroupSpec` checks this itself because its members type is a type
    parameter. pydantic does not enforce a type parameter's bound, so under
    `GroupSpec[Any, Any]` the annotation alone would let any value through.
    """
    if not isinstance(members, Mapping):
        msg = f"members must be a mapping of member names to specs, got {type(members).__name__}"
        raise ValueError(msg)  # noqa: TRY004 (pydantic reports ValueError as a validation error)
    not_names = [key for key in members if not isinstance(key, str)]
    if not_names:
        msg = f"member names must be strings, got {not_names!r}"
        raise ValueError(msg)


def model_like(a: BaseModel, b: BaseModel, exclude: IncEx = None, include: IncEx = None) -> bool:
    """
    Return whether two pydantic models dump to equal JSON documents.

    Only the fields selected by `include` and `exclude` are compared. Lists
    and tuples compare as equal (see `json_eq`).
    """

    a_dict = a.model_dump(exclude=exclude, include=include)
    b_dict = b.model_dump(exclude=exclude, include=include)
    return json_eq(a_dict, b_dict)


def ensure_multiple(data: Sequence[T]) -> Sequence[T]:
    """
    Ensure that there is at least one element in the sequence
    """
    if len(data) < 1:
        raise ValueError("Invalid length. Expected 1 or more, got 0.")
    return data


def json_eq(a: object, b: object) -> bool:
    """
    An equality check between python objects that recurses into dicts and sequences and ignores
    the difference between tuples and lists. Otherwise, it's just regular equality. Useful
    for comparing dicts that would become identical JSON, but where one has lists and the other
    has tuples.
    """
    # treat lists & tuples as the same "sequence" type
    seq_types = (list, tuple)

    # both are sequences → compare element-wise
    if isinstance(a, seq_types) and isinstance(b, seq_types):
        return len(a) == len(b) and all(json_eq(x, y) for x, y in zip(a, b, strict=False))

    # recurse into mappings
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(json_eq(a[k], b[k]) for k in a)

    # otherwise → regular equality
    return a == b


type StoredNodeType = Literal["array", "group"]
"""The type of a node in a stored layout."""

type StoredNodes = Mapping[str, tuple[StoredNodeType | None, object]]
"""
A stored layout, as each node's type and metadata document.

Each node is keyed by its key prefix (see `split_store_key`): `""` for the
root, `"a/b"` for the node at hierarchy path `/a/b`. The type is `None` if the
document does not state it. In that case, validation checks the document.
"""

_A_NODE: dict[StoredNodeType, str] = {"array": "an array", "group": "a group"}


def split_store_key(key: str) -> tuple[str, str]:
    """
    Split a store key into the node's key prefix and the document's name.

    For example, `"a/b/zarr.json"` becomes `("a/b", "zarr.json")`, and
    `"zarr.json"` becomes `("", "zarr.json")`.

    A node's path and its store keys have different forms. In Zarr V3, a
    node's hierarchy path starts with `/`: the root's path is `/`, and a
    member's path looks like `/a/b`. Its metadata key is that path with the
    leading `/` removed and `/zarr.json` appended (`a/b/zarr.json`). The
    root's metadata key is `zarr.json`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L382-L388).
    In Zarr V2, a node's logical path is normalized before it is used as the
    prefix of the node's keys (`a/b/.zarray`). Normalization strips any
    leading or trailing `/` and collapses repeated `/`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L250-L276).

    In both versions, a store key has no leading `/` and no empty segment.
    This function raises `ValueError` for a key that does. Such a key would
    name the same node as another key (`"/zarr.json"` as `"zarr.json"`) and
    replace that key's document.
    """
    *segments, name = key.split("/")
    if "" in segments:
        msg = (
            f"Invalid store key {key!r}: a store key is the node's path with no leading "
            "'/' and no empty segment ('zarr.json' for the root, 'a/b/zarr.json' for the "
            "node at '/a/b')."
        )
        raise ValueError(msg)
    return "/".join(segments), name


def stored_root(nodes: StoredNodes, node_type: StoredNodeType) -> object:
    """
    Return the root's metadata document from `nodes`.

    Raises `ValueError` if there is no root document, or if the root's
    document states that it is not a `node_type`.
    """
    if "" not in nodes:
        msg = f"The layout holds no {node_type}'s metadata at the root."
        raise ValueError(msg)
    root_type, document = nodes[""]
    if root_type is not None and root_type != node_type:
        msg = f"The layout's root is {_A_NODE[root_type]}, not {_A_NODE[node_type]}."
        raise ValueError(msg)
    return document


def stored_array(nodes: StoredNodes) -> object:
    """
    Return the metadata document of the array that `nodes` describe.

    Raises `ValueError` if `nodes` contain no array at the root, or contain
    any other node.
    """
    document = stored_root(nodes, "array")
    if len(nodes) > 1:
        others = ", ".join(repr(path) for path in nodes if path)
        msg = f"An array has no members, but the layout holds documents at {others}."
        raise ValueError(msg)
    return document


def stored_flat[A: BaseModel, G: BaseModel](
    nodes: StoredNodes, array_class: type[A], group_class: type[G]
) -> dict[str, A | G]:
    """
    Return the hierarchy that `nodes` describe, in the form `from_flat` accepts.

    A node whose type is `"array"` is validated as an `array_class`. Any other
    node, including one whose type is not stated, is validated as a
    `group_class`. Each node is keyed by its path in the form `to_flat` uses.
    That is the key prefix with a leading `/` (`"/a/b"`, the V3 hierarchy
    path), except that the root's path is `""`, not `/`.

    Raises `ValueError` if `nodes` contain no group at the root, or contain a
    node whose parent is not a group in `nodes` (a store has no implicit
    groups). Raises `ValidationError` if a node's document is not valid.
    """
    stored_root(nodes, "group")
    flat: dict[str, A | G] = {
        f"/{path}" if path else "": (
            array_class if node_type == "array" else group_class
        ).model_validate(document)
        for path, (node_type, document) in nodes.items()
    }
    for path in nodes:
        if path:
            parent = path.rpartition("/")[0]
            if not isinstance(flat.get(f"/{parent}" if parent else ""), group_class):
                where = repr(parent) if parent else "the root"
                msg = (
                    f"The layout holds documents at {path!r}, but no group's at {where}, "
                    "its parent."
                )
                raise ValueError(msg)
    return flat
