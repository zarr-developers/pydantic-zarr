from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    Self,
    TypeVar,
    overload,
    override,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

# pydantic resolves the `extra_fields` annotation at runtime
from zarr_metadata import JSONValue  # noqa: TC002

if TYPE_CHECKING:
    # both pydantic spellings below re-export pydantic_core's sentinel, and
    # unlike them this import resolves statically on every supported pydantic;
    # `as MISSING` marks the re-export explicit for strict checkers
    from pydantic_core import MISSING as MISSING
else:
    try:
        # the MISSING sentinel is stable from pydantic 2.14
        from pydantic import MISSING
    except ImportError:
        # on pydantic 2.12 / 2.13 it lives in the experimental module, with
        # the same object and behavior; re-exported here so users of
        # AttributesBase have one import spanning the supported pydantic range
        from pydantic.experimental.missing_sentinel import MISSING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import zarr
    from zarr.storage._common import StoreLike

type IncEx = set[int] | set[str] | dict[int, Any] | dict[str, Any] | None

type AccessMode = Literal["w", "w+", "r", "a"]

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


class StrictBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Serialize with `by_alias=True` unless told otherwise: a spec's own
        fields declare no aliases, so this only affects attribute models,
        whose aliases spell stored metadata keys that are not Python
        identifiers (`_FillValue`, `image-label`, `proj:code`). Without this
        default, a hierarchy would be silently written to storage under the
        Python field names.
        """
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    @override
    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize to JSON with `by_alias=True` unless told otherwise; see `model_dump`."""
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)


EXTRA_FIELDS = "extra_fields"
"""The name of the field in which `ExtraFieldsBase` keeps a document's extra fields."""


def flatten_extra_fields(document: dict[str, Any]) -> dict[str, Any]:
    """
    Turn a dumped spec's `extra_fields` mapping back into top-level members
    of the document, where the metadata document holds them.
    """
    extra = document.pop(EXTRA_FIELDS, None)
    if extra:
        document.update(extra)
    return document


class ExtraFieldsBase(StrictBase):
    """
    A base class for specs of metadata documents that may hold members the
    spec class does not declare ("extra fields").

    The members a document holds beyond the declared fields are gathered into
    the `extra_fields` mapping, by key, and dumped back as top-level members:
    `model_dump` and `model_dump_json` write the document as it was read.
    They are held in a field of their own rather than as pydantic extras
    (`extra="allow"`), because pydantic applies a model's `extra` setting to
    every TypedDict the model holds, which would open the closed TypedDicts
    of field and attribute types. Which extra fields a document may hold is
    up to each subclass; a key nobody validates is still kept out of
    `extra_fields` when it names a declared field.

    Attributes
    ----------
    extra_fields: Mapping[str, JSONValue], default = {}
        The document's members that this class does not declare, by key,
        with arrays held as tuples.
    """

    extra_fields: Mapping[str, Annotated[JSONValue, BeforeValidator(tuplify_json)]] = Field(
        default={}
    )

    @model_validator(mode="before")
    @classmethod
    def _collect_extra_fields(cls, data: object) -> object:
        """
        Gather the members of an input document that this class does not
        declare into `extra_fields`, alongside any given there directly.
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
        """Refuse an extra field named after a field this class declares."""
        declared = sorted(set(value) & set(cls.model_fields))
        if declared:
            msg = f"{declared} name fields of this spec, so they are not extra fields"
            raise ValueError(msg)
        return value

    @model_serializer(mode="wrap")
    def _serialize_document(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write the extra fields as top-level members of the dumped document."""
        return flatten_extra_fields(handler(self))

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """Show `extra_fields` last, and only when the document holds some."""
        extra: object = None
        for name, value in super().__repr_args__():
            if name == EXTRA_FIELDS:
                extra = value
                continue
            yield name, value
        if extra:
            yield EXTRA_FIELDS, extra

    def _updated(self, changes: Mapping[str, object]) -> Self:
        """
        A revalidated copy of this spec with `changes` applied: the engine of
        each subclass's `update`. The extra fields are passed as the
        `extra_fields` mapping, so a change to `extra_fields` replaces them.
        """
        document = self.model_dump(exclude={EXTRA_FIELDS})
        return type(self).model_validate({**document, EXTRA_FIELDS: self.extra_fields, **changes})


@cache
def _declared_keys(model: type[BaseModel]) -> frozenset[str]:
    """
    The keys of an input document that pydantic reads into a field of
    `model`: a field's alias or validation alias, and its name where pydantic
    accepts names. Cached per class, since every validation asks.
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


_EXTRA_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, Any])


def _dumps_extra_field(key: str, value: object, info: SerializationInfo) -> bool:
    """
    Whether the dump `info` describes writes the extra field `key`: as for a
    declared field, `include` must name it (when given), and neither
    `exclude` nor `exclude_none` may drop it whole.
    """
    include, exclude = info.include, info.exclude
    if include is not None and key not in include:
        return False
    # a set drops the key; a mapping drops it whole with `True` (or `...`)
    excluded = (
        exclude is not None
        and key in exclude
        and (isinstance(exclude, set) or exclude[key] in (True, ...))
    )
    return not excluded and not (info.exclude_none and value is None)


class _OpenModel(BaseModel):
    """
    A model that keeps the keys of its input it does not declare ("extra
    fields"), for documents whose keys are an open namespace: attributes,
    and the experimental v3 specs.

    The extra fields are read through `extra_fields` and dumped back as
    top-level keys, selected by a dump's `include`, `exclude` and
    `exclude_none` as declared fields are. They are kept apart from the
    fields rather than as pydantic extras (`extra="allow"`), because pydantic
    applies a model's `extra` setting to every TypedDict the model holds that
    has no configuration of its own: `allow` would open plain TypedDicts, and
    warns for closed ones. They are not a field either, so no input key names
    their container, and a frozen model hashes by its declared fields alone.

    A subclass that sets `extra` to `"forbid"` or `"ignore"` keeps no extra
    fields; setting `"allow"` is refused, since it would reach the TypedDicts.
    """

    _extra_fields: dict[str, Any] = PrivateAttr(default_factory=dict)

    @override
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("extra") == "allow":
            msg = (
                f"{cls.__name__} keeps the keys it does not declare without "
                "`extra='allow'`, which would also open the TypedDicts it holds: "
                "leave `extra` unset"
            )
            raise TypeError(msg)

    @property
    def extra_fields(self) -> Mapping[str, Any]:
        """The keys of the input this model does not declare, with their values as given."""
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

    @model_serializer(mode="wrap")
    def _serialize_document(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """Write the extra fields the dump selects as top-level keys of the dumped document."""
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
    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """
        Copy this model as pydantic does; a key of `update` that names no
        field is set as an extra field.
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

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """Show `extra_fields` last, and only when the model holds some."""
        yield from super().__repr_args__()
        if self._extra_fields:
            yield EXTRA_FIELDS, self._extra_fields


class AttributesBase(_OpenModel):
    """
    A base class for models of stored attributes ("spec-style" attributes).

    Models of the attribute conventions found in stored metadata (CF, OME,
    STAC, ...) need two behaviors plain pydantic models lack by default,
    which this class provides:

    - Keys that are not Python identifiers (`_FillValue`, `image-label`,
      `proj:code`) are declared as field aliases; serialization writes the
      alias, and validation accepts both spellings.
    - Optional keys are declared with the [`MISSING`
      sentinel](https://docs.pydantic.dev/latest/concepts/experimental/#missing-sentinel)
      (pydantic's, re-exported as `pydantic_zarr.core.MISSING` so one import
      spans the supported pydantic range) — `units: str | MISSING = MISSING`
      — and a field left at `MISSING` is
      excluded from serialization, so "absent key" is a value of its own
      rather than a serializer policy. The annotation then says exactly what
      the spec says: where `null` is meaningful it includes `None`
      (`str | None | MISSING`) and an assigned `None` keeps its key as
      `null`; where a key is present or absent but never null — most
      "optional keys are omitted" specs — it leaves `None` out, and an
      explicit `None` is a validation error.

    Unknown keys are preserved and round-trip: attributes read from a store
    keep the keys this model does not declare, read through the read-only
    `extra_fields` mapping, and dump them back as top-level keys (a dump's
    `include` and `exclude` name them by key), so the model is OPEN by
    default — declare `model_config` with `extra="forbid"` in a subclass for
    a closed one (`extra="ignore"` drops unknown keys instead; `extra="allow"`
    is refused). They are kept apart from pydantic's extras so that this
    model's openness does not reach the TypedDicts it holds: each nested
    TypedDict says its own, as it would anywhere else (a plain one drops
    unknown keys, a closed one rejects them, one with `extra_items` keeps
    them). `model_copy(update=...)` sets a key that names no field as an
    extra field; assigning one as an attribute is an error. Subclasses can
    override this via `model_config`, except alias serialization: a spec
    dumps its whole tree with `by_alias=True`, so attributes dumped through a
    spec use aliases whatever this model's own config says, unless the
    caller passes `by_alias=False` to the spec's dump.

    Attributes
    ----------
    extra_fields: Mapping[str, Any]
        The keys this model does not declare, with their values as given.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


def get_attribute(attributes: object, key: str, default: object = None) -> object:
    """
    Read one key from a node's attributes, whichever form they take: a
    `Mapping` is read by key, and a pydantic model by the field whose name or
    alias matches the key, extra (undeclared) fields included. Returns
    `default` when the key is absent — for a model, absent means the key would
    not appear in its serialization, so a field left at the `MISSING`
    sentinel reads as the default, exactly as the mapping form of the same
    document would. A cross-member validator on a group whose attributes
    type admits both forms needs exactly this.
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
    The structural shape path traversal needs from a group: a `members`
    mapping. The v2 and v3 `GroupSpec`
    both have it; `ArraySpec` and `BaseGroupSpec` do not, which is how
    traversal tells a group with members from every other node.
    """

    @property
    def members(self) -> Mapping[str, object]: ...


def resolve_member_path(node: object, path: str) -> object:
    """
    Return the node at a `/`-separated path of member names relative to
    `node` (`"a"`, `"a/b/c"`; leading and trailing `/` are ignored). The
    shared engine of the v2 and v3 `GroupSpec.resolve_path`, whose docstrings
    carry the user-facing contract.

    Raises
    ------
    ValueError
        If the path holds no member names, or an empty one (`"a//b"`).
    KeyError
        Naming the segment that failed: a member name that does not exist, or
        a segment that descends into a node that is not a group.
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


def parse_dtype_v2(value: npt.DTypeLike) -> str | list[tuple[Any, ...]]:
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
    # Assume that a non-string sequence represents a the Zarr V2 JSON form of a structured dtype.
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
    Normalize a fill value to the form the Zarr specs write in JSON.

    Numpy scalars become the equivalent Python scalars, complex numbers
    become ``(real, imaginary)`` pairs, and non-finite floats become the
    string forms both the Zarr V2 and V3 specs require (``"NaN"``,
    ``"Infinity"``, ``"-Infinity"``), since a metadata document holds only
    finite JSON numbers. Sequences (complex pairs, raw byte values) are
    normalized element-wise. Everything else passes through unchanged.
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
        for key in data:
            ensure_member_name(key)
    return data


def model_like(a: BaseModel, b: BaseModel, exclude: IncEx = None, include: IncEx = None) -> bool:
    """
    A similarity check for a pair pydantic.BaseModel, parametrized over included or excluded fields.


    """

    a_dict = a.model_dump(exclude=exclude, include=include)
    b_dict = b.model_dump(exclude=exclude, include=include)
    return json_eq(a_dict, b_dict)


# TODO: expose contains_array and contains_group as public functions in zarr-python
# and replace these custom implementations
def maybe_node(
    store: StoreLike, path: str, *, zarr_format: Literal[2, 3]
) -> zarr.Array | zarr.Group | None:
    """
    Return the array or group found at the store / path, if an array or group exists there.
    Otherwise return None.
    """
    from zarr.core.sync import sync
    from zarr.core.sync_group import get_node
    from zarr.storage._common import make_store_path

    # convert the storelike store argument to a Zarr store
    spath = sync(make_store_path(store, path=path))
    try:
        return get_node(spath.store, spath.path, zarr_format=zarr_format)
    except FileNotFoundError:
        return None


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
A stored layout read as each node's type and metadata document, keyed by the
node's key prefix (see `split_store_key`): `""` for the root, `"a/b"` for the
node at hierarchy path `/a/b`. The type is `None` when the document does not
say, and the document is then left for validation to judge.
"""

_A_NODE: dict[StoredNodeType, str] = {"array": "an array", "group": "a group"}


def split_store_key(key: str) -> tuple[str, str]:
    """
    Split a store key in a stored layout into the node's key prefix and the
    document's name: `"a/b/zarr.json"` into `("a/b", "zarr.json")`, and
    `"zarr.json"` into `("", "zarr.json")`.

    A node's path and its store keys are spelled differently. In Zarr V3 a
    node's hierarchy path starts with `/` (the root's is `/`, a member's
    `/a/b`), and its metadata key is that path with the leading `/` stripped,
    then `/zarr.json` appended (`a/b/zarr.json`); the root's is `zarr.json`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L382-L388).
    In Zarr V2 a node's logical path is normalized, stripping any leading or
    trailing `/` and collapsing repeated `/`, before it prefixes the node's
    keys (`a/b/.zarray`) (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L250-L276).
    Either way a store key has no leading `/` and no empty segment, so this
    raises `ValueError` for one that does: it would name the same node as
    another key (`"/zarr.json"` as `"zarr.json"`) and replace that key's
    document.
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
    The root's metadata document among `nodes`. Raises `ValueError` when there
    is none, or the root's document says it is not a `node_type`.
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
    The metadata document of the array `nodes` describe. Raises `ValueError`
    when `nodes` hold no array at the root, or hold any other node.
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
    The hierarchy `nodes` describe, as `from_flat` reads one: each node
    validated as an `array_class` or a `group_class` (a node whose type is not
    stated as the latter), keyed by its path in `to_flat`'s spelling. That is
    the key prefix with `/` put back in front (`"/a/b"`, the V3 hierarchy path),
    except that the root's is `""`, not `/`.

    Raises `ValueError` when `nodes` hold no group at the root, or a node whose
    parent is not a group among `nodes` (a store holds no implicit groups),
    and `ValidationError` when a node's document is not valid.
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


def walk_zarr_hierarchy(
    node: zarr.Array | zarr.Group, *, depth: int = -1
) -> Iterator[tuple[str, zarr.Array | zarr.Group]]:
    """
    Yield `(path, node)` for a zarr array or group and the members under it,
    `depth` levels deep, with paths in the flat-hierarchy form `to_flat` uses:
    `""` for `node` itself, then `"/a"`, `"/a/b"` and so on. A `depth` of -1
    walks the whole hierarchy and 0 yields `node` alone. The shared engine of
    the v2 and v3 `from_zarr_flat`.

    Raises
    ------
    ValueError
        If `depth` is lower than -1.
    """
    if depth < -1:
        msg = f"Invalid value for depth. Got {depth}, expected an integer greater than or equal to -1."
        raise ValueError(msg)
    yield "", node
    yield from _walk_members(node, "", depth)


def _walk_members(
    node: zarr.Array | zarr.Group, prefix: str, depth: int
) -> Iterator[tuple[str, zarr.Array | zarr.Group]]:
    import zarr

    if depth == 0 or not isinstance(node, zarr.Group):
        return
    for name, member in node.members():
        path = f"{prefix}/{name}"
        yield path, member
        yield from _walk_members(member, path, max(depth - 1, -1))
