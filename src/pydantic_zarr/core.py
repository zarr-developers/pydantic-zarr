from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    TypeVar,
    overload,
    override,
)

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

if TYPE_CHECKING:
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


class AttributesBase(BaseModel):
    """
    A base class for models of stored attributes ("spec-style" attributes).

    Models of the attribute conventions found in stored metadata (CF, OME,
    STAC, ...) need two behaviors plain pydantic models lack by default,
    which this class provides:

    - Keys that are not Python identifiers (`_FillValue`, `image-label`,
      `proj:code`) are declared as field aliases; serialization writes the
      alias, and validation accepts both spellings.
    - A field left unset whose value is `None` serializes as an absent key
      rather than `null`, matching specs where optional keys are omitted.
      A field explicitly set to `None` keeps its key, so a spec that
      distinguishes absent from null can express both.

    Unknown keys are preserved and round-trip (`extra="allow"`): attributes
    read from a store keep the keys this model does not declare, so the model
    is OPEN by default — declare `model_config` with `extra="forbid"` in a
    subclass for a closed one. Subclasses can override any of this via
    `model_config`.

    What an explicitly assigned `None` serializes as is the class-level
    `explicit_none` policy; the default keeps it as `null`.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="allow")

    explicit_none: ClassVar[Literal["keep", "omit", "refuse"]] = "keep"
    """
    What an explicitly assigned `None` means:
    `"keep"` (the default) serializes it as `null`, for specs where null is
    meaningful; `"omit"` drops the key, exactly like an unset field; and
    `"refuse"` rejects the assignment at validation, for specs where a key is
    absent or non-null, never null (most "optional keys are omitted" specs).
    """

    @model_validator(mode="after")
    def _refuse_explicit_none(self) -> Self:
        if type(self).explicit_none == "refuse":
            for name in self.model_fields_set:
                if name in type(self).model_fields and getattr(self, name) is None:
                    msg = (
                        f"Field {name!r} is None, and {type(self).__name__} refuses an "
                        'explicit None (explicit_none="refuse"): leave the field unset '
                        "for an absent key."
                    )
                    raise ValueError(msg)
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped = handler(self)
        omit_explicit = type(self).explicit_none == "omit"
        for name, field in type(self).model_fields.items():
            if name in self.model_fields_set and not omit_explicit:
                continue
            # the serialized key may be the field name or its alias,
            # depending on the serialization flags in effect
            for key in {name, field.serialization_alias or field.alias or name}:
                if key in dumped and dumped[key] is None:
                    del dumped[key]
        return dumped


def get_attribute(attributes: object, key: str, default: object = None) -> object:
    """
    Read one key from a node's attributes, whichever form they take: a
    `Mapping` is read by key, and a pydantic model by the field whose name or
    alias matches the key, extra (undeclared) fields included. Returns
    `default` when the key is absent. A cross-member validator on a group
    whose attributes type admits both forms needs exactly this.
    """
    if isinstance(attributes, Mapping):
        return attributes.get(key, default)
    if isinstance(attributes, BaseModel):
        for name, field in type(attributes).model_fields.items():
            if key == name or key in (
                field.alias,
                field.serialization_alias,
                field.validation_alias,
            ):
                return getattr(attributes, name)
        extra = attributes.__pydantic_extra__
        if extra is not None and key in extra:
            return extra[key]
    return default


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
