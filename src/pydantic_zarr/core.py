from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
import numpy.typing as npt
import zarr
from pydantic import BaseModel, ConfigDict
from zarr.core.sync_group import get_node

if TYPE_CHECKING:
    from zarr.abc.store import Store

IncEx: TypeAlias = set[int] | set[str] | dict[int, Any] | dict[str, Any] | None

AccessMode: TypeAlias = Literal["w", "w+", "r", "a"]


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


def stringify_dtype(value: npt.DTypeLike) -> str | list[tuple[Any, ...]]:
    """
    Convert a `numpy.dtype` object into a `str`.

    Parameters
    ----------
    value : `npt.DTypeLike`
        Some object that can be coerced to a numpy dtype

    Returns
    -------

    A numpy dtype string representation of `value`.
    """
    # handle structured dtypes, which must be a list of tuples
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [tuple(v) for v in value]
    return np.dtype(value).str


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
        [ensure_member_name(key) for key in data]
    return data


def model_like(a: BaseModel, b: BaseModel, exclude: IncEx = None, include: IncEx = None) -> bool:
    """
    A similarity check for a pair pydantic.BaseModel, parametrized over included or excluded fields.


    """

    a_dict = a.model_dump(exclude=exclude, include=include)
    b_dict = b.model_dump(exclude=exclude, include=include)

    return a_dict == b_dict


# TODO: expose contains_array and contains_group as public functions in zarr-python
# and replace these custom implementations
def contains_array(store: Store, path: str) -> bool:
    try:
        return isinstance(get_node(store, path, zarr_format=2), zarr.Array)
    except FileNotFoundError:
        return False


def contains_group(store: Store, path: str) -> bool:
    try:
        return isinstance(get_node(store, path, zarr_format=2), zarr.Group)
    except FileNotFoundError:
        return False
