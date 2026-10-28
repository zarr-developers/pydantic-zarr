"""
Downstream typing conformance, checked by pyright in CI.

This file holds the parameterizations downstream code writes, so the
typecheck job fails if they stop type-checking. It is checked by pyright
because the bugs it guards are variance bugs: pyrefly is variance-lenient
and mypy does not apply PEP 696 TypeVar defaults inside the lazy `type`
aliases, so neither sees an invariant bound refuse a narrowed spec — the
regression that reached a release candidate once (zarr-developers/pydantic-zarr#190).

It is not a pytest module; it only needs to type-check.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

import pydantic_zarr.v2 as v2
import pydantic_zarr.v3 as v3


class Attrs(BaseModel):
    """A narrowed attribute model, as downstreams write them."""

    units: str = "1"


class NarrowArrayV3(v3.ArraySpec[Attrs]):
    """A narrowed v3 array spec."""


class NarrowArrayV2(v2.ArraySpec[Attrs]):
    """A narrowed v2 array spec."""


# a group may be parameterized with narrowed member specs, alone or in unions
# with the Any-parameterized classes
GroupOfNarrowV3 = v3.GroupSpec[Mapping[str, object], NarrowArrayV3]
GroupOfUnionV3 = v3.GroupSpec[Attrs, v3.GroupSpec[Any, Any] | NarrowArrayV3]
GroupOfNarrowV2 = v2.GroupSpec[Mapping[str, object], NarrowArrayV2]
GroupOfUnionV2 = v2.GroupSpec[Attrs, v2.GroupSpec[Any, Any] | NarrowArrayV2]


def narrowed_members_v3(group: GroupOfNarrowV3) -> Mapping[str, NarrowArrayV3] | None:
    """A narrowed members mapping reads back at its narrowed type."""
    return group.members


def narrowed_members_v2(group: GroupOfNarrowV2) -> Mapping[str, NarrowArrayV2] | None:
    """Same contract on the v2 side."""
    return group.members


# The spec parameters are covariant (specs are frozen), so a narrowed spec is
# substitutable wherever a wider one is expected — the Any* aliases are
# ordinary wide types, and reads through them stay checked.


def widened_array_v3(spec: NarrowArrayV3) -> v3.AnyArraySpec:
    """A narrowed v3 array spec IS-A spec of the widest specialization."""
    return spec


def widened_group_v3(group: GroupOfNarrowV3) -> v3.AnyGroupSpec:
    """A narrowed v3 group spec widens too, members included."""
    return group


def widened_array_v2(spec: NarrowArrayV2) -> v2.AnyArraySpec:
    """Same contract on the v2 side."""
    return spec


def checked_read_through_wide_type(spec: v3.AnyArraySpec) -> Mapping[str, object] | BaseModel:
    """Attributes read through the wide type keep a checked type, not Any."""
    return spec.attributes
