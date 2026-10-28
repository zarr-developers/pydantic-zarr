"""
Downstream typing conformance, checked by ty and pyright in CI.

This file holds the parameterizations downstream code writes — narrowed
specs, groups parameterized with them, and covariant widening to the Any*
aliases — so the typecheck job fails if they stop type-checking. ty and
pyright check it because the bugs it guards are variance bugs: pyrefly (the
project checker) is variance-lenient, so a bound refusing a narrowed spec,
or covariance regressing to invariance, would pass it unseen — the former
reached a release candidate once (zarr-developers/pydantic-zarr#190).

It is not a pytest module; it only needs to type-check.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

import pydantic_zarr.v2 as v2
import pydantic_zarr.v3 as v3

if TYPE_CHECKING:
    import numpy.typing as npt
    from zarr_metadata import (
        ZarrV2ArrayMetadataJSON,
        ZarrV2GroupMetadataJSON,
        ZarrV2ZArrayJSON,
        ZarrV2ZAttrsJSON,
        ZarrV2ZGroupJSON,
        ZarrV3ArrayMetadataJSON,
        ZarrV3GroupMetadataJSON,
    )


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


def narrowed_members_v3(group: GroupOfNarrowV3) -> Mapping[str, NarrowArrayV3]:
    """A narrowed members mapping reads back at its narrowed type."""
    return group.members


def narrowed_members_v2(group: GroupOfNarrowV2) -> Mapping[str, NarrowArrayV2]:
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


def from_array_keeps_self(data: npt.NDArray[Any]) -> NarrowArrayV3:
    """
    `from_array` on a narrowed class returns the narrowed type (`Self`).

    Its `attributes` parameter is deliberately `TBaseAttr` rather than `TAttr`
    (a covariant parameter may not appear in argument positions), so the
    argument is checked against the class's attribute type by validation, not
    statically — the trade that keeps `Self` returns for subclasses.
    """
    return NarrowArrayV3.from_array(data, attributes=Attrs())


def documents_v3(
    array: NarrowArrayV3, group: GroupOfUnionV3
) -> tuple[ZarrV3ArrayMetadataJSON, ZarrV3GroupMetadataJSON]:
    """`to_metadata_json` returns the typed metadata document, whatever the parameterization."""
    return array.to_metadata_json(), group.to_metadata_json()


def documents_v2(
    array: NarrowArrayV2, group: GroupOfUnionV2
) -> tuple[ZarrV2ArrayMetadataJSON, ZarrV2GroupMetadataJSON]:
    """Same contract on the v2 side."""
    return array.to_metadata_json(), group.to_metadata_json()


def stored_documents_v3(
    array: NarrowArrayV3, group: GroupOfUnionV3
) -> tuple[ZarrV3ArrayMetadataJSON, ZarrV3GroupMetadataJSON]:
    """`to_stored_metadata_json` types each store key's document."""
    return array.to_stored_metadata_json()["zarr.json"], group.to_stored_metadata_json()[
        "zarr.json"
    ]


def stored_documents_v2(
    array: NarrowArrayV2, group: GroupOfUnionV2
) -> tuple[ZarrV2ZArrayJSON, ZarrV2ZAttrsJSON, ZarrV2ZGroupJSON, ZarrV2ZAttrsJSON]:
    """Same contract on the v2 side, with `.zattrs` always present."""
    stored_array = array.to_stored_metadata_json()
    stored_group = group.to_stored_metadata_json()
    return (
        stored_array[".zarray"],
        stored_array[".zattrs"],
        stored_group[".zgroup"],
        stored_group[".zattrs"],
    )


def stored_member_documents(
    group_v3: GroupOfUnionV3, group_v2: GroupOfUnionV2
) -> tuple[v3.ZarrV3StoredDocumentJSON, v2.ZarrV2StoredDocumentJSON]:
    """A member's document, at a key the type does not name, is any stored document."""
    return (
        group_v3.to_stored_metadata_json()["a/zarr.json"],
        group_v2.to_stored_metadata_json()["a/.zattrs"],
    )


def stored_documents_as_mapping(
    array_v3: NarrowArrayV3,
    group_v3: GroupOfUnionV3,
    array_v2: NarrowArrayV2,
    group_v2: GroupOfUnionV2,
) -> tuple[
    Mapping[str, ZarrV3ArrayMetadataJSON],
    Mapping[str, v3.ZarrV3StoredDocumentJSON],
    Mapping[str, ZarrV2ZArrayJSON | ZarrV2ZAttrsJSON],
    Mapping[str, v2.ZarrV2StoredDocumentJSON],
]:
    """
    An array's stored type is closed, and a group's admits only stored
    documents beyond its own, so each is a `Mapping` of its value types.
    """
    return (
        array_v3.to_stored_metadata_json(),
        group_v3.to_stored_metadata_json(),
        array_v2.to_stored_metadata_json(),
        group_v2.to_stored_metadata_json(),
    )


def stored_layout_round_trip(
    group_v3: GroupOfUnionV3, group_v2: GroupOfUnionV2
) -> tuple[GroupOfUnionV3, GroupOfUnionV2, v3.AnyBaseGroupSpec]:
    """
    `from_stored_metadata_json` reads a stored layout back at the class it is
    called on, and a `GroupSpec`'s layout is a `BaseGroupSpec`'s.
    """
    return (
        GroupOfUnionV3.from_stored_metadata_json(group_v3.to_stored_metadata_json()),
        GroupOfUnionV2.from_stored_metadata_json(group_v2.to_stored_metadata_json()),
        v3.BaseGroupSpec.from_stored_metadata_json(group_v3.to_stored_metadata_json()),
    )
