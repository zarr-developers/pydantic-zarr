"""
The members type parameter of `GroupSpec`, in v2 and v3: the second type
parameter is the type of the whole `members` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Generic, NotRequired, Union

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import TypedDict

from pydantic_zarr import v2, v3

if TYPE_CHECKING:
    from types import ModuleType


class NamedV2(TypedDict, closed=True):
    """A v2 group with exactly an array `a` and a group `g`."""

    a: v2.ArraySpec
    g: v2.GroupSpec


class NamedV3(TypedDict, closed=True):
    """A v3 group with exactly an array `a` and a group `g`."""

    a: v3.ArraySpec
    g: v3.GroupSpec


class Attrs(BaseModel):
    units: str = "1"


VERSIONS = pytest.mark.parametrize(
    ("module", "named"), [(v2, NamedV2), (v3, NamedV3)], ids=["v2", "v3"]
)


def _array(module: ModuleType) -> Any:
    if module is v2:
        return v2.ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    return v3.ArraySpec.create_default(shape=(4,), data_type="uint8")


class OptionalNamesV2(TypedDict, closed=True, total=False):
    a: v2.ArraySpec
    g: v2.GroupSpec


class OptionalNamesV3(TypedDict, closed=True, total=False):
    a: v3.ArraySpec
    g: v3.GroupSpec


class NamedAndExtraV2(TypedDict, extra_items=v2.ArraySpec):
    # its own configuration, so that the group's `extra="forbid"` is not
    # offered to a TypedDict that allows extra items (pydantic warns of that)
    __pydantic_config__ = ConfigDict()  # type: ignore[misc]

    g: NotRequired[v2.GroupSpec]


class NamedAndExtraV3(TypedDict, extra_items=v3.ArraySpec):
    # its own configuration, so that the group's `extra="forbid"` is not
    # offered to a TypedDict that allows extra items (pydantic warns of that)
    __pydantic_config__ = ConfigDict()  # type: ignore[misc]

    g: NotRequired[v3.GroupSpec]


class DatasetV2(v2.GroupSpec[Any, v2.TMembers], Generic[v2.TMembers]):
    """A generic subclass that passes the members type through."""


class DatasetV3(v3.GroupSpec[Any, v3.TMembers], Generic[v3.TMembers]):
    """A generic subclass that passes the members type through."""


OPTIONAL_NAMES = {v2: OptionalNamesV2, v3: OptionalNamesV3}
NAMED_AND_EXTRA = {v2: NamedAndExtraV2, v3: NamedAndExtraV3}
DATASET: dict[Any, Any] = {v2: DatasetV2, v3: DatasetV3}

KINDS = ["default", "mapping", "named", "optional names", "named and extra", "subclass"]


def _members_case(module: ModuleType, named: type, kind: str) -> tuple[Any, dict[str, Any]]:
    """A group class whose members are typed in the way `kind` names, with members it accepts."""
    array, group = _array(module), module.GroupSpec()
    return {
        "default": (module.GroupSpec, {"a": array, "g": group}),
        "mapping": (
            module.GroupSpec[Any, Mapping[str, module.ArraySpec]],
            {"a": array, "b": array},
        ),
        "named": (module.GroupSpec[Any, named], {"a": array, "g": group}),
        "optional names": (module.GroupSpec[Any, OPTIONAL_NAMES[module]], {"a": array}),
        "named and extra": (
            module.GroupSpec[Any, NAMED_AND_EXTRA[module]],
            {"g": group, "s0": array, "s1": array},
        ),
        "subclass": (DATASET[module][named], {"a": array, "g": group}),
    }[kind]


@VERSIONS
@pytest.mark.parametrize("kind", KINDS)
def test_members_type_parameter(module: ModuleType, named: type, kind: str) -> None:
    """
    The members parameter types a group's members in several ways. Left out,
    a group holds any arrays and groups. A `Mapping` constrains the type of
    every member. A closed `TypedDict` fixes the member names and the type of
    each; its names may be optional, and `extra_items` types the members it
    does not name. A generic subclass passes the members type through.

    In every case the members validate from specs and from their dumped
    documents, read back as specs, and survive the operations that build a
    group from another form: a flat mapping of nodes, the stored metadata
    documents, and `replace`.
    """
    cls, members = _members_case(module, named, kind)

    spec = cls(attributes={}, members=members)

    assert spec.members == members
    assert all(
        isinstance(member, module.ArraySpec | module.GroupSpec) for member in spec.members.values()
    )
    assert cls.model_validate(spec.model_dump()) == spec
    assert cls.from_flat(spec.to_flat()) == spec
    assert cls.from_stored_metadata_json(spec.to_stored_metadata_json()) == spec
    assert spec.replace(attributes={"k": 1}).members == members
    assert spec.like(spec.replace(members=members))
    assert spec.get_path(next(iter(members))) == next(iter(members.values()))


@VERSIONS
def test_any_members_type_copies_the_members(module: ModuleType, named: type) -> None:
    """
    `Any` as the members type means any mapping. The spec holds its own copy
    of the members, so a later change to the mapping it was built from does
    not reach it.
    """
    members = {"a": _array(module)}
    spec = module.GroupSpec[Any, Any](members=members)
    members["b/c"] = _array(module)
    assert list(spec.members) == ["a"]


@VERSIONS
def test_named_members_refuse_an_undeclared_member(module: ModuleType, named: type) -> None:
    """A closed `TypedDict` of members refuses a member it does not declare."""
    array, group = _array(module), module.GroupSpec()
    with pytest.raises(ValidationError, match="extra"):
        module.GroupSpec[Any, named](members={"a": array, "g": group, "extra": array})


@VERSIONS
def test_named_members_refuse_a_missing_member(module: ModuleType, named: type) -> None:
    """A `TypedDict` of members refuses a group that lacks a required member."""
    with pytest.raises(ValidationError, match=r"members\.g\n\s+Field required"):
        module.GroupSpec[Any, named](members={"a": _array(module)})


@VERSIONS
def test_named_members_are_required(module: ModuleType, named: type) -> None:
    """
    The default of `members`, an empty mapping, is validated against the
    members type, so a group whose members type requires a member cannot
    leave `members` out.
    """
    with pytest.raises(ValidationError, match="Field required"):
        module.GroupSpec[Any, named]()


@VERSIONS
def test_mapping_members_refuse_a_member_of_another_type(module: ModuleType, named: type) -> None:
    """A `Mapping` members type refuses a member that is not of its value type."""
    with pytest.raises(ValidationError, match="members"):
        module.GroupSpec[Any, Mapping[str, module.ArraySpec]](members={"g": module.GroupSpec()})


@VERSIONS
@pytest.mark.parametrize("members", [["a"], None, {1: "a"}], ids=["list", "none", "integer name"])
def test_members_are_a_mapping_under_any_members_type(
    module: ModuleType, named: type, members: object
) -> None:
    """
    `members` is a mapping with string keys whatever the members type is.
    pydantic does not enforce a type parameter's bound, so `GroupSpec[Any, Any]`
    would otherwise accept any value.
    """
    with pytest.raises(ValidationError, match="member"):
        module.GroupSpec[Any, Any](members=members)


@pytest.mark.parametrize(("module", "name"), [(v2, "a/b"), (v3, "__reserved")], ids=["v2", "v3"])
def test_member_names_are_validated_under_a_members_type(module: ModuleType, name: str) -> None:
    """Narrowing the members type keeps the member-name rules of the format."""
    with pytest.raises(ValidationError):
        module.GroupSpec[Any, Mapping[str, module.ArraySpec]](members={name: _array(module)})


MEMBER_TYPE_SPELLINGS = [
    "array",
    "narrowed array",
    "union",
    "typing union",
    "array alias",
    "item alias",
    "union of aliases",
    "annotated array",
    "array by name",
]


def _member_type(module: ModuleType, spelling: str) -> Any:
    return {
        "array": module.ArraySpec,
        "narrowed array": module.ArraySpec[Attrs],
        "union": module.ArraySpec | module.GroupSpec,
        "typing union": Union[module.ArraySpec, module.GroupSpec],
        "array alias": module.AnyArraySpec,
        "item alias": module.TBaseItem,
        "union of aliases": module.AnyGroupSpec | module.AnyArraySpec,
        "annotated array": Annotated[module.ArraySpec, "a note"],
        "array by name": "ArraySpec",
    }[spelling]


@VERSIONS
@pytest.mark.parametrize("spelling", MEMBER_TYPE_SPELLINGS)
def test_member_type_as_members_type_is_deprecated(
    module: ModuleType, named: type, spelling: str
) -> None:
    """
    Up to 0.11 the second parameter was the type of one member, and released
    packages spell it that way. A member type there still works, read as a
    mapping of that type, with a `DeprecationWarning` that says what to write.
    It is recognized through unions, aliases, `Annotated` and names given as
    strings.
    """
    with pytest.warns(DeprecationWarning, match=r"Mapping\[str, <member type>\]"):
        cls = module.GroupSpec[Any, _member_type(module, spelling)]

    spec = cls(members={"a": _array(module)})
    assert isinstance(spec.members["a"], module.ArraySpec)
    with pytest.raises(ValidationError, match="members"):
        cls(members={"a": 1})


@VERSIONS
@pytest.mark.parametrize("spelling", ["list of arrays", "optional mapping", "a scalar type"])
def test_non_mapping_members_type_is_refused(
    module: ModuleType, named: type, spelling: str
) -> None:
    """A members type that is neither a mapping nor a member type is refused
    where it is written, with an error that says what to write."""
    members_type = {
        "list of arrays": list[module.ArraySpec],
        "optional mapping": Mapping[str, module.ArraySpec] | None,
        "a scalar type": int,
    }[spelling]
    with pytest.raises(TypeError, match=r"whole members\s+mapping"):
        module.GroupSpec[Any, members_type]


@VERSIONS
def test_unresolved_members_type_is_passed_through(module: ModuleType, named: type) -> None:
    """A members type the check cannot tell apart, such as a name that does
    not resolve, is not refused at the parameterization."""
    assert module.GroupSpec[Any, "NotDefinedAnywhere"] is not None
