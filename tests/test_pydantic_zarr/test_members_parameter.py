"""
The members type parameter of `GroupSpec`, in v2 and v3: the second type
parameter is the type of the whole `members` mapping.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, NotRequired

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, with_config
from typing_extensions import TypedDict, TypeVar

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


# its own configuration, so that the group's `extra="forbid"` is not offered
# to a TypedDict that allows extra items (pydantic warns of that)
@with_config(ConfigDict())
class NamedAndExtraV2(TypedDict, extra_items=v2.ArraySpec):
    g: NotRequired[v2.GroupSpec]


# its own configuration, so that the group's `extra="forbid"` is not offered
# to a TypedDict that allows extra items (pydantic warns of that)
@with_config(ConfigDict())
class NamedAndExtraV3(TypedDict, extra_items=v3.ArraySpec):
    g: NotRequired[v3.GroupSpec]


class DatasetV2(v2.GroupSpec[v2.TBaseAttr, v2.TMembers], Generic[v2.TMembers]):
    """A generic subclass that passes the members type through."""


class DatasetV3(v3.GroupSpec[v3.TBaseAttr, v3.TMembers], Generic[v3.TMembers]):
    """A generic subclass that passes the members type through."""


TOwnMembersV2 = TypeVar("TOwnMembersV2", bound=v2.TBaseMembers)
TOwnMembersV3 = TypeVar("TOwnMembersV3", bound=v3.TBaseMembers)


class OwnDatasetV2(v2.GroupSpec[v2.TBaseAttr, TOwnMembersV2], Generic[TOwnMembersV2]):
    """A generic subclass with a members type variable of its own."""


class OwnDatasetV3(v3.GroupSpec[v3.TBaseAttr, TOwnMembersV3], Generic[TOwnMembersV3]):
    """A generic subclass with a members type variable of its own."""


# parameterized at module level, as a group class usually is, so that pydantic
# registers the classes where pickle looks them up
NamedGroupV2 = v2.GroupSpec[v2.TBaseAttr, NamedV2]
NamedGroupV3 = v3.GroupSpec[v3.TBaseAttr, NamedV3]

OPTIONAL_NAMES = {v2: OptionalNamesV2, v3: OptionalNamesV3}
OWN_DATASET: dict[Any, Any] = {v2: OwnDatasetV2, v3: OwnDatasetV3}
NAMED_GROUP: dict[Any, Any] = {v2: NamedGroupV2, v3: NamedGroupV3}
NAMED_AND_EXTRA = {v2: NamedAndExtraV2, v3: NamedAndExtraV3}
DATASET: dict[Any, Any] = {v2: DatasetV2, v3: DatasetV3}

KINDS = [
    "default",
    "mapping",
    "named",
    "optional names",
    "named and extra",
    "subclass",
    "subclass with its own type variable",
]


def _members_case(module: ModuleType, named: type, kind: str) -> tuple[Any, dict[str, Any]]:
    """A group class whose members are typed in the way `kind` names, with members it accepts."""
    array, group = _array(module), module.GroupSpec()
    return {
        "default": (module.GroupSpec, {"a": array, "g": group}),
        "mapping": (
            module.GroupSpec[module.TBaseAttr, Mapping[str, module.ArraySpec]],
            {"a": array, "b": array},
        ),
        "named": (module.GroupSpec[module.TBaseAttr, named], {"a": array, "g": group}),
        "optional names": (
            module.GroupSpec[module.TBaseAttr, OPTIONAL_NAMES[module]],
            {"a": array},
        ),
        "named and extra": (
            module.GroupSpec[module.TBaseAttr, NAMED_AND_EXTRA[module]],
            {"g": group, "s0": array, "s1": array},
        ),
        "subclass": (DATASET[module][named], {"a": array, "g": group}),
        "subclass with its own type variable": (
            OWN_DATASET[module][named],
            {"a": array, "g": group},
        ),
    }[kind]


@VERSIONS
@pytest.mark.parametrize("kind", KINDS)
def test_members_type_parameter(module: ModuleType, named: type, kind: str) -> None:
    """
    The members parameter types a group's members in several ways. Left out,
    a group holds any arrays and groups. A `Mapping` constrains the type of
    every member. A closed `TypedDict` fixes the member names and the type of
    each; its names may be optional, and `extra_items` types the members it
    does not name. A generic subclass passes the members type through, with
    the class's type variable or one of its own.

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
def test_parameterized_group_pickles(module: ModuleType, named: type) -> None:
    """A group class parameterized at module level round-trips through pickle."""
    spec = NAMED_GROUP[module](members={"a": _array(module), "g": module.GroupSpec()})
    assert pickle.loads(pickle.dumps(spec)) == spec


@VERSIONS
def test_group_holds_its_own_members_dict(module: ModuleType, named: type) -> None:
    """
    A group holds a `dict` copy of the mapping it is given, also under `Any`,
    where pydantic would otherwise keep the caller's object: a later change to
    that mapping does not reach the frozen spec, and a mapping type pydantic
    cannot serialize still dumps to JSON.
    """
    given = {"a": _array(module)}
    spec = module.GroupSpec[Any, Any](members=MappingProxyType(given))
    given["b"] = _array(module)

    assert type(spec.members) is dict
    assert list(spec.members) == ["a"]
    assert '"a"' in spec.model_dump_json()


@VERSIONS
def test_named_members_refuse_an_undeclared_member(module: ModuleType, named: type) -> None:
    """A closed `TypedDict` of members refuses a member it does not declare."""
    array, group = _array(module), module.GroupSpec()
    with pytest.raises(ValidationError, match="extra"):
        module.GroupSpec[module.TBaseAttr, named](members={"a": array, "g": group, "extra": array})


@VERSIONS
def test_named_members_refuse_a_missing_member(module: ModuleType, named: type) -> None:
    """A `TypedDict` of members refuses a group that lacks a required member."""
    with pytest.raises(ValidationError, match=r"members\.g\n\s+Field required"):
        module.GroupSpec[module.TBaseAttr, named](members={"a": _array(module)})


@VERSIONS
def test_named_members_are_required(module: ModuleType, named: type) -> None:
    """
    The default of `members`, an empty mapping, is validated against the
    members type, so a group whose members type requires a member cannot
    leave `members` out.
    """
    with pytest.raises(ValidationError, match="Field required"):
        module.GroupSpec[module.TBaseAttr, named]()


@VERSIONS
def test_mapping_members_refuse_a_member_of_another_type(module: ModuleType, named: type) -> None:
    """A `Mapping` members type refuses a member that is not of its value type."""
    with pytest.raises(ValidationError, match="members"):
        module.GroupSpec[module.TBaseAttr, Mapping[str, module.ArraySpec]](
            members={"g": module.GroupSpec()}
        )


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
        module.GroupSpec[module.TBaseAttr, Mapping[str, module.ArraySpec]](
            members={name: _array(module)}
        )


@VERSIONS
@pytest.mark.parametrize("case", ["an array type", "a union, with no members"])
def test_member_type_as_members_type_fails_validation(
    module: ModuleType, named: type, case: str
) -> None:
    """
    Up to 0.11 the second parameter was the type of one member. A class still
    written that way validates its whole members mapping as one member, and
    is refused. With a union that holds a group type, an empty mapping would
    pass as an empty group; the check that `members` is a mapping refuses it.
    """
    members_type, members = {
        "an array type": (module.ArraySpec, {"a": _array(module)}),
        "a union, with no members": (module.ArraySpec | module.GroupSpec, {}),
    }[case]
    with pytest.raises(ValidationError, match="members"):
        module.GroupSpec[module.TBaseAttr, members_type](members=members)
