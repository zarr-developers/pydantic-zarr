"""
The members type parameter of `GroupSpec`, in v2 and v3: the second type
parameter is the type of the whole `members` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Union

import pytest
from pydantic import BaseModel, ValidationError
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


@VERSIONS
@pytest.mark.parametrize("kind", ["default", "mapping", "named"])
def test_members_type_parameter(module: ModuleType, named: type, kind: str) -> None:
    """
    The members parameter types a group's members in three ways. Left out, a
    group holds any arrays and groups. A `Mapping` constrains the type of
    every member. A closed `TypedDict` fixes the member names and the type of
    each. In every case the members validate from specs and from their dumped
    documents, and read back as specs.
    """
    array, group = _array(module), module.GroupSpec()
    cls, members = {
        "default": (module.GroupSpec, {"a": array, "g": group}),
        "mapping": (
            module.GroupSpec[Any, Mapping[str, module.ArraySpec]],
            {"a": array, "b": array},
        ),
        "named": (module.GroupSpec[Any, named], {"a": array, "g": group}),
    }[kind]

    spec = cls(attributes={}, members=members)

    assert spec.members == members
    assert cls.model_validate(spec.model_dump()) == spec


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


@pytest.mark.parametrize(("module", "name"), [(v2, "a/b"), (v3, "__reserved")], ids=["v2", "v3"])
def test_member_names_are_validated_under_a_members_type(module: ModuleType, name: str) -> None:
    """Narrowing the members type keeps the member-name rules of the format."""
    with pytest.raises(ValidationError):
        module.GroupSpec[Any, Mapping[str, module.ArraySpec]](members={name: _array(module)})


@VERSIONS
@pytest.mark.parametrize("spelling", ["array", "narrowed array", "union", "typing union"])
def test_member_type_as_members_parameter_is_refused(
    module: ModuleType, named: type, spelling: str
) -> None:
    """
    Up to 0.11 the second parameter was the type of one member. That spelling
    is refused where it is written, with an error that says what to write.
    """
    member_type = {
        "array": module.ArraySpec,
        "narrowed array": module.ArraySpec[Attrs],
        "union": module.ArraySpec | module.GroupSpec,
        "typing union": Union[module.ArraySpec, module.GroupSpec],
    }[spelling]
    with pytest.raises(TypeError, match=r"whole members\s+mapping"):
        module.GroupSpec[Any, member_type]
