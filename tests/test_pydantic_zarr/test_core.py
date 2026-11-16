from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pydantic_zarr.core import (
    AttributesBase,
    ensure_member_name,
    json_eq,
    model_like,
    tuplify_json,
)
from pydantic_zarr.v2 import ArraySpec as ArraySpecV2
from pydantic_zarr.v2 import BaseGroupSpec as BaseGroupSpecV2
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import ArraySpec as ArraySpecV3
from pydantic_zarr.v3 import BaseGroupSpec as BaseGroupSpecV3
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import BaseModel


@pytest.mark.parametrize("data", ["/", "///", "a/b/", "a/b/vc"])
def test_parse_str_no_path(data: str) -> None:
    with pytest.raises(ValueError, match=r'Strings containing "/" are invalid\.'):
        ensure_member_name(data)


@pytest.mark.parametrize(
    ("input_obj", "expected_output"),
    [
        ({"key": [1, 2, 3]}, {"key": (1, 2, 3)}),
        ([1, [2, 3], 4], (1, (2, 3), 4)),
        ({"nested": {"list": [1, 2]}}, {"nested": {"list": (1, 2)}}),
        ([{"a": [1, 2]}, {"b": 3}], ({"a": (1, 2)}, {"b": 3})),
        ([], ()),
    ],
)
def test_tuplify_json(input_obj: Mapping[str, Any] | list[Any], expected_output: object) -> None:
    """
    Test that tuplify_json converts lists to tuples, with recursion inside sequences
    and dictionaries.
    """
    assert tuplify_json(input_obj) == expected_output


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # lists and tuples that would serialize to identical JSON are equal
        ([1, 2, 3], (1, 2, 3)),
        ((1, 2, 3), [1, 2, 3]),
        # recursion into nested sequences
        ([1, [2, 3], 4], (1, (2, 3), 4)),
        # recursion into mappings, with list/tuple values
        ({"foo": [1, 2, 3]}, {"foo": (1, 2, 3)}),
        ({"a": {"b": [1, 2]}}, {"a": {"b": (1, 2)}}),
        # mix of nested mappings and sequences
        ({"a": [{"b": [1, 2]}]}, {"a": ({"b": (1, 2)},)}),
        # scalars fall back to regular equality
        (1, 1),
        ("x", "x"),
        (None, None),
        # empty containers
        ([], ()),
        ({}, {}),
    ],
)
def test_json_eq_equal(a: object, b: object) -> None:
    """
    Lists and tuples are treated as equivalent, recursing into mappings and sequences.
    """
    assert json_eq(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # sequences of unequal length
        ([1, 2, 3], [1, 2]),
        ([1, 2], (1, 2, 3)),
        # differing element values
        ([1, 2, 3], [1, 2, 4]),
        ({"foo": [1, 2]}, {"foo": [1, 3]}),
        # mismatched container types (sequence vs mapping)
        ([1, 2], {"a": 1}),
        ({"a": 1}, [1, 2]),
        # differing mapping keys
        ({"a": 1}, {"b": 1}),
        ({"a": 1}, {"a": 1, "b": 2}),
        # differing scalars
        (1, 2),
        ("x", "y"),
    ],
)
def test_json_eq_unequal(a: object, b: object) -> None:
    """
    Inequality is reported for differing lengths, values, keys, or container kinds.
    """
    assert not json_eq(a, b)


@pytest.mark.parametrize("group_spec_cls", [GroupSpecV2, GroupSpecV3])
def test_model_like_list_vs_tuple_attrs(group_spec_cls: type) -> None:
    """
    Specs whose attributes differ only by list-vs-tuple (e.g. one constructed directly and
    one read back from storage where ``tuplify_json`` converted lists to tuples) compare as
    ``like`` each other.
    """
    a = group_spec_cls(attributes={"foo": [1, 2, 3], "bar": {"baz": [4, 5]}})
    b = group_spec_cls(attributes={"foo": (1, 2, 3), "bar": {"baz": (4, 5)}})
    assert model_like(a, b)
    assert a.like(b)
    assert b.like(a)


def test_attributes_base() -> None:
    """
    `AttributesBase` serializes by alias, keeps unknown keys, and leaves
    fields at the `MISSING` sentinel out of the dump; an assigned value —
    `None` included, where the annotation admits it — keeps its key.
    """
    from pydantic import Field

    from pydantic_zarr.core import MISSING, AttributesBase

    class CF(AttributesBase):
        fill: float | MISSING = Field(default=MISSING, alias="_FillValue")
        units: str | MISSING = MISSING
        path: str | MISSING | None = MISSING

    assert CF().model_dump() == {}
    assert CF(path=None).model_dump() == {"path": None}
    assert CF(**{"_FillValue": 1.5, "proj:code": "x"}).model_dump() == {
        "_FillValue": 1.5,
        "proj:code": "x",
    }
    assert CF.model_validate({"_FillValue": 1.5}).units is MISSING


def test_attributes_base_refuses_none_outside_annotation() -> None:
    """A `None` assigned to a field whose annotation has no `None` is a validation error."""
    import pytest
    from pydantic import ValidationError

    from pydantic_zarr.core import MISSING, AttributesBase

    class Attrs(AttributesBase):
        unit: str | MISSING = MISSING

    assert Attrs().model_dump() == {}
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Attrs(unit=None)


def test_get_attribute() -> None:
    """
    `get_attribute` reads a key from attributes in either form: mappings by
    key, models by field name or alias (extra fields included), with a default
    for absent keys.
    """
    from pydantic import Field

    from pydantic_zarr.core import MISSING, AttributesBase, get_attribute

    class CF(AttributesBase):
        fill: float | MISSING | None = Field(default=MISSING, alias="_FillValue")

    model = CF(**{"_FillValue": 1.5, "proj:code": "EPSG:32632"})
    assert get_attribute(model, "_FillValue") == 1.5
    assert get_attribute(model, "fill") == 1.5
    assert get_attribute(model, "proj:code") == "EPSG:32632"
    assert get_attribute(model, "missing", "default") == "default"
    assert get_attribute({"k": 2}, "k") == 2
    assert get_attribute({"k": 2}, "zz") is None
    # a MISSING field would not appear in the serialized document, so it reads
    # as absent, agreeing with the mapping form of the same document; an
    # assigned None is a present key holding null, and reads as None
    assert get_attribute(CF(), "_FillValue", 0.0) == 0.0
    assert get_attribute(CF(fill=None), "_FillValue", 0.0) is None

    from pydantic import AliasChoices

    class Choosy(AttributesBase):
        name_field: str | MISSING = Field(default=MISSING, validation_alias=AliasChoices("a", "b"))

    assert get_attribute(Choosy(a="x"), "b", "dflt") == "x"
    assert get_attribute(Choosy(), "a", "dflt") == "dflt"


class _Attrs(AttributesBase):
    units: str


@pytest.mark.parametrize(
    ("model", "properties", "open_"),
    [
        pytest.param(
            ArraySpecV2,
            {"shape", "chunks", "dtype", "attributes", "extra_fields"},
            False,
            id="v2-array",
        ),
        pytest.param(
            ArraySpecV3,
            {"shape", "data_type", "codecs", "attributes", "extra_fields"},
            False,
            id="v3-array",
        ),
        pytest.param(BaseGroupSpecV2, {"attributes", "zarr_format"}, False, id="v2-base-group"),
        pytest.param(
            BaseGroupSpecV3, {"attributes", "node_type", "extra_fields"}, False, id="v3-base-group"
        ),
        pytest.param(GroupSpecV2, {"attributes", "members"}, False, id="v2-group"),
        pytest.param(GroupSpecV3, {"attributes", "members", "extra_fields"}, False, id="v3-group"),
        pytest.param(_Attrs, {"units"}, True, id="attributes"),
    ],
)
def test_serialization_json_schema(
    model: type[BaseModel], properties: set[str], *, open_: bool
) -> None:
    """
    The serialization-mode JSON schema describes the model's fields, as the
    validation-mode one does, rather than collapsing to an open object: the
    specs' wrap model serializers carry no return annotation for pydantic to
    use in their place. An attributes model stays open to extra keys.
    """
    schema = model.model_json_schema(mode="serialization")
    assert schema == model.model_json_schema(mode="validation")
    definition = schema["$defs"][model.__name__] if "$ref" in schema else schema
    assert properties <= set(definition["properties"])
    assert definition.get("additionalProperties", True) is open_
