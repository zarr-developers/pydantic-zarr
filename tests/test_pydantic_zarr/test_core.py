from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import BaseModel, Field, TypeAdapter

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
    from collections.abc import Callable, Mapping


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


class _AliasedAttrs(BaseModel):
    """An attribute model whose key is not a Python identifier, without `serialize_by_alias`."""

    fill: int = Field(alias="_FillValue")
    units: str | None = None


class _SpecHolder(BaseModel):
    """A user model that holds a spec as a field."""

    spec: GroupSpecV2 | GroupSpecV3


def _aliased_groups() -> list[GroupSpecV2 | GroupSpecV3]:
    attrs = _AliasedAttrs(_FillValue=1)
    return [
        GroupSpecV2(
            attributes=attrs,
            members={"x": ArraySpecV2(shape=(4,), chunks=(4,), dtype="<f4", attributes=attrs)},
        ),
        GroupSpecV3(
            attributes=attrs,
            members={
                "x": ArraySpecV3.create_default(shape=(4,), data_type="float32", attributes=attrs)
            },
        ),
    ]


@pytest.mark.parametrize("spec", _aliased_groups(), ids=["v2", "v3"])
@pytest.mark.parametrize(
    ("dump", "expected"),
    [
        (lambda spec: spec.model_dump(), {"_FillValue": 1, "units": None}),
        (lambda spec: spec.model_dump(by_alias=None), {"_FillValue": 1, "units": None}),
        (lambda spec: json.loads(spec.model_dump_json()), {"_FillValue": 1, "units": None}),
        (
            lambda spec: TypeAdapter(type(spec)).dump_python(spec),
            {"_FillValue": 1, "units": None},
        ),
        (
            lambda spec: _SpecHolder(spec=spec).model_dump()["spec"],
            {"_FillValue": 1, "units": None},
        ),
        (lambda spec: spec.model_dump(exclude_none=True), {"_FillValue": 1}),
        (
            lambda spec: spec.model_dump(
                exclude={"attributes": {"units"}, "members": {"x": {"attributes": {"units"}}}}
            ),
            {"_FillValue": 1},
        ),
        (lambda spec: spec.model_dump(by_alias=False), {"fill": 1, "units": None}),
    ],
    ids=[
        "model_dump",
        "by_alias_none",
        "model_dump_json",
        "type_adapter",
        "held_by_a_model",
        "exclude_none",
        "nested_exclude",
        "by_alias_false",
    ],
)
def test_attribute_aliases_on_every_dump_path(
    spec: GroupSpecV2 | GroupSpecV3,
    dump: Callable[[GroupSpecV2 | GroupSpecV3], dict[str, Any]],
    expected: dict[str, object],
) -> None:
    """
    Every way of dumping a spec writes an attribute model under its aliases,
    at the root and in its members, unless the dump says `by_alias=False`;
    the dump's other settings reach into the attributes.
    """
    document = dump(spec)
    assert document["attributes"] == expected
    assert document["members"]["x"]["attributes"] == expected


class _ImageLabel(BaseModel):
    label: str = Field(alias="image-label")


@pytest.mark.parametrize("spec_cls", [GroupSpecV2, GroupSpecV3])
@pytest.mark.parametrize(
    ("dump", "expected"),
    [
        (lambda spec: spec.model_dump(), {"image-label": "x"}),
        (lambda spec: json.loads(spec.model_dump_json()), {"image-label": "x"}),
        (lambda spec: TypeAdapter(type(spec)).dump_python(spec), {"image-label": "x"}),
        (lambda spec: spec.model_dump(by_alias=False), {"label": "x"}),
    ],
    ids=["model_dump", "model_dump_json", "type_adapter", "by_alias_false"],
)
def test_attribute_mapping_models_dumped_by_alias(
    spec_cls: type[GroupSpecV2 | GroupSpecV3],
    dump: Callable[[GroupSpecV2 | GroupSpecV3], dict[str, Any]],
    expected: dict[str, object],
) -> None:
    """A model held in an attributes mapping is dumped under its aliases too."""
    spec = spec_cls(attributes={"ome": _ImageLabel(**{"image-label": "x"})}, members={})
    assert dump(spec)["attributes"] == {"ome": expected}


class _Attrs(AttributesBase):
    units: str


# how a spec with extra fields types the members it does not declare: as an
# extra field's value, in place of an `extra_fields` property
_EXTRA_MEMBERS = {
    "additionalProperties": {"$ref": "#/$defs/JSONValue"},
    "propertyNames": {"not": {"const": "extra_fields"}},
}
_CLOSED = {"additionalProperties": False}


@pytest.mark.parametrize(
    ("model", "properties", "undeclared"),
    [
        pytest.param(
            ArraySpecV2, {"shape", "chunks", "dtype", "attributes"}, _EXTRA_MEMBERS, id="v2-array"
        ),
        pytest.param(
            ArraySpecV3,
            {"shape", "data_type", "codecs", "attributes"},
            _EXTRA_MEMBERS,
            id="v3-array",
        ),
        pytest.param(BaseGroupSpecV2, {"attributes", "zarr_format"}, _CLOSED, id="v2-base-group"),
        pytest.param(
            BaseGroupSpecV3, {"attributes", "node_type"}, _EXTRA_MEMBERS, id="v3-base-group"
        ),
        pytest.param(GroupSpecV2, {"attributes", "members"}, _CLOSED, id="v2-group"),
        pytest.param(GroupSpecV3, {"attributes", "members"}, _EXTRA_MEMBERS, id="v3-group"),
        pytest.param(_Attrs, {"units"}, {}, id="attributes"),
    ],
)
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_json_schema(
    model: type[BaseModel],
    properties: set[str],
    undeclared: Mapping[str, object],
    mode: Literal["validation", "serialization"],
) -> None:
    """
    Both JSON schema modes describe the documents a model reads and writes,
    and agree: the serialization mode describes the model's fields rather
    than collapsing to an open object, since the specs' wrap model
    serializers carry no return annotation for pydantic to use in their
    place. A spec with extra fields holds them as top-level members, typed
    as an extra field's value, and never as an `extra_fields` member, at the
    top level or nested (a group's members); a spec without them is closed,
    and an attributes model stays open to any key.
    """
    schema = model.model_json_schema(mode=mode)
    assert schema == model.model_json_schema(mode="validation")
    definition = schema["$defs"][model.__name__] if "$ref" in schema else schema
    assert properties <= set(definition["properties"])
    assert {
        key: definition[key]
        for key in ("additionalProperties", "propertyNames")
        if key in definition
    } == undeclared
    nested = [definition, *schema.get("$defs", {}).values()]
    assert not [d for d in nested if "extra_fields" in d.get("properties", {})]
