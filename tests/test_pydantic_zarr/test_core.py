from __future__ import annotations

import pytest

from pydantic_zarr.core import ensure_member_name, json_eq, model_like, tuplify_json
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3


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
def test_tuplify_json(input_obj: object, expected_output: object) -> None:
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
