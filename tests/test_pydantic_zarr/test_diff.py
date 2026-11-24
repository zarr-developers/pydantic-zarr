"""Tests for `json_patch`, and for `diff` on v2 and v3 array and group specs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import jsonpatch
import pytest
from pydantic import BaseModel

from pydantic_zarr.core import NodeView, _json_equal, json_patch, node_patch
from pydantic_zarr.v2 import ArraySpec as ArraySpecV2
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import ArraySpec, BaseGroupSpec, CodecLike, GroupSpec

if TYPE_CHECKING:
    from pydantic_zarr.core import PatchOperation


def _applied(document: object, patch: list[PatchOperation]) -> object:
    """Apply `patch` to `document` with `jsonpatch`, an independent RFC 6902 implementation."""
    # a JSON round trip turns tuples into lists, which jsonpatch can change in place
    return jsonpatch.apply_patch(json.loads(json.dumps(document)), patch)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"a": 1}, {"a": 1}, []),
        ({"a": (1, 2)}, {"a": [1, 2]}, []),
        ({"a": 1}, {"a": 2}, [{"op": "replace", "path": "/a", "value": 2}]),
        ({"a": 1}, {}, [{"op": "remove", "path": "/a"}]),
        ({}, {"a": None}, [{"op": "add", "path": "/a", "value": None}]),
        ({"a": [1, 2]}, {"a": [1, 3]}, [{"op": "replace", "path": "/a/1", "value": 3}]),
        ({"a": [1, 2]}, {"a": [1]}, [{"op": "replace", "path": "/a", "value": [1]}]),
        ({"a": {"b": 1}}, {"a": {"b": True}}, [{"op": "replace", "path": "/a/b", "value": True}]),
        ({"a/b": 1, "c~d": 1}, {"a/b": 2}, [
            {"op": "replace", "path": "/a~1b", "value": 2},
            {"op": "remove", "path": "/c~0d"},
        ]),
        (1, "1", [{"op": "replace", "path": "", "value": "1"}]),
        ({"a": float("nan")}, {"a": float("nan")}, []),
        ({"a": 0.0}, {"a": -0.0}, [{"op": "replace", "path": "/a", "value": -0.0}]),
        # numbers compare by value; a bool is not a number
        ({"a": 1}, {"a": 1.0}, []),
        ({"a": 1}, {"a": True}, [{"op": "replace", "path": "/a", "value": True}]),
        ({"a": 10**400}, {"a": 10**400}, []),
        ({"a": 2**53 + 1}, {"a": float(2**53)}, [{"op": "replace", "path": "/a", "value": 2.0**53}]),
        ({"a": None}, {"a": 0}, [{"op": "replace", "path": "/a", "value": 0}]),
    ],
)
def test_json_patch(left: object, right: object, expected: list[PatchOperation]) -> None:
    """
    `json_patch` compares mappings key by key and sequences of the same length
    item by item, and its patch turns `left` into `right`.
    """
    patch = json_patch(left, right)
    assert patch == expected
    assert _json_equal(_applied(left, patch), right)


def test_json_patch_refuses_a_key_that_is_not_a_string() -> None:
    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        json_patch({1: "a"}, {})


ARRAY = ArraySpec.create_default(shape=(4,), data_type="float32", fill_value="NaN")
"""A float32 array whose fill value is written `"NaN"`."""

ARRAY_HEX_NAN = ARRAY.replace(fill_value="0x7fc00000")
"""`ARRAY` with its fill value written as the bits of the same NaN."""

class _Noted(ArraySpec):
    """An array spec with a field that a plain `GroupSpec` does not dump for its members."""

    note: str = "original"


# Type checkers refuse this members type, which is outside the bound, but
# pydantic accepts it, and dumps each member as a group without members.
_GroupOfBaseGroups = GroupSpec[dict[str, object], Mapping[str, BaseGroupSpec]]  # pyrefly: ignore[bad-specialization]

ARRAY_V2 = ArraySpecV2(shape=(4,), chunks=(2,), dtype="<f8", compressor={"id": "zlib"})

CODECS_WITH_GZIP: tuple[CodecLike, ...] = (
    *ARRAY.codecs,
    {"name": "gzip", "configuration": {"level": 1}},
)

FILL_VALUE_SPELLING: PatchOperation = {"op": "replace", "path": "/fill_value", "value": "0x7fc00000"}

CASES: list[tuple[str, object, object, list[PatchOperation], list[PatchOperation]]] = [
    # (id, left, right, the patch, the operations that change the meaning)
    ("identical", ARRAY, ARRAY, [], []),
    ("fill-value-spelling", ARRAY, ARRAY_HEX_NAN, [FILL_VALUE_SPELLING], []),
    (
        "chunk-key-encoding-spelling",
        ARRAY,
        ARRAY.replace(chunk_key_encoding={"name": "default"}),
        [{"op": "remove", "path": "/chunk_key_encoding/configuration"}],
        [],
    ),
    (
        "attribute-added",
        ARRAY,
        ARRAY.replace(attributes={"units": "m"}),
        [{"op": "add", "path": "/attributes/units", "value": "m"}],
        [{"op": "add", "path": "/attributes/units", "value": "m"}],
    ),
    (
        "attribute-key-with-slash",
        ARRAY.replace(attributes={"a/b": 1}),
        ARRAY.replace(attributes={"a/b": 2}),
        [{"op": "replace", "path": "/attributes/a~1b", "value": 2}],
        [{"op": "replace", "path": "/attributes/a~1b", "value": 2}],
    ),
    (
        "nan-attributes-are-equal",
        ARRAY.replace(attributes={"x": float("nan")}),
        ARRAY.replace(attributes={"x": float("nan")}),
        [],
        [],
    ),
    (
        "bool-attribute-is-not-a-number",
        ARRAY.replace(attributes={"x": True}),
        ARRAY.replace(attributes={"x": 1}),
        [{"op": "replace", "path": "/attributes/x", "value": 1}],
        [{"op": "replace", "path": "/attributes/x", "value": 1}],
    ),
    (
        "data-type-and-fill-value",
        ARRAY,
        ARRAY.replace(data_type="float64", fill_value="0x7ff8000000000000"),
        [
            {"op": "replace", "path": "/data_type", "value": "float64"},
            {"op": "replace", "path": "/fill_value", "value": "0x7ff8000000000000"},
        ],
        [
            {"op": "replace", "path": "/data_type", "value": "float64"},
            {"op": "replace", "path": "/fill_value", "value": "0x7ff8000000000000"},
        ],
    ),
    (
        "shape-item",
        ARRAY,
        ARRAY.replace(shape=(8,)),
        [{"op": "replace", "path": "/shape/0", "value": 8}],
        [{"op": "replace", "path": "/shape/0", "value": 8}],
    ),
    (
        "codec-added",
        ARRAY,
        ARRAY.replace(codecs=CODECS_WITH_GZIP),
        [{"op": "replace", "path": "/codecs", "value": CODECS_WITH_GZIP}],
        [{"op": "replace", "path": "/codecs", "value": CODECS_WITH_GZIP}],
    ),
    (
        "extra-field-added",
        ARRAY,
        ARRAY.replace(extra_fields={"foo": {"must_understand": False}}),
        [{"op": "add", "path": "/foo", "value": {"must_understand": False}}],
        [{"op": "add", "path": "/foo", "value": {"must_understand": False}}],
    ),
    (
        "nested-member-spelling",
        GroupSpec(members={"x": ARRAY, "g": GroupSpec(members={"y": ARRAY})}),
        GroupSpec(members={"x": ARRAY, "g": GroupSpec(members={"y": ARRAY_HEX_NAN})}),
        [{"op": "replace", "path": "/members/g/members/y/fill_value", "value": "0x7fc00000"}],
        [],
    ),
    (
        "members-on-one-side",
        GroupSpec(members={"x": ARRAY, "y": ARRAY}),
        GroupSpec(members={"x": ARRAY, "z": GroupSpec(members={})}),
        [
            {"op": "remove", "path": "/members/y"},
            {"op": "add", "path": "/members/z", "value": GroupSpec(members={}).model_dump()},
        ],
        [
            {"op": "remove", "path": "/members/y"},
            {"op": "add", "path": "/members/z", "value": GroupSpec(members={}).model_dump()},
        ],
    ),
    (
        "member-changes-type",
        GroupSpec(members={"x": ARRAY}),
        GroupSpec(members={"x": GroupSpec(members={})}),
        [{"op": "replace", "path": "/members/x", "value": GroupSpec(members={}).model_dump()}],
        [{"op": "replace", "path": "/members/x", "value": GroupSpec(members={}).model_dump()}],
    ),
    (
        "group-and-base-group",
        GroupSpec(members={"x": ARRAY}),
        BaseGroupSpec(),
        [{"op": "remove", "path": "/members"}],
        [{"op": "remove", "path": "/members"}],
    ),
    (
        "base-group-and-group",
        BaseGroupSpec(attributes={"a": 1}),
        GroupSpec(members={"x": ARRAY}),
        [
            {"op": "remove", "path": "/attributes/a"},
            {"op": "add", "path": "/members", "value": {"x": ARRAY.model_dump()}},
        ],
        [
            {"op": "remove", "path": "/attributes/a"},
            {"op": "add", "path": "/members", "value": {"x": ARRAY.model_dump()}},
        ],
    ),
    (
        "member-subclass-field-not-in-the-group-document",
        GroupSpec(members={"x": _Noted(**ARRAY.model_dump())}),
        GroupSpec(members={"x": _Noted(**ARRAY.model_dump(), note="changed")}),
        [],
        [],
    ),
    (
        "base-group-extra-field-named-members",
        GroupSpec(members={"x": ARRAY}),
        BaseGroupSpec.model_validate({"members": {"must_understand": False}}),
        [
            {"op": "remove", "path": "/members/x"},
            {"op": "add", "path": "/members/must_understand", "value": False},
        ],
        [
            {"op": "remove", "path": "/members/x"},
            {"op": "add", "path": "/members/must_understand", "value": False},
        ],
    ),
    (
        "member-declared-without-members",
        _GroupOfBaseGroups(members={"g": GroupSpec(members={"x": ARRAY})}),
        _GroupOfBaseGroups(members={"g": GroupSpec(members={}, attributes={"a": 1})}),
        [{"op": "add", "path": "/members/g/attributes/a", "value": 1}],
        [{"op": "add", "path": "/members/g/attributes/a", "value": 1}],
    ),
    (
        "v2-compressor-spelling",
        ARRAY_V2,
        ARRAY_V2.replace(compressor={"id": "zlib", "level": 1}),
        [{"op": "add", "path": "/compressor/level", "value": 1}],
        [],
    ),
    (
        "v2-group",
        GroupSpecV2(members={"x": ARRAY_V2, "y": ARRAY_V2}, attributes={"a": 1}),
        GroupSpecV2(members={"x": ARRAY_V2.replace(chunks=(4,))}, attributes={"a": 2}),
        [
            {"op": "replace", "path": "/attributes/a", "value": 2},
            {"op": "replace", "path": "/members/x/chunks/0", "value": 4},
            {"op": "remove", "path": "/members/y"},
        ],
        [
            {"op": "replace", "path": "/attributes/a", "value": 2},
            {"op": "replace", "path": "/members/x/chunks/0", "value": 4},
            {"op": "remove", "path": "/members/y"},
        ],
    ),
]


@pytest.mark.parametrize(
    ("left", "right", "patch", "changes_of_meaning"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_diff(
    left: object,
    right: object,
    patch: list[PatchOperation],
    changes_of_meaning: list[PatchOperation],
) -> None:
    """
    `diff` returns the JSON Patch from one spec's document to the other's, and
    with `semantic=True`, the operations of that patch that change the meaning.
    """
    spec = cast("Any", left)
    assert spec.diff(right) == patch
    assert spec.diff(right, semantic=True) == changes_of_meaning
    assert _json_equal(_applied(spec.model_dump(), patch), cast("Any", right).model_dump())


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (ARRAY, GroupSpec(members={})),
        (GroupSpec(members={}), ARRAY),
        (BaseGroupSpec(), ARRAY),
        (ARRAY_V2, GroupSpecV2(members={})),
    ],
    ids=["v3-array-group", "v3-group-array", "v3-base-group-array", "v2-array-group"],
)
def test_diff_refuses_an_array_and_a_group(left: object, right: object) -> None:
    with pytest.raises(ValueError, match="A diff compares two arrays or two groups"):
        cast("Any", left).diff(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [(ARRAY, ARRAY_V2), (ARRAY_V2, ARRAY), (ARRAY, ARRAY.model_dump())],
    ids=["v3-v2", "v2-v3", "dict"],
)
def test_diff_refuses_a_value_that_is_not_a_node_of_the_same_version(
    left: object, right: object
) -> None:
    with pytest.raises(TypeError, match="ArraySpec, GroupSpec or BaseGroupSpec, got"):
        cast("Any", left).diff(right)


class _Letters(BaseModel):
    a: str = "old"
    b: str = "old"
    c: str = "old"
    d: str = "old"


def _letters_view(node: _Letters) -> NodeView[_Letters]:
    def read(document: Mapping[str, object]) -> object:
        # the meaning: whether most of a, b and c are "new", and the value of d
        return (sum(document[key] == "new" for key in "abc") >= 2, document["d"])  # noqa: PLR2004

    return NodeView("array", read, None)


def _replaced(*keys: str) -> list[PatchOperation]:
    return [{"op": "replace", "path": f"/{key}", "value": "new"} for key in keys]


@pytest.mark.parametrize(
    ("right", "expected"),
    [
        # one of a, b and c alone does not change the meaning
        (_Letters(a="new"), []),
        # d changes the meaning on its own
        (_Letters(a="new", d="new"), _replaced("d")),
        # a, b and c change it together, though no single one does
        (_Letters(a="new", b="new", c="new"), _replaced("a", "b", "c")),
        # so they are kept beside d, which is kept on its own
        (_Letters(a="new", b="new", c="new", d="new"), _replaced("a", "b", "c", "d")),
    ],
    ids=["one", "independent", "together", "together-with-independent"],
)
def test_diff_keeps_operations_that_change_meaning_together(
    right: _Letters, expected: list[PatchOperation]
) -> None:
    """
    An operation is left out only if it keeps the meaning on its own, and all
    the operations left out keep it together. So an empty semantic patch
    always means the same meaning.
    """
    assert node_patch(_Letters(), right, _letters_view, semantic=True) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [(ARRAY, ARRAY), (GroupSpec(members={}), BaseGroupSpec()), (ARRAY_V2, ARRAY_V2)],
    ids=["v3-array", "v3-group", "v2-array"],
)
def test_like_is_deprecated(left: object, right: object) -> None:
    with pytest.warns(DeprecationWarning, match="`like` is deprecated"):
        cast("Any", left).like(right)
