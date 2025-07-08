from __future__ import annotations

import pytest

from pydantic_zarr.core import ensure_member_name, tuplify_json


@pytest.mark.parametrize("data", ["/", "///", "a/b/", "a/b/vc"])
def test_parse_str_no_path(data: str) -> None:
    with pytest.raises(ValueError, match='Strings containing "/" are invalid.'):
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
