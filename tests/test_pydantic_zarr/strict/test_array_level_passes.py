"""Tests for the array-level cross-field and pipeline validation passes
(Task 12): dimensionality, sharding divisibility, codec-pipeline type-flow.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import AnyCoreArraySpec


def _doc(**over: object) -> dict[str, object]:
    base = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "shape": (8, 8),
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "attributes": {},
    }
    base.update(over)
    return base


def test_array_rejects_chunk_grid_ndim_mismatch() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(
            _doc(chunk_grid={"name": "regular", "configuration": {"chunk_shape": (8,)}})
        )


def test_array_rejects_dimension_names_mismatch() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(_doc(dimension_names=("x",)))


def test_array_valid_2d_passes() -> None:
    TypeAdapter(AnyCoreArraySpec).validate_python(_doc(dimension_names=("y", "x")))
