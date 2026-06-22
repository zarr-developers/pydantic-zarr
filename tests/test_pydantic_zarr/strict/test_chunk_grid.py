import pytest

from pydantic_zarr.strict.v3.chunk_grid.rectilinear import ndim_of as rect_ndim
from pydantic_zarr.strict.v3.chunk_grid.rectilinear import rectilinear
from pydantic_zarr.strict.v3.chunk_grid.regular import ndim_of as reg_ndim
from pydantic_zarr.strict.v3.chunk_grid.regular import regular


def test_regular_builds_and_ndim() -> None:
    m = regular((4, 4))
    assert m == {"name": "regular", "configuration": {"chunk_shape": (4, 4)}}
    assert reg_ndim(m) == 2


def test_regular_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        regular((0, 4))


def test_rectilinear_builds_and_ndim() -> None:
    m = rectilinear(((1, 3), 4))
    assert rect_ndim(m) == 2
