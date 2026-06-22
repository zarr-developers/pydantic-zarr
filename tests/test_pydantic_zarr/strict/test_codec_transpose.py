import pytest

from pydantic_zarr.strict.v3.codec.transpose import dtype_out, kind, ndim_of, transpose


def test_transpose_builds_and_ndim() -> None:
    m = transpose((1, 0))
    assert m == {"name": "transpose", "configuration": {"order": (1, 0)}}
    assert ndim_of(m) == 2
    assert kind == "array_array"
    assert dtype_out(m, "int32") == "int32"


@pytest.mark.parametrize("bad", [(0, 0), (5, 1), (1, 2, 2)])
def test_transpose_rejects_non_permutation(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        transpose(bad)
