import pytest

from pydantic_zarr.strict.v3.codec.blosc import blosc, ndim_of


def test_blosc_builds() -> None:
    m = blosc("zstd", 5, "shuffle", 0)
    assert m["configuration"]["cname"] == "zstd"
    assert ndim_of(m) is None


def test_blosc_clevel_range() -> None:
    with pytest.raises(ValueError):
        blosc("zstd", 99, "shuffle", 0)
