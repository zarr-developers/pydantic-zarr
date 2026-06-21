import pytest

from pydantic_zarr.strict.v3.codec.sharding_indexed import (
    kind,
    ndim_of,
    sharding_indexed,
)


def test_sharding_builds_defaults() -> None:
    m = sharding_indexed((4, 4))
    cfg = m["configuration"]
    assert cfg["chunk_shape"] == (4, 4)
    assert cfg["index_codecs"][-1]["name"] == "crc32c"
    assert ndim_of(m) == 2
    assert kind == "array_bytes"


def test_sharding_rejects_nonpositive_chunk_shape() -> None:
    with pytest.raises(ValueError):
        sharding_indexed((0, 4))
