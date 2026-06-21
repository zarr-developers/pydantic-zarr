import pytest

from pydantic_zarr.strict.v3.codec.bytes import bytes_codec
from pydantic_zarr.strict.v3.codec.bytes import kind as bytes_kind
from pydantic_zarr.strict.v3.codec.bytes import ndim_of as bytes_ndim
from pydantic_zarr.strict.v3.codec.crc32c import crc32c
from pydantic_zarr.strict.v3.codec.gzip import gzip
from pydantic_zarr.strict.v3.codec.gzip import ndim_of as gzip_ndim
from pydantic_zarr.strict.v3.codec.zstd import zstd


def test_bytes_codec_builds() -> None:
    assert bytes_codec("little") == {"name": "bytes", "configuration": {"endian": "little"}}
    assert bytes_ndim(bytes_codec()) is None
    assert bytes_kind == "array_bytes"


def test_gzip_level_range() -> None:
    assert gzip(5)["configuration"]["level"] == 5
    with pytest.raises(ValueError):
        gzip(99)
    assert gzip_ndim(gzip(1)) is None


def test_zstd_builds_no_level_bound() -> None:
    assert zstd(3, checksum=True)["configuration"] == {"level": 3, "checksum": True}


def test_crc32c_builds() -> None:
    assert crc32c() == {"name": "crc32c"}
