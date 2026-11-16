"""
The `auto_*` readers behind the v2 and v3 `ArraySpec.from_array` probe a
duck-typed array for attributes, and check what they read before returning it
as the type they promise: a value of the wrong type raises `TypeError` there,
rather than flowing on unchecked.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel

from pydantic_zarr import v2, v3
from pydantic_zarr.core import as_shape


class Attrs(BaseModel):
    a: int = 1


def test_as_shape() -> None:
    """A shape is read as a tuple of Python ints, from Python or numpy integers."""
    shape = as_shape([np.int64(2), 3])
    assert shape == (2, 3)
    assert all(type(n) is int for n in shape)


@pytest.mark.parametrize("value", [3, "ab"])
def test_as_shape_rejects_non_iterable(value: object) -> None:
    with pytest.raises(TypeError, match="Expected an iterable of integers"):
        as_shape(value)


def test_as_shape_rejects_non_integer() -> None:
    with pytest.raises(TypeError):
        as_shape([1.5])


def test_v2_auto_readers() -> None:
    codec = {"id": "gzip", "level": 1}
    array = SimpleNamespace(chunks=[np.int64(2)], attrs=Attrs(), compressor=codec, filters=(codec,))
    assert v2.auto_chunks(array) == (2,)
    assert v2.auto_attributes(array) == Attrs()
    assert v2.auto_compresser(array) == codec
    assert v2.auto_filters(array) == [codec]
    # attributes that are neither a mapping nor a model are skipped
    assert v2.auto_attributes(SimpleNamespace(attrs="no", attributes={"a": 1})) == {"a": 1}
    assert v2.auto_compresser(SimpleNamespace(compressor=None)) is None
    assert v2.auto_filters(SimpleNamespace(filters=None)) is None


def test_v2_auto_compresser_rejects_non_codec() -> None:
    with pytest.raises(TypeError, match="Expected a numcodecs codec or a mapping"):
        v2.auto_compresser(SimpleNamespace(compressor="gzip"))


def test_v2_auto_filters_rejects_non_iterable() -> None:
    with pytest.raises(TypeError, match="Expected an iterable of codecs"):
        v2.auto_filters(SimpleNamespace(filters={"id": "gzip"}))


def test_v2_auto_filters_rejects_non_codec_member() -> None:
    with pytest.raises(TypeError, match="Expected a numcodecs codec or a mapping"):
        v2.auto_filters(SimpleNamespace(filters=["gzip"]))


def test_v3_auto_readers() -> None:
    named = {"name": "x", "configuration": {"k": 1}}
    encoding = {"name": "v2", "configuration": {"separator": "."}}
    metadata = SimpleNamespace(
        dimension_names=["x", None],
        chunk_key_encoding=None,
        to_dict=lambda: {"chunk_key_encoding": encoding},
    )
    array = SimpleNamespace(
        chunk_shape=[np.int64(4)],
        codecs=["bytes", named],
        storage_transformers=[named],
        metadata=metadata,
    )
    assert v3.auto_chunk_grid(array) == {
        "name": "regular",
        "configuration": {"chunk_shape": (4,)},
    }
    assert v3.auto_codecs(array) == ("bytes", named)
    assert v3.auto_storage_transformers(array) == (named,)
    assert v3.auto_dimension_names(array) == ("x", None)
    assert v3.auto_chunk_key_encoding(array) == encoding
    assert v3.auto_dimension_names(SimpleNamespace(metadata=SimpleNamespace())) is None


def test_v3_auto_codecs_rejects_non_iterable() -> None:
    with pytest.raises(TypeError, match="Expected an iterable"):
        v3.auto_codecs(SimpleNamespace(codecs={"name": "bytes"}))


def test_v3_auto_codecs_rejects_malformed_codec() -> None:
    with pytest.raises(TypeError, match="Expected a named configuration"):
        v3.auto_codecs(SimpleNamespace(codecs=[{"configuration": {}}]))


def test_v3_auto_storage_transformers_rejects_malformed_transformer() -> None:
    with pytest.raises(TypeError, match="Expected a named configuration"):
        v3.auto_storage_transformers(SimpleNamespace(storage_transformers=["x"]))


def test_v3_auto_dimension_names_rejects_non_string_name() -> None:
    metadata = SimpleNamespace(dimension_names=["x", 1])
    with pytest.raises(TypeError, match="Expected a dimension name"):
        v3.auto_dimension_names(SimpleNamespace(metadata=metadata))


def test_v3_auto_chunk_key_encoding_rejects_unreadable_metadata() -> None:
    metadata = SimpleNamespace(chunk_key_encoding=None, to_dict=lambda: None)
    with pytest.raises(TypeError, match="Cannot read the chunk key encoding"):
        v3.auto_chunk_key_encoding(SimpleNamespace(metadata=metadata))
