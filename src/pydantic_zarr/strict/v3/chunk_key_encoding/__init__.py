# src/pydantic_zarr/strict/v3/chunk_key_encoding/__init__.py
from __future__ import annotations

from zarr_metadata import DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata

from pydantic_zarr.strict.v3._registry import element_name
from pydantic_zarr.strict.v3.chunk_key_encoding._spec import KeyEncodingSpec

from . import default as _default
from . import v2 as _v2

# Re-export builders
default = _default.default
v2 = _v2.v2

_ChunkKeyEncoding = DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata

_MODULES = (_default, _v2)
KEY_ENCODINGS: dict[str, KeyEncodingSpec] = {m.SPEC.name: m.SPEC for m in _MODULES}


def key_encoding_spec_for(k: object) -> KeyEncodingSpec | None:
    name = element_name(k)
    return KEY_ENCODINGS.get(name) if name is not None else None


__all__ = [
    "KEY_ENCODINGS",
    "KeyEncodingSpec",
    "_ChunkKeyEncoding",
    "default",
    "key_encoding_spec_for",
    "v2",
]
