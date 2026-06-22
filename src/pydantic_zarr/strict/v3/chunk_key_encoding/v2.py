# src/pydantic_zarr/strict/v3/chunk_key_encoding/v2.py
from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.v3.chunk_key_encoding.v2 import V2ChunkKeyEncodingObject

from pydantic_zarr.strict.v3.chunk_key_encoding._spec import KeyEncodingSpec

if TYPE_CHECKING:
    from zarr_metadata.v3.chunk_key_encoding.v2 import (
        V2ChunkKeyEncodingMetadata,
        V2ChunkKeyEncodingSeparator,
    )


def v2(separator: V2ChunkKeyEncodingSeparator = ".") -> V2ChunkKeyEncodingMetadata:
    meta: V2ChunkKeyEncodingMetadata = {"name": "v2", "configuration": {"separator": separator}}
    validate_v2(meta)
    return meta


def validate_v2(meta: V2ChunkKeyEncodingMetadata) -> None:
    """No intrinsic check: ``separator`` is a Literal enforced by the type system."""


SPEC = KeyEncodingSpec(
    name="v2",
    metadata_type=V2ChunkKeyEncodingObject,
    validate=validate_v2,
)
