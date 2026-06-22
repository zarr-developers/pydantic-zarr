# src/pydantic_zarr/strict/v3/chunk_key_encoding/default.py
from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncodingObject

from pydantic_zarr.strict.v3.chunk_key_encoding._spec import KeyEncodingSpec

if TYPE_CHECKING:
    from zarr_metadata.v3.chunk_key_encoding.default import (
        DefaultChunkKeyEncodingMetadata,
        DefaultChunkKeyEncodingSeparator,
    )


def default(separator: DefaultChunkKeyEncodingSeparator = "/") -> DefaultChunkKeyEncodingMetadata:
    meta: DefaultChunkKeyEncodingMetadata = {
        "name": "default",
        "configuration": {"separator": separator},
    }
    validate_default(meta)
    return meta


def validate_default(meta: DefaultChunkKeyEncodingMetadata) -> None:
    """No intrinsic check: ``separator`` is a Literal enforced by the type system."""


SPEC = KeyEncodingSpec(
    name="default",
    metadata_type=DefaultChunkKeyEncodingObject,
    validate=validate_default,
)
