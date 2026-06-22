# src/pydantic_zarr/strict/v3/chunk_key_encoding/_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _KeyEncodingValidate(Protocol):
    def __call__(self, meta: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class KeyEncodingSpec:
    name: str
    metadata_type: type
    validate: _KeyEncodingValidate
