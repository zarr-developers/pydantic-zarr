from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from typing import Any


class _CodecValidate(Protocol):
    def __call__(self, meta: Any) -> None: ...


class _CodecNdimOf(Protocol):
    def __call__(self, meta: Any) -> int | None: ...


class _CodecDtypeOut(Protocol):
    def __call__(self, meta: Any, input_dtype: str) -> str: ...


@dataclass(frozen=True, slots=True)
class CodecSpec:
    name: str
    kind: Literal["array_array", "array_bytes", "bytes_bytes"]
    metadata_type: type
    has_dtype_dependent_config: bool
    validate: _CodecValidate
    ndim_of: _CodecNdimOf
    dtype_out: _CodecDtypeOut
