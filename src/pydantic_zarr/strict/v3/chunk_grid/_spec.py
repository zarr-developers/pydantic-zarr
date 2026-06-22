from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _GridValidate(Protocol):
    def __call__(self, meta: Any) -> None: ...


class _GridNdimOf(Protocol):
    def __call__(self, meta: Any) -> int | None: ...


@dataclass(frozen=True, slots=True)
class GridSpec:
    name: str
    metadata_type: type
    validate: _GridValidate
    ndim_of: _GridNdimOf
