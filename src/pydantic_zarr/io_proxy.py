from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic_zarr.base import GroupLike
from pydantic_zarr.zarr_v2.v2 import GroupSpec, TAttr, TMember
from pydantic_zarr.zarr_v3.v3 import ArraySpec


class ZarrV2IO(Protocol):
    def create_array(
        self,
        *,
        path: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: Any,
        fill_value: Any,
        order: Any,
        compressor: Any,
        filters: Any,
        dimension_separator: Any,
        attributes: Any,
        overwrite: bool,
    ) -> Any: ...

    def create_group(
        self,
        *,
        path: str,
        attributes: Mapping[str, object],
        members: Mapping[str, ArraySpec[TAttr] | GroupSpec[TAttr, TMember]],
    ) -> GroupLike: ...
