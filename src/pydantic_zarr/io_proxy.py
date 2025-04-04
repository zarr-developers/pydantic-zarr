from __future__ import annotations

from typing import Any, Protocol

from pydantic_zarr.base import GroupLike
from pydantic_zarr.zarr_v2.array_proxy import ArrayV2Proxy


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
    ) -> ArrayV2Proxy[Any]: ...

    def create_group(self, *, path: str, attributes: Any, members: Any) -> GroupLike: ...
