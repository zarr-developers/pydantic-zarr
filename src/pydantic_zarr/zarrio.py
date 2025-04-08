from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_zarr.base import ArrayV2Config, GroupV2Config


class ZarrV2IO(abc.ABC):
    @abc.abstractmethod
    async def write_array(self, *, path: str, metadata: ArrayV2Config) -> None: ...

    @abc.abstractmethod
    async def write_group(self, *, path: str, metadata: GroupV2Config) -> None: ...
