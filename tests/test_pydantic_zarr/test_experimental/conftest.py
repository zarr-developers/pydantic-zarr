from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


@dataclass
class FakeArray:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass
class FakeH5PyArray(FakeArray):
    attrs: Mapping[str, Any]
    chunks: tuple[int, ...]


@dataclass
class FakeDaskArray(FakeArray):
    chunksize: tuple[int, ...]
    chunks: tuple[tuple[int, ...], ...]


@dataclass
class FakeXarray(FakeArray):
    chunksizes: dict[str, tuple[int, ...]]
    chunks: tuple[tuple[int, ...], ...] | None
