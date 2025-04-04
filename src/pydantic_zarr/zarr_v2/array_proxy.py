from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")


class RegularChunks(TypedDict):
    read_shape: tuple[int, ...]
    write_shape: tuple[int, ...] | None


class RectilinearChunks(TypedDict):
    read_shape: tuple[tuple[int, ...], ...]
    write_shape: tuple[tuple[int, ...], ...] | None


class ArrayV2Proxy(abc.ABC, Generic[T]):
    _array: T

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    @abc.abstractmethod
    def dtype(self) -> str | dict[str, Any] | list[Any]: ...

    @property
    @abc.abstractmethod
    def chunks(self) -> tuple[int, ...]: ...

    @property
    @abc.abstractmethod
    def attributes(self) -> Mapping[str, Any]: ...

    @property
    @abc.abstractmethod
    def filters(self) -> list[dict[str, Any]] | None: ...

    @property
    @abc.abstractmethod
    def compressor(self) -> dict[str, Any] | None: ...

    @property
    @abc.abstractmethod
    def fill_value(self) -> Any: ...

    @property
    @abc.abstractmethod
    def order(self) -> Literal["C", "F"]: ...

    @property
    @abc.abstractmethod
    def dimension_separator(self) -> Literal[".", "/"]: ...

    @property
    @abc.abstractmethod
    def codecs(self) -> list[dict[str, Any]]: ...

    @property
    @abc.abstractmethod
    def dimension_names(self) -> list[str] | None: ...
