from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_zarr.base import ArrayLike, ArrayV2Config

if TYPE_CHECKING:
    from typing import Literal

from ._tensorstore._import import TENSORSTORE_INSTALLED
from .zarr_python._import import ZARR_PYTHON_INSTALLED


def zarrify_array_v2(array: object) -> ArrayV2Config:
    if ZARR_PYTHON_INSTALLED:
        import zarr

        if isinstance(array, zarr.Array):
            from pydantic_zarr.engine.zarr_python import zarrify as pyz_zarrify

            return pyz_zarrify(array, zarr_format=2)
        else:
            pass

    if TENSORSTORE_INSTALLED:
        import tensorstore

        if isinstance(array, tensorstore.TensorStore):
            from pydantic_zarr.engine._tensorstore import zarrify as ts_zarrify

            return ts_zarrify(array, zarr_format=2)
        else:
            pass

    if isinstance(array, ArrayLike):
        from pydantic_zarr.engine.numpy_like import zarrify as np_zarrify

        return np_zarrify(array, zarr_format=2)
    else:
        raise ValueError(f"Unsupported array type: {type(array)}")
