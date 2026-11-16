from __future__ import annotations

import importlib

import pytest

import pydantic_zarr.experimental


def test_deprecated_import() -> None:
    """Importing `pydantic_zarr.experimental` warns that it goes in 0.13."""
    with pytest.warns(
        DeprecationWarning,
        match=r"`pydantic_zarr\.experimental` is deprecated and will be removed in pydantic-zarr 0\.13",
    ):
        importlib.reload(pydantic_zarr.experimental)
