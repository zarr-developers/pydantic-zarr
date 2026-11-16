"""
Experimental module for pydantic-zarr.

!!! warning "Deprecated"
    `pydantic_zarr.experimental` is deprecated and will be removed in
    pydantic-zarr 0.13. Importing it, or any of its submodules, emits a
    `DeprecationWarning`. Use `pydantic_zarr.v2` and `pydantic_zarr.v3`
    instead: they define `BaseGroupSpec`, and a subclass of their `ArraySpec`
    or `GroupSpec` can annotate `attributes` and `members` directly, as the
    experimental classes do.

This module contains refactored versions of the core modules with
breaking API changes.

The main changes in the experimental module:
- Removed generic type parameters from ArraySpec and GroupSpec
- Simplified type system using concrete union types
- Added BaseGroupSpec for group-only operations
"""

import warnings

EXPERIMENTAL_REMOVAL_VERSION = "0.13"
"""The pydantic-zarr release that removes `pydantic_zarr.experimental`."""

warnings.warn(
    "`pydantic_zarr.experimental` is deprecated and will be removed in pydantic-zarr "
    f"{EXPERIMENTAL_REMOVAL_VERSION}. Use `pydantic_zarr.v2` and `pydantic_zarr.v3` instead: "
    "they define `BaseGroupSpec`, and a subclass of their `ArraySpec` or `GroupSpec` can "
    "annotate `attributes` and `members` directly.",
    DeprecationWarning,
    # attribute the warning to the module that imported this package, past
    # the import machinery, so Python's default filters show it to scripts
    stacklevel=2,
    skip_file_prefixes=("<frozen importlib",),
)

from . import core, v2, v3  # noqa: E402

__all__ = ["core", "v2", "v3"]
