"""
Experimental module for pydantic-zarr.

This module contains refactored versions of the core modules with
breaking API changes. Use with caution as the API is not yet stable.

The main changes in the experimental module:
- Removed generic type parameters from ArraySpec and GroupSpec
- Simplified type system using concrete union types
- Added BaseGroupSpec for group-only operations

To use the experimental module:

    from pydantic_zarr.experimental import v2, v3

    # Use v2.ArraySpec, v2.GroupSpec, etc. instead of the main module versions
"""

from . import core, v2, v3

__all__ = ["core", "v2", "v3"]
