"""
Private helpers for validating metadata documents with ``zarr-metadata``.

The models in `pydantic_zarr.v2` and `pydantic_zarr.v3` describe metadata
documents field by field. ``zarr-metadata`` validates whole documents against
the Zarr specs. It validates each extension point (data type, chunk grid,
chunk key encoding, codecs, storage transformers) with the definition in a
scope that claims (recognizes) the extension point's name.

These helpers connect the two. They resolve the scope carried by a pydantic
validation context, and they turn a ``MetadataValidationError`` into the
``ValueError`` that a pydantic ``model_validator`` reports.

The scope is passed through pydantic's validation context, with the same
conventions as ``zarr_metadata.pydantic``: either the context *is* a
``ZarrV3Context``, or it is a mapping that contains one under ``CONTEXT_KEY``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zarr_metadata.pydantic import CONTEXT_KEY
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    CORE_CONTEXT,
    ZarrV3Context,
)

if TYPE_CHECKING:
    from zarr_metadata import MetadataValidationError

__all__ = [
    "CONTEXT_KEY",
    "CORE_AND_EXTENSIONS_CONTEXT",
    "CORE_CONTEXT",
    "format_lines",
    "format_problems",
    "scope_from_context",
]

# In a mapping validation context, `CONTEXT_KEY` is the key of the scope used
# to validate Zarr V3 metadata. It is the same key that `zarr_metadata.pydantic`
# uses, so one validation context works for models with field types from
# either library.


def scope_from_context(context: object) -> ZarrV3Context | None:
    """
    Return the Zarr V3 scope carried by a pydantic validation context.

    The context may be a ``ZarrV3Context`` itself, or a mapping that contains
    one under ``CONTEXT_KEY``. Returns ``None`` if the context carries no
    scope. Callers then use their default scope
    (``CORE_AND_EXTENSIONS_CONTEXT``).

    Raises
    ------
    TypeError
        If the context contains a value under ``CONTEXT_KEY`` that is not a
        ``ZarrV3Context``.
    """
    if isinstance(context, ZarrV3Context):
        return context
    if isinstance(context, Mapping) and CONTEXT_KEY in context:
        scope = context[CONTEXT_KEY]
        if not isinstance(scope, ZarrV3Context):
            msg = f"{CONTEXT_KEY}: the scope to read in must be a ZarrV3Context, got {scope!r}"
            raise TypeError(msg)
        return scope
    return None


def format_problems(
    error: MetadataValidationError, node_type: str = "array", *, zarr_format: int = 3
) -> str:
    """
    Format every problem in a ``MetadataValidationError`` as one message.

    Each problem appears on its own line as ``<loc>: <message>``. A document
    with several problems reports all of them in one validation error,
    instead of one error per construction attempt.
    """
    lines = []
    for problem in error.problems:
        loc = ".".join(str(part) for part in problem.loc)
        prefix = f"{loc}: " if loc else ""
        lines.append(f"{prefix}{problem.message}")
    return format_lines(lines, node_type, zarr_format=zarr_format)


def format_lines(lines: list[str], node_type: str, *, zarr_format: int = 3) -> str:
    """Format ``<loc>: <message>`` lines as one invalid-metadata message for a node type."""
    return f"Invalid Zarr V{zarr_format} {node_type} metadata:\n" + "\n".join(lines)
