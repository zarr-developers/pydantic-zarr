"""
Private helpers for validating metadata documents with ``zarr-metadata``.

The models in `pydantic_zarr.v2` and `pydantic_zarr.v3` describe metadata
documents field by field; ``zarr-metadata`` validates whole documents against
the Zarr specs, reading each extension point (data type, chunk grid, chunk key
encoding, codecs, storage transformers) through the definition that claims its
name in a scope.
These helpers connect the two: they resolve the scope a pydantic validation
context carries and turn a ``MetadataValidationError`` into the ``ValueError``
a pydantic ``model_validator`` reports.

The scope travels through pydantic's validation context under the same
conventions as ``zarr_metadata.pydantic``: either the context *is* a
``ZarrV3Context``, or it is a mapping holding one under ``CONTEXT_KEY``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zarr_metadata.v3.definition import ZarrV3Context

if TYPE_CHECKING:
    from zarr_metadata import MetadataValidationError

CONTEXT_KEY = "zarr_metadata_context"
"""
The key of a mapping validation context under which the scope to read Zarr V3
metadata in sits. The same key `zarr_metadata.pydantic` uses, so one validation
context serves models holding field types from either library.
"""


def scope_from_context(context: object) -> ZarrV3Context | None:
    """
    Resolve the Zarr V3 scope a pydantic validation context carries.

    The context may be a ``ZarrV3Context`` itself, or a mapping holding one
    under ``CONTEXT_KEY``. Returns ``None`` when the context carries no scope,
    which readers take as their default scope (``CORE_AND_EXTENSIONS``).

    Raises
    ------
    TypeError
        If the context holds a value under ``CONTEXT_KEY`` that is not a
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

    Each problem appears on its own line as ``<loc>: <message>``, so a
    document with several problems reports all of them in one validation
    error instead of one per construction attempt.
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
