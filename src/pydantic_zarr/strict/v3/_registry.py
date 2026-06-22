from __future__ import annotations

from typing import Any


def config_required(metadata_type: type) -> bool:
    """A bare-string short form is permitted iff the configuration is optional.

    Reads the metadata *object* TypedDict's optional keys.  When ``configuration``
    is NOT optional (i.e. required), the bare-string form is not permitted.
    """
    return "configuration" not in metadata_type.__optional_keys__  # type: ignore[attr-defined]


def element_name(x: Any) -> str | None:
    """Extract an element's name from its object form (dict) or bare-string form."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        name = x.get("name")
        return name if isinstance(name, str) else None
    return None
