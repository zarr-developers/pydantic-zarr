"""
Write specs with zarr-python, and read back the documents it stored.

The tests check agreement with zarr-python through the workflow users are
pointed to (examples/zarr_io.py): zarr-python writes the documents
`to_stored_metadata_json` gives, and `from_stored_metadata_json` reads the
documents zarr-python stored. Each function imports zarr itself, so a test
module can import these helpers when zarr is not installed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zarr
    from zarr.abc.store import Store
    from zarr.core.group import GroupMetadata
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

    from pydantic_zarr import v2, v3


def write_hierarchy(
    spec: v2.AnyArraySpec | v2.AnyBaseGroupSpec | v3.AnyArraySpec | v3.AnyBaseGroupSpec,
    store: Store,
    path: str = "",
) -> zarr.Array | zarr.Group:
    """
    Write the documents `spec.to_stored_metadata_json()` gives to `store` at
    `path` with `zarr.create_hierarchy`, and return the node written there.
    Each node's documents become the zarr-python metadata object zarr-python
    builds from them; no array data is written.
    """
    import zarr
    from zarr.core.group import GroupMetadata
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

    # the documents as JSON holds them, with tuples as arrays
    stored: dict[str, dict[str, Any]] = json.loads(json.dumps(spec.to_stored_metadata_json()))
    documents_by_node: dict[str, dict[str, dict[str, Any]]] = {}
    for key, document in stored.items():
        node_path, _, name = key.rpartition("/")
        documents_by_node.setdefault(node_path, {})[name] = document
    root = path.strip("/")
    nodes: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {}
    for node_path, documents in documents_by_node.items():
        key = "/".join(part for part in (root, node_path) if part)
        if "zarr.json" in documents:
            document = documents["zarr.json"]
            nodes[key] = (
                GroupMetadata.from_dict(document)
                if document["node_type"] == "group"
                else ArrayV3Metadata.from_dict(document)
            )
        elif ".zarray" in documents:
            nodes[key] = ArrayV2Metadata.from_dict(
                {**documents[".zarray"], "attributes": documents.get(".zattrs", {})}
            )
        else:
            nodes[key] = GroupMetadata.from_dict(
                {**documents[".zgroup"], "attributes": documents.get(".zattrs", {})}
            )
    written = dict(zarr.create_hierarchy(store=store, nodes=nodes))
    return written[root]


def _repaired(document: object) -> object:
    """
    `document` as the readers take it: zarr-python before 3.4 writes
    `"consolidated_metadata": null` into a group's `zarr.json` to say the group
    has none, and the readers refuse a null there, so it is dropped.
    """
    if isinstance(document, dict) and document.get("consolidated_metadata", {}) is None:
        return {key: value for key, value in document.items() if key != "consolidated_metadata"}
    return document


def read_documents(node: zarr.Array | zarr.Group) -> dict[str, object]:
    """
    The metadata documents zarr-python stores for `node` and, for a group,
    every node under it, keyed by store key relative to `node`: the mapping
    `from_stored_metadata_json` takes.
    """
    import zarr
    from zarr.core.buffer import default_buffer_prototype

    prototype = default_buffer_prototype()
    nodes: list[tuple[str, zarr.Array | zarr.Group]] = [("", node)]
    if isinstance(node, zarr.Group):
        nodes.extend(node.members(max_depth=None))
    documents: dict[str, object] = {}
    for member_path, member in nodes:
        for name, buffer in member.metadata.to_buffer_dict(prototype).items():
            key = f"{member_path}/{name}" if member_path else name
            documents[key] = _repaired(json.loads(buffer.to_bytes()))
    return documents
