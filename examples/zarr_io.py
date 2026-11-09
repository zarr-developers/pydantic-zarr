# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-zarr", "zarr>=3.0.5"]
# ///
"""
Reading and writing pydantic-zarr specs with zarr-python.

pydantic-zarr models metadata documents and does no IO of its own: a spec's
``to_stored_metadata_json`` gives the documents a store holds, keyed as the
store keys them, and ``from_stored_metadata_json`` reads such a mapping back.
Moving the documents in and out of a store is zarr-python's job. These two
functions do that, for Zarr V2 and V3 alike, and the example scripts import
them; the usage guides show the same code.
"""

# --8<-- [start:write]
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from pydantic_zarr import v2, v3

if TYPE_CHECKING:
    from zarr.abc.store import Store


def write_hierarchy(
    spec: v2.AnyArraySpec | v2.AnyGroupSpec | v3.AnyArraySpec | v3.AnyGroupSpec,
    store: Store,
    path: str = "",
    *,
    overwrite: bool = False,
) -> zarr.Array | zarr.Group:
    """
    Write the metadata documents of `spec` to `store` at `path` with zarr-python,
    and return the node written there. No array data is written.

    Each node of the hierarchy becomes a zarr-python metadata object, built
    from the node's metadata document, and `zarr.create_hierarchy` writes them
    all, refusing to overwrite an existing node unless `overwrite` is set.
    zarr-python drops the extra fields of a document it cannot store: an
    array's before zarr-python 3.1, a group's before 3.4.
    """
    flat = v2.to_flat(spec) if spec.zarr_format == 2 else v3.to_flat(spec)
    nodes: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {}
    for node_path, node in flat.items():
        # the document as JSON holds, with tuples as arrays
        document = json.loads(json.dumps(node.to_metadata_json()))
        key = "/".join(part for part in (path.strip("/"), node_path.strip("/")) if part)
        if "shape" not in document:
            nodes[key] = GroupMetadata.from_dict(document)
        elif document["zarr_format"] == 2:
            nodes[key] = ArrayV2Metadata.from_dict(document)
        else:
            nodes[key] = ArrayV3Metadata.from_dict(document)
    written = dict(zarr.create_hierarchy(store=store, nodes=nodes, overwrite=overwrite))
    return written[path.strip("/")]


# --8<-- [end:write]


# --8<-- [start:read]
def read_documents(node: zarr.Array | zarr.Group) -> dict[str, object]:
    """
    The stored metadata documents of `node` and, for a group, of every node
    under it, keyed by store key relative to `node`: the mapping
    `from_stored_metadata_json` takes.

    `to_buffer_dict` gives a node's documents as zarr-python writes them
    (`zarr.json`, or `.zarray`/`.zgroup` and `.zattrs`), so this reads what the
    store holds, not zarr-python's in-memory view of it. One repair: zarr-python
    before 3.4 writes `"consolidated_metadata": null` into every group's
    `zarr.json` to say the group has none, and the readers refuse a null there,
    so a null is dropped as the absence it stands for.
    """
    prototype = default_buffer_prototype()
    nodes: list[tuple[str, zarr.Array | zarr.Group]] = [("", node)]
    if isinstance(node, zarr.Group):
        nodes.extend(node.members(max_depth=None))
    documents: dict[str, object] = {}
    for member_path, member in nodes:
        for name, buffer in member.metadata.to_buffer_dict(prototype).items():
            document = json.loads(buffer.to_bytes())
            if isinstance(document, dict) and document.get("consolidated_metadata", {}) is None:
                del document["consolidated_metadata"]
            documents[f"{member_path}/{name}" if member_path else name] = document
    return documents


# --8<-- [end:read]

if __name__ == "__main__":
    # --8<-- [start:round_trip]
    import numpy as np

    array = v3.ArraySpec.from_array(np.zeros((4, 4), dtype="uint8"), attributes={"k": 1})
    group = v3.GroupSpec(
        attributes={"a": 1}, members={"x": array, "sub": v3.GroupSpec(members={"y": array})}
    )
    store = zarr.storage.MemoryStore()
    write_hierarchy(group, store, "root")
    documents = read_documents(zarr.open_group(store, path="root", mode="r"))
    assert v3.GroupSpec.from_stored_metadata_json(documents) == group
    print(sorted(documents))
    # > ['sub/y/zarr.json', 'sub/zarr.json', 'x/zarr.json', 'zarr.json']
    # --8<-- [end:round_trip]

    array_v2 = v2.ArraySpec(shape=(4, 4), chunks=(2, 2), dtype="uint8", attributes={"k": 1})
    group_v2 = v2.GroupSpec(
        attributes={"a": 1}, members={"x": array_v2, "sub": v2.GroupSpec(members={"y": array_v2})}
    )
    store = zarr.storage.MemoryStore()
    write_hierarchy(group_v2, store, "root")
    documents = read_documents(zarr.open_group(store, path="root", mode="r", zarr_format=2))
    assert v2.GroupSpec.from_stored_metadata_json(documents) == group_v2
    written = write_hierarchy(array_v2, store, "alone")
    assert isinstance(written, zarr.Array)
    assert v2.ArraySpec.from_stored_metadata_json(read_documents(written)) == array_v2
    print("zarr_io.py: round trips hold for Zarr V2 and V3")
