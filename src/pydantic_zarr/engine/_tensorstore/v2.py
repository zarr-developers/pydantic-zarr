from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorstore as ts

from pydantic_zarr.base import ArrayMetadataV2Config
from pydantic_zarr.zarrio import ZarrV2IO

from .models import DataType, KVStore, ZarrDriver

if TYPE_CHECKING:
    from pydantic_zarr.base import ArrayV2Config, GroupV2Config


V2_ARRAY_KEY = b".zarray"
V2_GROUP_KEY = b".zgroup"
V2_ATTRS_KEY = b".zattrs"
SEPARATOR = b"/"


def parse_dtype_name(dtype_name: str) -> DataType:
    return np.dtype(dtype_name).name


async def read_attrs_v2(store: ts.KvStore) -> dict[str, object]:
    zattrs_result = await store.read(V2_ATTRS_KEY)
    if zattrs_result.state == "missing":
        return {}
    return json.loads(zattrs_result.value)  # type: ignore[no-any-return]


async def read_group_v2(store: ts.KvStore) -> GroupV2Config:
    zgroup_result = await store.read(V2_GROUP_KEY)
    if zgroup_result.state == "missing":
        raise FileNotFoundError(f"No {V2_GROUP_KEY.decode()} file found in the store.")
    zgroup_meta = json.loads(zgroup_result.value)
    attributes = await read_attrs_v2(store)
    return zgroup_meta | {"attributes": attributes}  # type: ignore[no-any-return]


async def read_array_v2(store: ts.KvStore) -> ArrayV2Config:
    zarray_result = await store.read(V2_ARRAY_KEY)
    if zarray_result.state == "missing":
        raise FileNotFoundError(f"No {V2_ARRAY_KEY.decode()} file found in the store.")
    zarray_metadata: ArrayMetadataV2Config = json.loads(zarray_result.value)
    attributes = await read_attrs_v2(store)
    return zarray_metadata | {"attributes": attributes}  # type: ignore[return-value]


async def read_node_v2(store: ts.KvStore) -> GroupV2Config | ArrayV2Config:
    try:
        return await read_array_v2(store)
    except FileNotFoundError as e:
        try:
            return await read_group_v2(store)
        except FileNotFoundError:
            raise FileNotFoundError("No .zarray or .zgroup file found in the store.") from e


def get_member_keys(prefixes: Sequence[bytes]) -> tuple[bytes, ...]:
    member_prefixes = ()
    meta_by_level = defaultdict(list)

    # classify by the number of slashes in the prefix to ensure that we traverse breadth-first
    for prefix in prefixes:
        if prefix.endswith((V2_GROUP_KEY, V2_ARRAY_KEY)):
            meta_by_level[prefix.count(SEPARATOR)].append(prefix)

    for level, node_prefixes in meta_by_level.items():
        if level == 0:
            if node_prefixes == [V2_GROUP_KEY]:
                member_prefixes += tuple(node_prefixes)
            else:
                raise ValueError(
                    f"Invalid hierarchy: {node_prefixes} does not contain valid group metadata."
                )
        else:
            for prefix in node_prefixes:
                # check if the parent prefix is in the dict of group metadata documents
                if level == 1:
                    parent_group_meta = V2_GROUP_KEY
                else:
                    parent_group_meta = (
                        SEPARATOR.join(prefix.split(SEPARATOR)[:-2]) + SEPARATOR + V2_GROUP_KEY
                    )
                if level - 1 in meta_by_level and parent_group_meta in meta_by_level[level - 1]:
                    member_prefixes += (prefix,)

    return member_prefixes


async def read_members_v2(store: ts.KvStore) -> dict[bytes, ArrayV2Config | GroupV2Config]:
    """
    Read the members of a group.
    """
    # TODO: handle proper hierarchy traversal. tensorstore list operation is not scoped to a particular
    # group, so we need to filter out the members that are not in the group.
    group = await read_group_v2(store)
    maybe_member_names = tuple(
        filter(lambda v: v not in (V2_GROUP_KEY,), get_member_keys(await store.list()))
    )
    read_futs: list[Coroutine[Any, Any, ArrayV2Config | GroupV2Config]] = []
    for member_name in maybe_member_names:
        if member_name.endswith(b".zarray"):
            read_futs.append(read_array_v2(store / member_name.removesuffix(b".zarray")))
        elif member_name.endswith(b".zgroup"):
            read_futs.append(read_group_v2(store / member_name.removesuffix(b".zgroup")))
        else:
            pass
    members = await asyncio.gather(*read_futs, return_exceptions=True)
    return {b"": group} | {
        name.rsplit(b"/", 1)[0]: node
        for name, node in zip(maybe_member_names, members, strict=True)
    }


async def write_group_v2(metadata: GroupV2Config, *, kvstore: KVStore) -> Any:
    futs = []
    attrs_meta = metadata.pop("attributes", None)
    zgroup_meta = metadata

    if attrs_meta is not None and len(attrs_meta) > 0:
        futs.append(kvstore.write(V2_ATTRS_KEY, json.dumps(attrs_meta)))
    futs.append(kvstore.write(V2_GROUP_KEY, json.dumps(zgroup_meta)))
    return await asyncio.gather(*futs, return_exceptions=True)


async def write_array_v2(
    metadata: ArrayV2Config,
    *,
    kvstore: KVStore | dict[str, Any],
    open: bool = True,
    create: bool = False,
    delete_existing: bool = False,
    assume_metadata: bool = False,
    assume_cached_metadata: bool = False,
) -> ts.TensorStore:
    spec = ZarrDriver(
        driver="zarr",
        kvstore=kvstore,
        metadata=metadata,
        open=open,
        create=create,
        delete_existing=delete_existing,
        assume_metadata=assume_metadata,
        assume_cached_metadata=assume_cached_metadata,
    )
    return await ts.open(spec.model_dump(exclude_none=True))


class TensorstoreZarrIO(ZarrV2IO):
    """
    Tensorstore Zarr IO backend for reading and writing Zarr arrays and groups.
    """

    def __init__(self, store: KVStore) -> None:
        self.store = store

    async def write_array(self, *, path: str, metadata: ArrayV2Config) -> ts.TensorStore:
        return await write_array_v2(metadata, kvstore=self.store / path)

    async def write_group(self, *, path: str, metadata: GroupV2Config):
        return await super().write_group(path=path, metadata=metadata)
