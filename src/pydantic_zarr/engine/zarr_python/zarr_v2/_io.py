"""
A zarr-python based backend for pydantic-zarr.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import zarr
from zarr import create_group
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.storage._common import StorePath, contains_array, contains_group

from pydantic_zarr.zarr_v2.v2 import ArraySpec, BaseAttr, BaseItem, GroupSpec

if TYPE_CHECKING:
    from zarr.abc.store import Store


async def create_spec(node: zarr.Group | zarr.Array) -> GroupSpec[Any, Any] | ArraySpec[Any]:
    """
    Create a GroupSpec or ArraySpec from a stored `Zarr.Group` `Zarr.Array`.
    """

    if isinstance(node, zarr.Array):
        return create_arrayspec(node)
    elif isinstance(node, zarr.Group):
        return create_groupspec(node)
    else:
        raise TypeError(f"node must be a zarr.Array or zarr.Group, got {type(node)}")


async def write_spec(
    model: GroupSpec[Any, Any],
    store: Store,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: object,
) -> zarr.Group | zarr.Array:
    """
    Serialize a `GroupSpec` to a Zarr group at a specific path in a zarr `Store`.
    This operation will create metadata documents in the store.
    Parameters
    ----------
    store : zarr.BaseStore
        The storage backend that will manifest the group and its contents.
    path : str
        The location of the group inside the store.
    overwrite: bool, default = False
        Whether to overwrite existing objects in storage to create the Zarr group.
    **kwargs : Any
        Additional keyword arguments that will be passed to `zarr.create_array` for creating
        sub-arrays.
    Returns
    -------
    zarr.Group
        A zarr group that is structurally identical to the provided spec.

    """

    spec_dict = model.model_dump(exclude={"members": True})
    attrs = spec_dict.pop("attributes")
    if await contains_group(StorePath(store=store, path=path), zarr_format=2):
        extant_group = zarr.group(store, path=path)
        if not model.like(extant_group):
            if not overwrite:
                msg = (
                    f"A group already exists at path {path}. "
                    "That group is structurally dissimilar to the group you are trying to store."
                    "Call to_zarr with overwrite=True to overwrite that group."
                )
                raise ContainsGroupError(msg)
        else:
            if not overwrite:
                # if the extant group is structurally identical to self, and overwrite is false,
                # then just return the extant group
                return extant_group

    elif await contains_array(StorePath(store=store, path=path), zarr_format=2) and not overwrite:
        msg = (
            f"An array already exists at path {path}. "
            "Call to_zarr with overwrite=True to overwrite the array."
        )
        raise ContainsArrayError(msg)
    else:
        result = zarr.create_group(
            store=store, overwrite=overwrite, path=path, zarr_format=2, attributes=attrs
        )

    # consider raising an exception if a partial GroupSpec is provided
    if model.members is not None:
        for name, member in model.members.items():
            subpath = os.path.join(path, name)
            member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

    return result


def create_groupspec(group: zarr.Group, *, depth: int = -1) -> GroupSpec[BaseAttr, BaseItem]:
    """
    Create a GroupSpec from an instance of `zarr.Group`. Subgroups and arrays contained in the
    Zarr group will be converted to instances of `GroupSpec` and `ArraySpec`, respectively,
    and these spec instances will be stored in the `members` attribute of the parent
    `GroupSpec`.

    This is a recursive function, and the depth of recursion can be controlled by the `depth`
    keyword argument. The default value for `depth` is -1, which directs this function to
    traverse the entirety of a `zarr.Group`. This may be slow for large hierarchies, in which
    case setting `depth` to a positive integer can limit how deep into the hierarchy the
    recursion goes.

    Parameters
    ----------
    group : zarr.Group
        The Zarr group to model.
    depth: int
        An integer which may be no lower than -1. Determines how far into the tree to parse.

    Returns
    -------
    An instance of GroupSpec that represents the structure of the Zarr hierarchy.
    """

    result: GroupSpec[BaseAttr, BaseItem]
    attributes = group.attrs.asdict()
    members = {}
    if depth == 0:
        return GroupSpec(attributes=attributes, members={})
    for name, item in group.members():
        if isinstance(item, zarr.Array):
            # convert to dict before the final typed GroupSpec construction
            item_out = create_arrayspec(item).model_dump()
        elif isinstance(item, zarr.Group):
            # convert to dict before the final typed GroupSpec construction
            item_out = create_groupspec(item).model_dump()
        else:
            msg = (
                f"Unparseable object encountered: {type(item)}. Expected zarr.Array or zarr.Group."
            )

            raise TypeError(msg)
        members[name] = item_out

    result = GroupSpec(attributes=attributes, members=members)
    return result


async def write_groupspec(
    spec: GroupSpec[Any, Any], store: Store, path: str, *, overwrite: bool = False, **kwargs: object
) -> zarr.Group:
    """
    Serialize this `GroupSpec` to a Zarr group at a specific path in a `zarr.BaseStore`.
    This operation will create metadata documents in the store.
    Parameters
    ----------
    store : zarr.BaseStore
        The storage backend that will manifest the group and its contents.
    path : str
        The location of the group inside the store.
    overwrite: bool, default = False
        Whether to overwrite existing objects in storage to create the Zarr group.
    **kwargs : Any
        Additional keyword arguments that will be passed to `zarr.create` for creating
        sub-arrays.
    Returns
    -------
    zarr.Group
        A zarr group that is structurally identical to `self`.

    """

    spec_dict = spec.model_dump(exclude={"members": True})
    attrs = spec_dict.pop("attributes")
    if not overwrite:
        if await contains_group(StorePath(store, path), zarr_format=2):
            extant_group = zarr.group(store, path=path)
            if not spec.like(extant_group, exclude={"members"}):
                msg = (
                    f"A group already exists at path {path}. "
                    "That group is structurally dissimilar to the group you are trying to store."
                    "Call to_zarr with overwrite=True to overwrite that group."
                )
                raise ContainsGroupError(msg)
            else:
                # if the extant group is structurally identical to self, and overwrite is false,
                # then just return the extant group
                return extant_group

        elif await contains_array(StorePath(store, path), zarr_format=2):
            msg = (
                f"An array already exists at path {path}. "
                "Call to_zarr with overwrite=True to overwrite the array."
            )
            raise ContainsArrayError(msg)
    else:
        result = create_group(store=store, overwrite=overwrite, path=path, attributes=attrs)

    # consider raising an exception if a partial GroupSpec is provided
    if spec.members is not None:
        for name, member in spec.members.items():
            subpath = "/".join([path, name])
            member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

    return result


def create_arrayspec(array: zarr.Array) -> ArraySpec[BaseAttr]:
    """
    Create an `ArraySpec` from an instance of `zarr.Array`.

    Parameters
    ----------
    array : zarr.Array

    Returns
    -------
    ArraySpec
        An instance of `ArraySpec` with properties derived from `array`.

    Examples
    --------
    >>> import zarr
    >>> from pydantic_zarr.v2 import ArraySpec
    >>> x = zarr.create((10,10))
    >>> ArraySpec.from_zarr(x)
    ArraySpec(zarr_version=2, attributes={}, shape=(10, 10), chunks=(10, 10), dtype='<f8', fill_value=0.0, order='C', filters=None, dimension_separator='.', compressor={'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0})

    """
    return ArraySpec(
        shape=array.shape,
        chunks=array.chunks,
        dtype=str(array.dtype),
        # explicitly cast to numpy type and back to python
        # so that int 0 isn't serialized as 0.0
        fill_value=array.dtype.type(array.fill_value).tolist(),
        order=array.order,
        filters=array.filters,
        dimension_separator=array._dimension_separator,
        compressor=array.compressor,
        attributes=array.attrs.asdict(),
    )


async def write_arrayspec(
    spec: ArraySpec[Any],
    store: Store,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: object,
) -> zarr.Array:
    """
    Serialize an `ArraySpec` to a Zarr array at a specific path in a Zarr store. This operation
    will create metadata documents in the store, but will not write any chunks.

    Parameters
    ----------
    store : instance of zarr.BaseStore
        The storage backend that will manifest the array.
    path : str
        The location of the array inside the store.
    overwrite: bool, default = False
        Whether to overwrite existing objects in storage to create the Zarr array.
    **kwargs : Any
        Additional keyword arguments are passed to `zarr.create`.
    Returns
    -------
    zarr.Array
        A Zarr array that is structurally identical to `self`.
    """

    spec_dict = spec.model_dump()
    attrs = spec_dict.pop("attributes")

    if await contains_array(StorePath(store, path), zarr_format=2):
        extant_array = zarr.open_array(store, path=path, mode="r")

        if not spec.like(extant_array):
            if not overwrite:
                msg = (
                    f"An array already exists at path {path}. "
                    "That array is structurally dissimilar to the array you are trying to "
                    "store. Call to_zarr with overwrite=True to overwrite that array."
                )
                raise ContainsArrayError(msg)
        else:
            if not overwrite:
                # extant_array is read-only, so we make a new array handle that
                # takes **kwargs
                return zarr.open_array(store=extant_array.store, path=extant_array.path, **kwargs)
    result = zarr.create(store=store, path=path, overwrite=overwrite, **spec_dict, **kwargs)
    result.attrs.put(attrs)
    return result
