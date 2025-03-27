"""
A zarr-python based backend for pydantic-zarr.
"""

import os
from typing import Any

from pydantic_zarr.v2 import ArraySpec, GroupSpec

from .versions import _zarr_python_version

zarr_python_version =  _zarr_python_version()
if zarr_python_version is not None and zarr_python_version.startswith("2."):
    import zarr
    from zarr.errors import ContainsArrayError, ContainsGroupError
    from zarr.storage import BaseStore, contains_array, contains_group, init_group
    def from_zarr() -> GroupSpec[Any, Any] | ArraySpec[Any]:
        """
        Create a groupspec or arrayspec from a zarr group or array.
        """
        pass

    def to_zarr(
            model: GroupSpec[Any, Any], 
            store: BaseStore,
            path: str, *,
            overwrite: bool = False, 
            **kwargs: object
        ) -> zarr.Group | zarr.Array:
            """
            Serialize a `GroupSpec` to a Zarr group at a specific path in a `zarr.BaseStore`.
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

            spec_dict = model.model_dump(exclude={"members": True})
            attrs = spec_dict.pop("attributes")
            if contains_group(store, path):
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

            elif contains_array(store, path) and not overwrite:
                msg = (
                    f"An array already exists at path {path}. "
                    "Call to_zarr with overwrite=True to overwrite the array."
                )
                raise ContainsArrayError(msg)
            else:
                init_group(store=store, overwrite=overwrite, path=path)

            result = zarr.group(store=store, path=path, overwrite=overwrite)
            result.attrs.put(attrs)
            # consider raising an exception if a partial GroupSpec is provided
            if model.members is not None:
                for name, member in model.members.items():
                    subpath = os.path.join(path, name)
                    member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

            return result