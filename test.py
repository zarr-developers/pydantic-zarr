import zarr

from pydantic_zarr.v3 import ArraySpec

ArraySpec.from_array(zarr.empty((1, 1, 1)))
