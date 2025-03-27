from __future__ import annotations

from typing import Any, Literal

from pydantic import Json, AnyUrl, BaseModel, Field

from pydantic_zarr._v2 import ArrayMetadataSpec, ArraySpec

DataType = Literal[
    "bool",
    "char",
    "byte", 
    "int4",
    "int8",
    "uint8","int16" , "uint16" , "int32" , "uint32" , "int64" , "uint64" , "float8_e4m3fn" , "float8_e4m3fnuz" , "float8_e4m3b11fnuz" , "float8_e5m2" , "float8_e5m2fnuz" , "float16" , "bfloat16" , "float32" , "float64" , "complex64" , "complex128" , "string" , "ustring" , "json"]

class IndexTransform(BaseModel, frozen=True):
    input_rank: int
    input_exclusive_min: int | tuple[int, ...]
    input_exclusive_max: int | tuple[int, ...]
    input_shape: int | tuple[int, ...]
    input_labels: tuple[str, ...]
    output: tuple[OutputIndexMap, ...]

class OutputIndexMap(BaseModel, frozen=True):
    input_dimension: int = Field(..., description="Input dimension index")
    output_dimension: int = Field(..., description="Output dimension index")
    strides: tuple[int, ...] = Field(..., description="Strides")
    offset: int = Field(..., description="Offset")
    size: int = Field(..., description="Size")
    broadcast_sizes: tuple[int, ...] | None = Field(None, description="Broadcast sizes")
    broadcast_dimensions: tuple[int, ...] | None = Field(None, description="Broadcast dimensions")

class KVStore(BaseModel, frozen=True):
    driver: Literal["file", "memory"]

class ZarrDriver(BaseModel, frozen=True):
    driver: Literal["zarr"]
    kvstore: AnyUrl | KVStore
    context: dict[str, Any] | None = None
    dtype: DataType | None = None
    rank: int | None = None
    transform: IndexTransform | None = None
    open: bool = True
    create: bool = False
    delete_existing: bool = False
    assume_metadata: bool = False
    assume_cached_metadata: bool = False
    metadata: ArrayMetadataSpec



