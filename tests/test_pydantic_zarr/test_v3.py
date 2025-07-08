import numpy as np
import zarr

from pydantic_zarr.v3 import (
    ArraySpec,
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingConfig,
    GroupSpec,
    NamedConfig,
    RegularChunking,
    RegularChunkingConfig,
)


def test_serialize_deserialize() -> None:
    array_attributes = {"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]}

    group_attributes = {"group": True}

    array_spec = ArraySpec(
        attributes=array_attributes,
        shape=[1000, 1000],
        dimension_names=["rows", "columns"],
        data_type="float64",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        codecs=[NamedConfig(name="GZip", configuration={"level": 1})],
        fill_value="NaN",
        storage_transformers=[],
    )

    GroupSpec(attributes=group_attributes, members={"array": array_spec})


def test_from_array() -> None:
    array_spec = ArraySpec.from_array(np.arange(10))
    assert array_spec == ArraySpec(
        zarr_format=3,
        node_type="array",
        attributes={},
        shape=(10,),
        data_type="<i8",
        chunk_grid=RegularChunking(
            name="regular", configuration=RegularChunkingConfig(chunk_shape=[10])
        ),
        chunk_key_encoding=DefaultChunkKeyEncoding(
            name="default", configuration=DefaultChunkKeyEncodingConfig(separator="/")
        ),
        fill_value=0,
        codecs=[],
        storage_transformers=[],
        dimension_names=[None],
    )


def test_from_zarr() -> None:
    array_spec = ArraySpec.from_zarr(zarr.ones((1, 2)))
    assert array_spec == ArraySpec(
        zarr_format=3,
        node_type="array",
        attributes={},
        shape=(1, 2),
        data_type="<f8",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": (1, 2)}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        fill_value=1.0,
        codecs=(NamedConfig(name="zstd", configuration={"level": 0, "checksum": False}),),
        storage_transformers=(),
        dimension_names=None,
    )
