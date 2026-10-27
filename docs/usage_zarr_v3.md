# Usage (Zarr V3)

## Defining Zarr v3 hierarchies

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec, NamedConfig

array_attributes = {"baz": [1, 2, 3]}
group_attributes = {"foo": 42, "bar": False}

array_spec = ArraySpec(
    attributes=array_attributes,
    shape=[1000, 1000],
    dimension_names=["rows", "columns"],
    data_type="uint8",
    chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
    chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
    codecs=[
        NamedConfig(name="bytes"),
        NamedConfig(name="gzip", configuration={"level": 1}),
    ],
    storage_transformers=(),
    fill_value=0,
)

spec = GroupSpec(attributes=group_attributes, members={"array": array_spec})
print(spec.model_dump_json(indent=2))
"""
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "foo": 42,
    "bar": false
  },
  "members": {
    "array": {
      "zarr_format": 3,
      "node_type": "array",
      "attributes": {
        "baz": [
          1,
          2,
          3
        ]
      },
      "shape": [
        1000,
        1000
      ],
      "data_type": "uint8",
      "chunk_grid": {
        "name": "regular",
        "configuration": {
          "chunk_shape": [
            1000,
            100
          ]
        }
      },
      "chunk_key_encoding": {
        "name": "default",
        "configuration": {
          "separator": "/"
        }
      },
      "fill_value": 0,
      "codecs": [
        {
          "name": "bytes"
        },
        {
          "name": "gzip",
          "configuration": {
            "level": 1
          }
        }
      ],
      "storage_transformers": [],
      "dimension_names": [
        "rows",
        "columns"
      ]
    }
  }
}
"""
```

For a single-byte data type like `uint8`, the `bytes` codec needs no `endian`
configuration; a no-op `{"endian": ...}` configuration passed anyway is dropped
at validation, matching the spelling `zarr-python` stores.
