# Usage (Zarr V3)

## Defining Zarr v3 hierarchies

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec
from zarr_metadata import NamedConfigV3

array_attributes = {"baz": [1, 2, 3]}
group_attributes = {"foo": 42, "bar": False}

array_spec = ArraySpec(
    attributes=array_attributes,
    shape=[1000, 1000],
    dimension_names=["rows", "columns"],
    data_type="uint8",
    chunk_grid=NamedConfigV3(name="regular", configuration={"chunk_shape": [1000, 100]}),
    chunk_key_encoding=NamedConfigV3(name="default", configuration={"separator": "/"}),
    codecs=[NamedConfigV3(name="gzip", configuration={"level": 1})],
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
      "storage_transformers": [],
      "dimension_names": [
        "rows",
        "columns"
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
          "name": "gzip",
          "configuration": {
            "level": 1
          }
        }
      ]
    }
  }
}
"""
```

## Serializing to JSON (`to_json`)

Both `ArraySpec` and `GroupSpec` provide a `to_json()` method that returns a spec-conformant
metadata document (the contents of `zarr.json`) as a plain Python `dict`.

```python
import json
from pydantic_zarr.v3 import ArraySpec, GroupSpec

arr = ArraySpec(
    shape=(100, 100),
    data_type="float32",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value=0.0,
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)

doc = arr.to_json()
print(type(doc))
#> <class 'dict'>
print(doc["zarr_format"])
#> 3
print(doc["data_type"])
#> float32

grp = GroupSpec(attributes={"owner": "me"}, members={"arr": arr})
grp_doc = grp.to_json()
# members are excluded — to_json() returns only the zarr.json content for this node
print(list(grp_doc.keys()))
#> ['zarr_format', 'node_type', 'attributes']
```

## Loose vs strict validation

`ArraySpec` is the *loose* class. It accepts any syntactically valid `data_type` string and any
JSON-compatible `fill_value`. This is suitable for generic code that does not know the dtype at
construction time, or for typed-attribute workflows where `ArraySpec[MyAttrs]` is used.

`StrictArraySpec` is a *strict* discriminated union exported from `pydantic_zarr.v3`. Pydantic
selects the appropriate per-dtype variant based on the `data_type` field and then validates the
`fill_value` against that dtype's allowed values. For example, `float64` accepts numeric fill
values as well as the special strings `"NaN"`, `"Infinity"`, and `"-Infinity"`, while integer
types do not.

Because `StrictArraySpec` is a `Union` type alias (not a class), you cannot call
`StrictArraySpec(...)` directly. Use `TypeAdapter` to validate data from a `dict`, or construct
one of the concrete per-dtype classes (e.g. `_Float64ArraySpec`) directly. In practice the most
common pattern is to use `TypeAdapter`:

```python
from pydantic import TypeAdapter, ValidationError
from pydantic_zarr.v3 import StrictArraySpec

ta = TypeAdapter(StrictArraySpec)

# float64 accepts "NaN" as a fill value
arr_float = ta.validate_python(
    {
        "shape": (100,),
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "NaN",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
)
print(type(arr_float).__name__)
#> _Float64ArraySpec

# int64 rejects "NaN" — fill_value must be an integer
try:
    ta.validate_python(
        {
            "shape": (100,),
            "data_type": "int64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
        }
    )
except ValidationError:
    print("int64 + 'NaN' rejected as expected")
#> int64 + 'NaN' rejected as expected
```

`StrictGroupSpec` is the group counterpart. Its `members` must recursively contain only
`StrictArraySpec` or `StrictGroupSpec` instances:

```python
from pydantic_zarr.v3 import StrictGroupSpec

grp = StrictGroupSpec(
    attributes={},
    members={
        "arr": {
            "zarr_format": 3,
            "node_type": "array",
            "shape": (100,),
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
        }
    },
)
from pydantic_zarr._strict_v3 import _Float64ArraySpec
print(isinstance(grp.members["arr"], _Float64ArraySpec))
#> True
```

> **Note:** `StrictArraySpec` does not support generic attributes (`ArraySpec[MyAttrs]`). The
> `attributes` field is `Mapping[str, object]`. Users who need typed attributes should use the
> loose `ArraySpec[MyAttrs]`.
