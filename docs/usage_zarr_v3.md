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

The strict classes couple `fill_value` validation to the array's `data_type`. For example,
`float64` accepts numeric fill values as well as the special strings `"NaN"`, `"Infinity"`, and
`"-Infinity"`, while integer types do not. There are three usage paths:

### Path 1 — Build a strict spec ergonomically: `StrictArraySpec`

`StrictArraySpec` is a *single constructible class* exported from `pydantic_zarr.v3`. Its
`fill_value` is annotated loosely (`JSONValue`) but validated at runtime against the per-`data_type`
rules. An unrecognized `data_type` is rejected.

```python
from pydantic import ValidationError
from pydantic_zarr.v3 import StrictArraySpec

# float64 accepts "NaN" as a fill value
arr_float = StrictArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)
print(type(arr_float).__name__)
#> StrictArraySpec

# int64 rejects "NaN" — fill_value must be an integer
try:
    StrictArraySpec(
        shape=(100,),
        data_type="int64",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value="NaN",
        codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
        attributes={},
    )
except ValidationError:
    print("int64 + 'NaN' rejected as expected")
#> int64 + 'NaN' rejected as expected
```

### Path 2 — Build with precise static types: per-dtype classes

For each supported Zarr v3 data type there is a corresponding class whose `fill_value` field
carries the exact static type, giving mypy and IDEs precise information. All per-dtype classes are
importable from `pydantic_zarr.v3`:

`BoolArraySpec`, `Int8ArraySpec`, `Int16ArraySpec`, `Int32ArraySpec`, `Int64ArraySpec`,
`Uint8ArraySpec`, `Uint16ArraySpec`, `Uint32ArraySpec`, `Uint64ArraySpec`,
`Float16ArraySpec`, `Float32ArraySpec`, `Float64ArraySpec`,
`Complex64ArraySpec`, `Complex128ArraySpec`, `RawArraySpec`.

```python
from pydantic_zarr.v3 import Float64ArraySpec

arr = Float64ArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)
print(type(arr).__name__)
#> Float64ArraySpec
```

### Path 3 — Validate an existing `zarr.json` dict: `AnyStrictArraySpec`

`AnyStrictArraySpec` is the discriminated union over all per-dtype classes. It is the right
validation target when you have a raw `dict` (e.g. parsed from `zarr.json`) and want Pydantic to
route to the precise per-dtype class based on `data_type`.

```python
from pydantic import TypeAdapter, ValidationError
from pydantic_zarr.v3 import AnyStrictArraySpec

ta = TypeAdapter(AnyStrictArraySpec)

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
#> Float64ArraySpec

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
`AnyStrictArraySpec` variants or `StrictGroupSpec` instances:

```python
from pydantic_zarr.v3 import Float64ArraySpec, StrictGroupSpec

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
print(isinstance(grp.members["arr"], Float64ArraySpec))
#> True
```

> **Note:** Strict classes do not support generic attributes (`ArraySpec[MyAttrs]`). The
> `attributes` field is `Mapping[str, object]`. Users who need typed attributes should use the
> loose `ArraySpec[MyAttrs]`.
