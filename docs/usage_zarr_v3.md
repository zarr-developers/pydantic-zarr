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

The *strict* classes couple `fill_value` validation to the array's `data_type`. For example,
`float64` accepts numeric fill values as well as the special strings `"NaN"`, `"Infinity"`, and
`"-Infinity"`, while integer types do not.

### Core vs Extra strict families

There are **two strict families**, distinguished by which Zarr vocabulary they accept:

| Feature | **Core** | **Extra** |
|---|---|---|
| Chunk grid | `regular` only | `regular` **or** `rectilinear` |
| Codecs (objects) | blosc, bytes, crc32c, gzip, sharding\_indexed, transpose, zstd | Core + `scale_offset`, `cast_value` |
| Codec name strings | known Core names only | known Core + Extra names |
| Unknown codec string | **rejected** | **rejected** |
| Unknown `data_type` | **rejected** | **rejected** |

Each family exposes the same three usage paths.

### Path 1 — Build a strict spec ergonomically: `CoreArraySpec` / `ExtraArraySpec`

`CoreArraySpec` and `ExtraArraySpec` are *single constructible classes* whose `fill_value` is
annotated loosely (`JSONValue`) but validated at runtime against the per-`data_type` rules. An
unrecognized `data_type` is rejected.

```python
from pydantic import ValidationError
from pydantic_zarr.v3 import CoreArraySpec

# float64 accepts "NaN" as a fill value
arr_float = CoreArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)
print(type(arr_float).__name__)
#> CoreArraySpec

# int64 rejects "NaN" — fill_value must be an integer
try:
    CoreArraySpec(
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

# Unknown codec names are rejected
try:
    CoreArraySpec(
        shape=(100,),
        data_type="float64",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0.0,
        codecs=[{"name": "made_up"}],
        attributes={},
    )
except ValidationError:
    print("Unknown codec 'made_up' rejected as expected")
    #> Unknown codec 'made_up' rejected as expected
```

### Path 2 — Build with precise static types: per-dtype classes

For each supported Zarr v3 data type there is a corresponding class in both families whose
`fill_value` field carries the exact static type, giving mypy and IDEs precise information.

**Core family** (importable from `pydantic_zarr.v3`):
`CoreBoolArraySpec`, `CoreInt8ArraySpec`, `CoreInt16ArraySpec`, `CoreInt32ArraySpec`,
`CoreInt64ArraySpec`, `CoreUint8ArraySpec`, `CoreUint16ArraySpec`, `CoreUint32ArraySpec`,
`CoreUint64ArraySpec`, `CoreFloat16ArraySpec`, `CoreFloat32ArraySpec`, `CoreFloat64ArraySpec`,
`CoreComplex64ArraySpec`, `CoreComplex128ArraySpec`, `CoreRawArraySpec`.

**Extra family** (importable from `pydantic_zarr.v3`):
`ExtraBoolArraySpec`, `ExtraInt8ArraySpec`, ..., `ExtraFloat64ArraySpec`, ...,
`ExtraComplex128ArraySpec`, `ExtraRawArraySpec` — one-to-one mirror of the Core family.

```python
from pydantic_zarr.v3 import CoreFloat64ArraySpec

arr = CoreFloat64ArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)
print(type(arr).__name__)
#> CoreFloat64ArraySpec
```

### Path 3 — Validate an existing `zarr.json` dict: `AnyCoreArraySpec` / `AnyExtraArraySpec`

`AnyCoreArraySpec` and `AnyExtraArraySpec` are discriminated unions over all per-dtype classes in
their respective families. They are the right validation targets when you have a raw `dict` (e.g.
parsed from `zarr.json`) and want Pydantic to route to the precise per-dtype class based on
`data_type`.

```python
from pydantic import TypeAdapter, ValidationError
from pydantic_zarr.v3 import AnyCoreArraySpec

ta = TypeAdapter(AnyCoreArraySpec)

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
#> CoreFloat64ArraySpec

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

`CoreGroupSpec` is the group counterpart for the Core family. Its `members` must recursively
contain only `AnyCoreArraySpec` variants or `CoreGroupSpec` instances:

```python
from pydantic_zarr.v3 import CoreFloat64ArraySpec, CoreGroupSpec

grp = CoreGroupSpec(
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
print(isinstance(grp.members["arr"], CoreFloat64ArraySpec))
#> True
```

### Core vs Extra difference in practice

The Extra family accepts `rectilinear` chunk grids and extra codecs such as `scale_offset`,
while Core rejects them.

```python
from pydantic import ValidationError
from pydantic_zarr.v3 import CoreArraySpec, ExtraArraySpec

# rectilinear chunk_grid: accepted by Extra, rejected by Core
extra_arr = ExtraArraySpec(
    shape=(20,),
    data_type="float64",
    chunk_grid={
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": (5, 5, 5, 5)},
    },
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value=0.0,
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)
print(type(extra_arr).__name__)
#> ExtraArraySpec

try:
    CoreArraySpec(
        shape=(20,),
        data_type="float64",
        chunk_grid={
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (5, 5, 5, 5)},
        },
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0.0,
        codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
        attributes={},
    )
except ValidationError:
    print("CoreArraySpec rejects rectilinear chunk_grid")
    #> CoreArraySpec rejects rectilinear chunk_grid

# scale_offset codec: accepted by Extra, rejected by Core
extra_arr2 = ExtraArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value=0.0,
    codecs=[
        {"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    ],
    attributes={},
)
print(type(extra_arr2).__name__)
#> ExtraArraySpec

try:
    CoreArraySpec(
        shape=(100,),
        data_type="float64",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0.0,
        codecs=[
            {"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ],
        attributes={},
    )
except ValidationError:
    print("CoreArraySpec rejects scale_offset codec")
    #> CoreArraySpec rejects scale_offset codec
```

> **Note:** Strict classes do not support generic attributes (`ArraySpec[MyAttrs]`). The
> `attributes` field is `Mapping[str, object]`. Users who need typed attributes should use the
> loose `ArraySpec[MyAttrs]`.
