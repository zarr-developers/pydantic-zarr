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

!!! note
    Strict validation checks the JSON *type* of `fill_value` against the data type (as modelled by
    `zarr-metadata`); it does not enforce numeric range. An out-of-range integer fill value (e.g.
    `999` for `int8`) is not rejected, and because Python treats `bool` as a subtype of `int`, a
    boolean fill value is accepted for integer dtypes.

### Core vs Extra strict families

There are **two strict families**, distinguished by which Zarr vocabulary they accept:

| Feature | **Core** | **Extra** |
|---|---|---|
| Chunk grid | `regular` only | `regular` **or** `rectilinear` |
| Codecs (objects) | blosc, bytes, crc32c, gzip, sharding\_indexed, transpose, zstd | Core + `scale_offset`, `cast_value` |
| Codec name strings | known Core names only | known Core + Extra names |
| Unknown codec string | **rejected** | **rejected** |
| Unknown `data_type` | **rejected** | **rejected** |

The `Extra` family is the right default for code that reads real-world archives (many use
`scale_offset` and similar extensions); reach for `Core` when you specifically want to reject
anything outside the core spec.

### Which class should I use?

Within a family there are three array types. They differ only in *how* you reach them and *how
precisely* `fill_value` is typed — they validate the same rules:

| You have… | Use | `fill_value` static type |
|---|---|---|
| a known dtype, want it checked but typed loosely | `CoreArraySpec` / `ExtraArraySpec` | `JSONValue` (any JSON) |
| a known dtype, want precise IDE/mypy types | `CoreFloat64ArraySpec`, … (per-dtype) | the exact per-dtype type |
| a raw `dict` (e.g. parsed `zarr.json`) to validate | `AnyCoreArraySpec` / `AnyExtraArraySpec` via `TypeAdapter` | routed to the matching per-dtype class |

Groups use `CoreGroupSpec` / `ExtraGroupSpec`, whose `members` are validated recursively.

Every strict array also needs `shape`, `chunk_grid`, `chunk_key_encoding`, `codecs`, and
`attributes`. To keep the examples focused on the strict behaviour, each one starts by collecting
those universal fields into a `COMMON` dict and spreads it with `**COMMON`.

### Constructing and the `data_type` ↔ `fill_value` coupling

`CoreArraySpec` / `ExtraArraySpec` are single constructible classes. `fill_value` is annotated
loosely (`JSONValue`) but validated at runtime against the per-`data_type` rules; an unrecognized
`data_type` or codec is rejected:

```python {group="strict-v3"}
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import (
    AnyCoreArraySpec,
    CoreArraySpec,
    CoreFloat64ArraySpec,
    CoreGroupSpec,
    ExtraArraySpec,
)

# the universal fields every strict array needs; spread with **COMMON below
COMMON = {
    "shape": (100,),
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100]}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    "attributes": {},
}

# float64 accepts "NaN"
arr = CoreArraySpec(data_type="float64", fill_value="NaN", **COMMON)
print(type(arr).__name__)
#> CoreArraySpec

# int64 rejects "NaN" — fill_value must be an integer
try:
    CoreArraySpec(data_type="int64", fill_value="NaN", **COMMON)
except ValidationError:
    print("int64 + 'NaN' rejected")
    #> int64 + 'NaN' rejected

# an unknown codec name is rejected
try:
    CoreArraySpec(
        data_type="float64",
        fill_value=0.0,
        **{**COMMON, "codecs": [{"name": "made_up"}]},
    )
except ValidationError:
    print("unknown codec 'made_up' rejected")
    #> unknown codec 'made_up' rejected
```

### Per-dtype classes: precise static types

For each data type there is a class in both families whose `fill_value` field carries the *exact*
static type, so mypy and your IDE know what a valid fill value is. This is the difference from the
single `CoreArraySpec` class, whose `fill_value` is the wide `JSONValue`:

```python {group="strict-v3"}
# the single class's fill_value is the wide JSONValue (any JSON; checked only at runtime):
#   CoreArraySpec.fill_value:        JSONValue
# the per-dtype class's fill_value is the exact float64 fill type, which mypy/IDEs see:
#   CoreFloat64ArraySpec.fill_value: float | int | "NaN" | "Infinity" | "-Infinity" | HexFloat64
print(CoreArraySpec.model_fields["fill_value"].annotation.__name__)
#> JSONValue

arr = CoreFloat64ArraySpec(data_type="float64", fill_value="Infinity", **COMMON)
print(type(arr).__name__)
#> CoreFloat64ArraySpec
```

The full per-dtype class list (each importable from `pydantic_zarr.v3`, with an `Extra` mirror):
`CoreBoolArraySpec`, `CoreInt8ArraySpec`/`16`/`32`/`64`, `CoreUint8ArraySpec`/`16`/`32`/`64`,
`CoreFloat16ArraySpec`/`32`/`64`, `CoreComplex64ArraySpec`/`128`, `CoreRawArraySpec`.

### Validating a raw `dict`: `AnyCoreArraySpec` / `AnyExtraArraySpec`

When you have a `dict` (e.g. parsed from a `zarr.json`), validate it through the family's
discriminated union — Pydantic routes to the precise per-dtype class by `data_type`:

```python {group="strict-v3"}
ta = TypeAdapter(AnyCoreArraySpec)
arr = ta.validate_python({"data_type": "float64", "fill_value": "NaN", **COMMON})
print(type(arr).__name__)  # routed to the precise per-dtype class
#> CoreFloat64ArraySpec
```

`CoreGroupSpec` is the group counterpart; its `members` are validated recursively to
`AnyCoreArraySpec` variants or nested `CoreGroupSpec` instances (raw dicts are coerced):

```python {group="strict-v3"}
grp = CoreGroupSpec(
    attributes={},
    members={"arr": {"data_type": "float64", "fill_value": "NaN", **COMMON}},
)
print(isinstance(grp.members["arr"], CoreFloat64ArraySpec))
#> True
```

### Core vs Extra in practice

The Extra family accepts `rectilinear` chunk grids and extension codecs such as `scale_offset`;
Core rejects them:

```python {group="strict-v3"}
rectilinear = {
    "name": "rectilinear",
    "configuration": {"kind": "inline", "chunk_shapes": (5, 5, 5, 5)},
}

# Extra accepts a rectilinear chunk grid; Core rejects it
extra_arr = ExtraArraySpec(
    data_type="float64", fill_value=0.0, **{**COMMON, "chunk_grid": rectilinear}
)
print(type(extra_arr).__name__)
#> ExtraArraySpec

try:
    CoreArraySpec(
        data_type="float64", fill_value=0.0, **{**COMMON, "chunk_grid": rectilinear}
    )
except ValidationError:
    print("Core rejects rectilinear chunk_grid")
    #> Core rejects rectilinear chunk_grid

# Extra accepts the scale_offset codec; Core rejects it
scale_offset = [
    {"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},
    {"name": "bytes", "configuration": {"endian": "little"}},
]
extra_arr2 = ExtraArraySpec(
    data_type="float64", fill_value=0.0, **{**COMMON, "codecs": scale_offset}
)
print(type(extra_arr2).__name__)
#> ExtraArraySpec

try:
    CoreArraySpec(
        data_type="float64", fill_value=0.0, **{**COMMON, "codecs": scale_offset}
    )
except ValidationError:
    print("Core rejects scale_offset codec")
    #> Core rejects scale_offset codec
```

> **Note:** Strict classes do not support generic attributes. The `attributes` field is
> `Mapping[str, object]`; if you need typed attributes, use the loose `ArraySpec[MyAttrs]`.
