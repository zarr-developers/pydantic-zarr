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

### Ergonomic construction

The per-dtype and family-wide strict classes expose a `.create()` classmethod that fills every
field with a sensible default, so you can focus only on the fields you care about:

```python {group="strict-v3"}
import numpy as np

# bare call — defaults to float64, empty shape, NaN fill_value
bare = CoreArraySpec.create()
print(type(bare).__name__)
#> CoreArraySpec
print(bare.data_type)
#> float64

# specify only shape — everything else gets sensible defaults
shaped = CoreArraySpec.create(shape=(10, 10))
print(type(shaped).__name__)
#> CoreArraySpec
print(shaped.shape)
#> (10, 10)
print(shaped.data_type)
#> float64

# per-dtype class with a single override
inf_arr = CoreFloat64ArraySpec.create(fill_value="Infinity")
print(type(inf_arr).__name__)
#> CoreFloat64ArraySpec
print(inf_arr.fill_value)
#> Infinity

# derive a strict spec directly from a numpy array
derived = CoreFloat64ArraySpec.from_array(np.zeros((4, 4), dtype="float64"))
print(type(derived).__name__)
#> CoreFloat64ArraySpec
print(derived.shape)
#> (4, 4)
print(derived.data_type)
#> float64
```

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
# chunk_shapes has one entry per array dimension; COMMON uses shape=(100,), so one entry here
rectilinear = {
    "name": "rectilinear",
    "configuration": {"kind": "inline", "chunk_shapes": (5,)},
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

## Building codec and chunk-grid metadata

`pydantic_zarr.v3` exports per-element *builder* functions that construct codec, chunk-grid, and
chunk-key-encoding metadata dicts with typed arguments and validate them at build time. Every
builder returns a plain `dict` (a `TypedDict`-shaped mapping) that is ready to drop into any
`codecs` list, `chunk_grid`, or `chunk_key_encoding` field. The codec builders are named
`*_codec` (`bytes_codec`, `blosc_codec`, `transpose_codec`, `sharding_indexed_codec`, …), the grid
builders `*_grid` (`regular_grid`, `rectilinear_grid`), and the chunk-key-encoding builders
`default_chunk_key_encoding` / `v2_chunk_key_encoding`.

```python {group="builders"}
from pydantic_zarr.v3 import (
    blosc_codec,
    regular_grid,
    sharding_indexed_codec,
    transpose_codec,
)

# regular chunk grid — a plain dict ready for any ArraySpec
chunk_grid = regular_grid((10, 10))
print(chunk_grid)
#> {'name': 'regular', 'configuration': {'chunk_shape': (10, 10)}}

# transpose codec (axes permutation)
txp = transpose_codec((1, 0))
print(txp)
#> {'name': 'transpose', 'configuration': {'order': (1, 0)}}

# blosc codec (cname, clevel, shuffle, blocksize)
bl = blosc_codec("zstd", 5, "shuffle", 0)
print(bl)
"""
{
    'name': 'blosc',
    'configuration': {
        'cname': 'zstd',
        'clevel': 5,
        'shuffle': 'shuffle',
        'blocksize': 0,
    },
}
"""

# sharding_indexed — inner chunk shape; defaults to (bytes,) inner codecs + (bytes, crc32c) index
shard = sharding_indexed_codec((4, 4))
print(shard)
"""
{
    'name': 'sharding_indexed',
    'configuration': {
        'chunk_shape': (4, 4),
        'codecs': ({'name': 'bytes', 'configuration': {'endian': 'little'}},),
        'index_codecs': (
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {'name': 'crc32c'},
        ),
    },
}
"""
```

Builders validate at construction time — invalid arguments raise `ValueError` immediately:

```python {group="builders"}
try:
    transpose_codec((0, 0))          # (0, 0) is not a permutation of range(2)
except ValueError as e:
    print(e)
    #> transpose order (0, 0) is not a permutation of range(2)

try:
    blosc_codec("zstd", 99, "shuffle", 0)   # clevel must be in [0, 9]
except ValueError as e:
    print(e)
    #> blosc clevel 99 out of range [0, 9]
```

## Validation errors strict mode catches

Strict specs validate three things a loose `ArraySpec` does not: each codec/grid is internally
consistent, the codec pipeline is structurally valid and dimensionally agrees with the array, and
the codec data-type flow is correct. The checks fire whether you build a spec in Python or validate
a parsed `zarr.json` dict. Each raises one clear message; read it from `exc.errors()[0]["msg"]`.

```python {group="strict-errors"}
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import (
    AnyCoreArraySpec,
    CoreArraySpec,
    ExtraArraySpec,
    bytes_codec,
    cast_value_codec,
    sharding_indexed_codec,
    transpose_codec,
)


def reason(exc: ValidationError) -> str:
    # the strict checks raise one ValueError; pydantic prefixes it with "Value error, "
    return exc.errors()[0]["msg"].removeprefix("Value error, ")


# every example below varies one field of this valid 2-D baseline
COMMON = {
    "shape": (8, 8),
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "attributes": {},
}
BYTES = bytes_codec()  # the array->bytes codec every pipeline needs


# 1. dimensionality — the chunk grid's rank must match the array's rank
try:
    CoreArraySpec(
        data_type="int32",
        fill_value=0,
        codecs=[BYTES],
        **{**COMMON, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8,)}}},
    )
except ValidationError as e:
    print(reason(e))
    #> chunk_grid ndim 1 != array ndim 2


# 2. sharding — the inner chunk shape must evenly divide the outer chunk grid
try:
    CoreArraySpec(data_type="int32", fill_value=0, codecs=[sharding_indexed_codec((3, 3))], **COMMON)
except ValidationError as e:
    print(reason(e))
    #> sharding inner (3, 3) does not evenly divide outer (8, 8)


# 3. pipeline — exactly one array->bytes codec is required (transpose alone leaves zero)
try:
    CoreArraySpec(data_type="int32", fill_value=0, codecs=[transpose_codec((1, 0))], **COMMON)
except ValidationError as e:
    print(reason(e))
    #> codec pipeline must have exactly one array->bytes codec, found 0


# 4. cast_value (extra family) — scalars must be valid for the dtype at that pipeline position.
#    Here the array is int32, so the encode input scalar "NaN" is not a valid int32 value.
try:
    ExtraArraySpec(
        data_type="int32",
        fill_value=0,
        codecs=[cast_value_codec("float64", scalar_map={"encode": [("NaN", 0.0)]}), BYTES],
        **COMMON,
    )
except ValidationError as e:
    print(reason(e))
    #> cast_value encode input scalar 'NaN' invalid for dtype 'int32'
```

The same checks run when you validate a raw `dict` parsed from a `zarr.json` file. Strict mode also
rejects the bare-string short form for any codec whose configuration is required — `transpose` needs
an `order`, so `"transpose"` on its own is invalid (only `bytes`, `crc32c`, and `scale_offset` may
appear as bare strings):

```python {group="strict-errors"}
adapter = TypeAdapter(AnyCoreArraySpec)

# a bare-string "transpose" is rejected — transpose requires a configuration
try:
    adapter.validate_python(
        {"data_type": "int32", "fill_value": 0, "codecs": ["transpose", BYTES], **COMMON}
    )
except ValidationError:
    print("bare 'transpose' rejected (configuration required)")
    #> bare 'transpose' rejected (configuration required)

# the object form, with the required configuration, validates
ok = adapter.validate_python(
    {
        "data_type": "int32",
        "fill_value": 0,
        "codecs": [transpose_codec((1, 0)), BYTES],
        **COMMON,
    }
)
print(type(ok).__name__)
#> CoreInt32ArraySpec
```
