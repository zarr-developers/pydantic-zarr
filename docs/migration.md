# Migration guide

This release integrates the `zarr-metadata` package as a new runtime dependency
(`zarr-metadata>=0.3.0`). Several hand-rolled type definitions that duplicated types
now provided by `zarr-metadata` have been removed. The table below lists every removed
symbol and its replacement.

## Removed exports and replacements

### `pydantic_zarr.v3`

| Removed symbol | Replacement |
| --- | --- |
| `NamedConfig`, `AnyNamedConfig` | `zarr_metadata.NamedConfigV3` |
| `RegularChunking`, `RegularChunkingConfig` | plain dicts or `zarr_metadata.RegularChunkGridMetadata` |
| `DefaultChunkKeyEncoding`, `DefaultChunkKeyEncodingConfig` | `zarr_metadata.DefaultChunkKeyEncodingMetadata` |
| `V2ChunkKeyEncoding`, `V2ChunkKeyEncodingConfig` | `zarr_metadata.V2ChunkKeyEncodingMetadata` |
| `FillValue` union | `zarr_metadata.JSONValue` (loose) or `StrictArraySpec` (per-dtype) |

### `pydantic_zarr.v2`

| Removed symbol | Replacement |
| --- | --- |
| `MemoryOrder` | `zarr_metadata.ArrayOrderV2` |
| `DimensionSeparator` | `zarr_metadata.ArrayDimensionSeparatorV2` |
| `CodecDict` | `zarr_metadata.CodecMetadataV2` |

## New features in this release

### `ArraySpec.to_json()` and `GroupSpec.to_json()`

Both v2 and v3 `ArraySpec` and `GroupSpec` now expose a `to_json()` method that returns a
spec-conformant metadata document as a plain Python `dict`.

For v3, `to_json()` returns the full contents of a `zarr.json` file (array metadata or group
metadata, respectively).

For v2, `to_json()` returns the metadata with attributes included inline.

See [Usage (Zarr V3)](usage_zarr_v3.md#serializing-to-json-to_json) and
[Usage (Zarr V2)](usage_zarr_v2.md#serializing-to-json-to_json-and-to_store_json).

### `ArraySpec.to_store_json()` and `GroupSpec.to_store_json()` (v2 only)

The v2 `ArraySpec` and `GroupSpec` also expose `to_store_json()`, which returns the
on-disk document pair that Zarr v2 uses:

- `ArraySpec.to_store_json()` returns `{".zarray": ..., ".zattrs": ...}`.
- `GroupSpec.to_store_json()` returns `{".zgroup": ..., ".zattrs": ...}`.

### `StrictArraySpec`, per-dtype classes, `AnyStrictArraySpec`, and `StrictGroupSpec` (v3 only)

Three classes/types related to strict validation are now exported from `pydantic_zarr.v3`:

- **`StrictArraySpec`** — a directly constructible class. Its `fill_value` is annotated loosely
  (`JSONValue`) but validated at runtime against the rules for the given `data_type`. An
  unrecognized `data_type` is rejected. Use this when you want a single ergonomic constructor
  without caring about the static fill-value type.

- **Per-dtype classes** — one class per Zarr v3 data type
  (`BoolArraySpec`, `Int8ArraySpec`, `Int16ArraySpec`, `Int32ArraySpec`, `Int64ArraySpec`,
  `Uint8ArraySpec`, `Uint16ArraySpec`, `Uint32ArraySpec`, `Uint64ArraySpec`,
  `Float16ArraySpec`, `Float32ArraySpec`, `Float64ArraySpec`,
  `Complex64ArraySpec`, `Complex128ArraySpec`, `RawArraySpec`).
  Each has a precisely typed `fill_value` field. Use these when you want mypy/IDE to know the
  exact fill type at static analysis time.

- **`AnyStrictArraySpec`** — a discriminated union over all per-dtype classes. This is the
  validation target when you have a raw `dict` (e.g. parsed from `zarr.json`) and want Pydantic
  to route to the correct per-dtype class based on `data_type`.

`StrictGroupSpec` is the group counterpart. Its `members` must recursively contain only
`AnyStrictArraySpec` variants or `StrictGroupSpec` instances.

```python
from pydantic import TypeAdapter
from pydantic_zarr.v3 import AnyStrictArraySpec, Float64ArraySpec, StrictArraySpec

# Path 1: ergonomic single class
arr1 = StrictArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)

# Path 2: per-dtype class for precise static types
arr2 = Float64ArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)

# Path 3: validate a zarr.json dict into the precise per-dtype class
arr3 = TypeAdapter(AnyStrictArraySpec).validate_python(
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
print(type(arr3).__name__)
#> Float64ArraySpec
```

See [Usage (Zarr V3)](usage_zarr_v3.md#loose-vs-strict-validation) for a full example.
