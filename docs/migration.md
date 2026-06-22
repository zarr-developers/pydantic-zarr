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
| `FillValue` union | `zarr_metadata.JSONValue` (loose) or `CoreArraySpec`/`ExtraArraySpec` (per-dtype) |

### `pydantic_zarr.v2`

| Removed symbol | Replacement |
| --- | --- |
| `MemoryOrder` | `zarr_metadata.ArrayOrderV2` |
| `DimensionSeparator` | `zarr_metadata.ArrayDimensionSeparatorV2` |
| `CodecDict` | `zarr_metadata.CodecMetadataV2` |

## Loose-spec behavioral changes

The loose `ArraySpec`/`GroupSpec` keep the same field *names*, but adopting `zarr-metadata`
changed some field *types* and validation behavior. If you use the loose specs (not the new
`Core`/`Extra` strict families), check these:

| Change | Before | After | What to do |
| --- | --- | --- | --- |
| Sequence fields on the model | `list` | `tuple` | v2 `ArraySpec.filters` and v3 `chunk_grid`'s `chunk_shape` are now tuples on the model and in `model_dump()`. Code reading `spec.filters` as a mutable list (`.append(...)`, `isinstance(x, list)`) must adjust. |
| `to_json()` output | JSON arrays | JSON arrays | **Unchanged** — `to_json()` still emits lists, so the on-disk `zarr.json` / `.zarray` is byte-compatible. |
| Bare-string named-configs (v3) | rejected | accepted | `chunk_grid`, `chunk_key_encoding`, and `codecs` now accept the bare-string short form (e.g. `chunk_grid="regular"`) as well as the `{name, configuration}` object form. |
| `fill_value` / `dtype` type | narrow unions | `zarr_metadata.JSONValue` / zarr-metadata dtype types | Loose specs still validate these as syntax only. For dtype-aware `fill_value` validation, use `CoreArraySpec` / `ExtraArraySpec`. |

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

### Core and Extra strict families (v3 only)

Two families of strict validation classes are now exported from `pydantic_zarr.v3`:

**Core family** — restricts chunk grids to `regular` and codecs to the seven types defined in
the core Zarr v3 spec (blosc, bytes, crc32c, gzip, sharding_indexed, transpose, zstd):

- **`CoreArraySpec`** — directly constructible. Its `fill_value` is annotated loosely
  (`JSONValue`) but validated at runtime against the rules for the given `data_type`. An
  unrecognized `data_type` is rejected.

- **Core per-dtype classes** —
  `CoreBoolArraySpec`, `CoreInt8ArraySpec`, `CoreInt16ArraySpec`, `CoreInt32ArraySpec`,
  `CoreInt64ArraySpec`, `CoreUint8ArraySpec`, `CoreUint16ArraySpec`, `CoreUint32ArraySpec`,
  `CoreUint64ArraySpec`, `CoreFloat16ArraySpec`, `CoreFloat32ArraySpec`, `CoreFloat64ArraySpec`,
  `CoreComplex64ArraySpec`, `CoreComplex128ArraySpec`, `CoreRawArraySpec`.
  Each has a precisely typed `fill_value` field for static analysis.

- **`AnyCoreArraySpec`** — discriminated union over all Core per-dtype classes. Use with
  `TypeAdapter(AnyCoreArraySpec).validate_python(doc)` to route by `data_type`.

- **`CoreGroupSpec`** — group whose members are recursively `AnyCoreArraySpec | CoreGroupSpec`.

**Extra family** — extends Core by also accepting `rectilinear` chunk grids and two additional
codec types (`scale_offset`, `cast_value`):

- **`ExtraArraySpec`**, **Extra per-dtype classes** (`ExtraBoolArraySpec`, `ExtraInt8ArraySpec`,
  ..., `ExtraFloat64ArraySpec`, ..., `ExtraComplex128ArraySpec`, `ExtraRawArraySpec`),
  **`AnyExtraArraySpec`**, **`ExtraGroupSpec`** — exact mirrors of the Core family.

```python
from pydantic import TypeAdapter
from pydantic_zarr.v3 import AnyCoreArraySpec, CoreArraySpec, CoreFloat64ArraySpec

# Path 1: ergonomic single class
arr1 = CoreArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)

# Path 2: per-dtype class for precise static types
arr2 = CoreFloat64ArraySpec(
    shape=(100,),
    data_type="float64",
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [100]}},
    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
    fill_value="NaN",
    codecs=[{"name": "bytes", "configuration": {"endian": "little"}}],
    attributes={},
)

# Path 3: validate a zarr.json dict into the precise per-dtype class
arr3 = TypeAdapter(AnyCoreArraySpec).validate_python(
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
#> CoreFloat64ArraySpec
```

See [Usage (Zarr V3)](usage_zarr_v3.md#loose-vs-strict-validation) for a full example.
