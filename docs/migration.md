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

### `StrictArraySpec` and `StrictGroupSpec` (v3 only)

`StrictArraySpec` is a discriminated union exported from `pydantic_zarr.v3`. Each variant
corresponds to a specific Zarr v3 data type and couples the `fill_value` validation to that
dtype. For example, `float64` arrays accept `"NaN"`, `"Infinity"`, and `"-Infinity"` as fill
values, while integer types do not.

`StrictGroupSpec` is the group counterpart. Its `members` must recursively contain only
`StrictArraySpec` or `StrictGroupSpec` instances.

Because `StrictArraySpec` is a `Union` type alias rather than a class, validate data from
`dict` using `pydantic.TypeAdapter`:

```python
from pydantic import TypeAdapter, ValidationError
from pydantic_zarr.v3 import StrictArraySpec

ta = TypeAdapter(StrictArraySpec)
arr = ta.validate_python(
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
```

See [Usage (Zarr V3)](usage_zarr_v3.md#loose-vs-strict-validation) for a full example.
