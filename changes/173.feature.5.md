The strict v3 metadata builder functions are now importable directly from `pydantic_zarr.v3`:
the codec builders (`bytes_codec`, `crc32c_codec`, `gzip_codec`, `zstd_codec`, `blosc_codec`,
`transpose_codec`, `sharding_indexed_codec`, `scale_offset_codec`, `cast_value_codec`), the
chunk-grid builders (`regular_grid`, `rectilinear_grid`), and the chunk-key-encoding builders
(`default_chunk_key_encoding`, `v2_chunk_key_encoding`). Previously these lived only in deep
per-element modules. Each returns a validated metadata dict ready to drop into a strict spec's
`codecs` / `chunk_grid` / `chunk_key_encoding` field.
