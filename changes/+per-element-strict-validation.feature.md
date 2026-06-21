Added per-element builder functions (`pydantic_zarr.strict.v3.codec.transpose`, `...chunk_grid.regular`,
etc.) for constructing Zarr v3 codec and chunk-grid metadata with typed arguments and validation, and
strengthened strict `Core`/`Extra` validation: codec/grid intrinsic checks (transpose must be a
permutation, chunk shapes must be positive, blosc/gzip level in 0-9), array-dimensionality consistency
(chunk_grid/dimension_names/sharding must match the array's rank), and codec-pipeline type-flow
(cast_value scalars must be dtype-correct for their position in the codec pipeline).
