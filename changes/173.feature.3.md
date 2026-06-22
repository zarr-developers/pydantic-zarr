Strict `Core`/`Extra` v3 array specs now support ergonomic construction: `.create()` builds a
spec with sensible defaults for every field (e.g. `CoreArraySpec.create(shape=(10, 10))`), and
`.from_array(array)` derives a strict spec directly from a numpy or zarr array, mirroring the
loose `ArraySpec.from_array`.
