Adopting `zarr-metadata` changed the field *types* of the loose `ArraySpec`/`GroupSpec` (the
field *names* are unchanged). Two user-visible behavioral differences:

- Sequence-valued fields are now tuples on the model and in `model_dump()` (Python mode): v2
  `ArraySpec.filters` and v3 `chunk_grid` `chunk_shape` are tuples rather than lists. The
  spec-conformant `to_json()` output is unaffected — it still emits JSON arrays (lists). Code that
  read `spec.filters` expecting a mutable `list` (e.g. `.append(...)` or `isinstance(x, list)`)
  must adjust.
- The loose v3 `chunk_grid`, `chunk_key_encoding`, and `codecs` fields now accept the bare-string
  short form (e.g. `chunk_grid="regular"`) in addition to the `{name, configuration}` object form.
  Previously only the object form was accepted. `fill_value` (v2/v3) and `dtype`/`data_type` are
  now typed as `zarr_metadata.JSONValue` / the zarr-metadata dtype types; loose specs continue to
  validate these as syntax only (use the `Core`/`Extra` strict families for dtype-aware checks).

This is a behavioural change for code that relied on the previous list-typed attributes or on
bare-string named-configs being rejected.
