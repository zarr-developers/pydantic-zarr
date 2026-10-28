# xarray-style datasets

When [xarray](https://xarray.dev) writes a netCDF-style `Dataset` with
`Dataset.to_zarr(..., zarr_format=3)`, it produces a flat layout: a root
group whose attributes are the dataset's global attributes (`Conventions`,
`title`, ...), with one array member per data variable and per coordinate.
Dimensions use the native Zarr V3 `dimension_names` field. Coordinates come
in two kinds: a *dimension coordinate* is a 1-D array named after its own
dimension (the netCDF convention), while an *auxiliary coordinate* may have
any rank — 2-D latitude and longitude on a curvilinear grid, say — as long
as each of its dimensions matches a dimension of the data arrays; data
variables name their auxiliary coordinates in the CF `coordinates`
attribute. Every array carries CF attributes such as `units`,
`standard_name`, and `long_name`; `_FillValue` appears as a base64 string,
because xarray's Zarr V3 fill-value coder encodes the attribute as the
base64 of the value's bytes (eight bytes — the value is cast to `float64`,
even for a `float32` variable). This page builds a typed model of that layout: CF attribute
models with reserved-name aliases, `ArraySpec` subclasses for variables and
coordinates, a cross-member dimension-consistency validator, and a closed
member set for one concrete dataset. The same model classes validate real
xarray-written stores read back with `from_zarr`. (xarray also writes a
`consolidated_metadata` block into the root `zarr.json` by default;
pydantic-zarr does not model consolidated metadata yet.)

All the code on this page comes from one runnable script,
`examples/xarray_dataset.py`, which executes top to bottom: the claims below
are its `assert` statements.

## CF attributes with reserved names

CF attribute keys are not all legal pydantic field names: `_FillValue`
starts with an underscore, which pydantic treats as private. Declare it as a
field alias on [`AttributesBase`][pydantic_zarr.core.AttributesBase], which
serializes by alias, accepts both spellings on input, and preserves unknown
keys. Optional keys are declared with the
[`pydantic.MISSING`](https://docs.pydantic.dev/latest/concepts/experimental/#missing-sentinel)
sentinel: a field left at `MISSING` stays out of the document, and because a
CF attribute is a string or absent — never `null` — `None` appears in no
annotation, so an explicit `None` is refused at validation:

```python
--8<-- "examples/xarray_dataset.py:attrs"
```

## A typed Dataset layout

The layout rules live in three `ArraySpec` / `GroupSpec` subclasses:

- `VariableSpec` narrows `dimension_names` to a required `tuple[str, ...]`:
  xarray always names every dimension. Auxiliary coordinates are plain
  `VariableSpec`s — nothing in the stored layout distinguishes them from
  data variables except the `coordinates` attribute that references them.
- `DimensionCoordinateSpec` is exactly 1-D with a single dimension name.
- `DatasetSpec` holds the rules an array cannot state about itself, because
  a spec never knows its own member name: every dimension name must have one
  consistent length across all members; a member named after a dimension it
  uses must be that dimension's 1-D coordinate; and every name in a member's
  CF `coordinates` attribute must resolve to a member.

A concrete dataset then fixes its member set with a closed `TypedDict`:
undeclared names are refused and missing members are required. Arrays are
built with [`ArraySpec.from_shape_and_data_type`][pydantic_zarr.v3.ArraySpec.from_shape_and_data_type],
which fills in the whole-shape chunk grid, the `/` chunk key encoding, and a
data-type-derived fill value; only the codec pipeline is spelled out, since
xarray appends a `zstd` codec to the default `bytes` codec. The dataset
round-trips through an in-memory store, and the three invalid layouts at the
end — a mismatched dimension length, an array named after a dimension it
uses without being its coordinate, and a dangling `coordinates` reference —
are each refused with a `ValidationError`:

```python
--8<-- "examples/xarray_dataset.py:dataset"
```

Reading a real xarray-written store back is the same `from_zarr` call on the
typed class: members are re-validated against the narrowed annotations, so
`ClimateDataset.from_zarr(zarr.open_group(...))` returns typed attributes
and members — or raises `ValidationError` if the store does not follow the
layout this page encodes.

## Fidelity against real xarray

The script ends by proving the model writes exactly what xarray writes: it
builds the same curvilinear `Dataset` with xarray (`assign_coords` makes
`lat`/`lon` auxiliary coordinates, so xarray itself emits the
`coordinates="lat lon"` attribute on the data variable), writes it with
`ds.to_zarr(..., zarr_format=3, consolidated=False)` — consolidated
metadata is disabled because pydantic-zarr does not model that block yet —
writes the model to a second store, and asserts that the two hierarchies
hold the same set of `zarr.json` documents and that each pair of documents
is equal once parsed. Chunk payloads are not compared: the specs describe
metadata.

```python
--8<-- "examples/xarray_dataset.py:reference"
```
