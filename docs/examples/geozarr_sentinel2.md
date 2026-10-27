# GeoZarr: a Sentinel-2 product

[GeoZarr](https://github.com/zarr-developers/geozarr-spec) stores georeferenced
rasters in Zarr using CF-style metadata: data variables carry `standard_name`,
`units`, a `grid_mapping` attribute naming a CRS variable, and an
`_ARRAY_DIMENSIONS` attribute that must agree with the Zarr V3
`dimension_names`. This page models a slice of a Sentinel-2 Level-2A product —
a resolution dataset with a fixed set of spectral bands — and shows three
patterns: attribute models with non-identifier keys on `AttributesBase`,
closed-`TypedDict` members for hierarchies whose member names are fixed by a
spec, and cross-member validation (`get_attribute`, `get_path`) for rules like
"every `grid_mapping` reference resolves".

All the code on this page is one runnable script,
[`examples/geozarr_sentinel2.py`](https://github.com/zarr-developers/pydantic-zarr/blob/main/examples/geozarr_sentinel2.py),
which executes top to bottom and asserts everything the page claims.

## CF attributes and arrays

CF and GeoZarr attribute keys are often not Python identifiers
(`_ARRAY_DIMENSIONS`, `proj:code`). On `AttributesBase`, a `Field(alias=...)`
is the whole story: both spellings validate, and serialization — including the
metadata written by `to_zarr` — uses the alias.
`ArraySpec.from_shape_and_data_type` builds the rest of the array metadata
around the attributes: the chunk grid is size-guessed with the same heuristic
zarr-python applies when no chunks are given (a 10980×10980 Sentinel-2 band
gets chunks of (687, 1373)), and the fill value and codecs are derived from
the data type.

```python
--8<-- "examples/geozarr_sentinel2.py:attrs"
```

## A typed resolution dataset

A Sentinel-2 product fixes its member names: the 10 m dataset holds exactly
the four native 10 m bands plus the coordinate and CRS variables. A closed
`TypedDict` turns that into a validated contract — unknown names and missing
bands are refused. The shared `x`/`y`/`spatial_ref` trio lives in a non-closed
base that each closed resolution dict inherits. Cross-member rules go in
`model_validator`s on a `GroupSpec` subclass: `get_attribute` reads an
attribute key from member attributes whether they are a model or a plain
mapping, and `get_path` resolves a relative member path without raising. The
section ends with a store round trip — `from_zarr` on the typed subclass
re-validates everything it reads — and checks that the stored attribute
documents carry the alias spellings.

```python
--8<-- "examples/geozarr_sentinel2.py:dataset"
```

## Invalid layouts are refused

The closed `TypedDict` refuses member names outside the spec, and the
cross-member validator refuses a `grid_mapping` reference that does not
resolve.

```python
--8<-- "examples/geozarr_sentinel2.py:refusals"
```

## Checking against a reference writer

GeoZarr has no single canonical writer to compare against, so the script
checks the model against the nearest reference for this layer: xarray writing
the equivalent CF dataset (one resolution dataset with `grid_mapping` /
`spatial_ref` and 1-D `x`/`y` coordinates — the CF layer, not the full
Sentinel-2 product tree). The reference dataset and the model go to two
separate stores, and every `zarr.json` is compared as parsed JSON: same
documents, same keys, same values. The only accommodation is telling the model
which codecs the reference writer uses.

```python
--8<-- "examples/geozarr_sentinel2.py:reference"
```

The full version of this model — all three resolution datasets, the quality
masks, the multiscales layout with per-level `spatial:shape` checks, and
coordinate-length validation — follows exactly these patterns, scaled up.
