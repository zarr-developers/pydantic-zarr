# OME-Zarr 0.6

[OME-Zarr 0.6](https://ngff.openmicroscopy.org/specifications/0.6/) stores a
multiscale image as a Zarr group whose `ome` attributes declare named
coordinate systems, coordinate transformations, and the resolution-level
arrays those transformations describe. This page models a trimmed version of
that layout and shows four `pydantic-zarr` patterns: spec-style attribute
models on `AttributesBase` (absent-vs-null handled by the `explicit_none`
policy), a discriminated union of transformation types with cross-field
validators, a `GroupSpec` subclass that checks its metadata against its
members with `get_path`, and a store round trip with `to_zarr` / `from_zarr`.
The code on this page is the runnable script
[`examples/ome_zarr_06.py`](https://github.com/zarr-developers/pydantic-zarr/blob/main/examples/ome_zarr_06.py),
included section by section; because both sections share one namespace, the
image section builds directly on the transformation classes instead of
redeclaring them, and its expectations are asserts the script checks on every
run.

## Coordinate systems and transformations

A coordinate system is a named list of axes; a transformation maps points
between two systems it references by `name` and/or `path`. OME-Zarr 0.6
treats every optional key as *absent, never null*, which is exactly
`explicit_none = "refuse"` on a shared base class — with one deliberate
exception: the spec says a transformation reference's `path` "MAY be omitted
or null", so `CoordinateReference` keeps the default `"keep"` policy and an
explicit `"path": null` survives as written. The sequence transformation
shows a cross-field rule: its scale and translation members must agree on
dimensionality.

```python
--8<-- "examples/ome_zarr_06.py:transforms"
```

## The image group

An image is a group whose `ome.multiscales` metadata must agree with the
group's actual members: every `datasets[*].path` resolves (via `get_path`) to
an array whose dimensionality matches the coordinate systems, every level
maps into the same intrinsic system, and the scale vector length equals the
image dimensionality. The resolution-level arrays are metadata-only
`ArraySpec.from_shape_and_data_type` documents, the hierarchy round-trips
through an in-memory Zarr v3 store, and invalid layouts are refused at
construction. The `Scale` and `SequenceTransform` classes from the previous
section are reused directly — their `input`/`output` are optional there
because wrapper transforms omit them, so `Dataset` checks they are present
and restricts each level to a single scale or a sequence of one scale
followed by one translation, as the spec does. (The full spec also allows
`identity` per dataset and defines eight more transformation types this
trimmed model leaves out.)

```python
--8<-- "examples/ome_zarr_06.py:image"
```

## Reproducing a reference writer's metadata

[ngff-zarr](https://pypi.org/project/ngff-zarr/) (0.50+) is, at the time of
writing, the installable implementation that writes OME-Zarr 0.6, so the
script's final section uses it as ground truth rather than falling back to
the spec repository's example documents: it converts synthetic pixels into a
two-level 0.6 image with `to_ngff_zarr(..., version="0.6")`, rebuilds the
same image with the models above (hand-typed structure; only the free-form
provenance values — `"@type"`, `name`, `type`, `metadata` — are adopted from
the reference), writes it with `to_zarr`, and asserts that every `zarr.json`
in the two stores parses to identical JSON. Note the nested `scale0/image`
dataset paths, which `get_path` resolves through the intermediate groups.
The section skips, loudly, when ngff-zarr is not installed.

```python
--8<-- "examples/ome_zarr_06.py:reference"
```
