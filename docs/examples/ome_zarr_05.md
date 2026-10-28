# OME-Zarr 0.5

[OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) stores multiscale
bioimages as Zarr v3 hierarchies: a group whose attributes carry OME metadata
under the namespaced `ome` key, with the resolution levels as array members
and an optional `labels` subgroup. This example models that layout with
`pydantic-zarr`, showing four patterns: attribute models built on
`AttributesBase` (hyphenated spec keys via aliases, optional keys omitted
rather than written as `null`), a discriminated union for the spec's closed
set of coordinate transformations, typed members via a PEP 728 `TypedDict`
with `extra_items` (one fixed member name next to dynamically named
resolution levels), and cross-field validators that tie the metadata to the
actual members with `GroupSpec.get_path`.

The code on this page is the runnable script
[`examples/ome_zarr_05.py`](https://github.com/zarr-developers/pydantic-zarr/blob/main/examples/ome_zarr_05.py),
which carries the shared imports and runs top to bottom, asserting everything
it claims.

## Attribute models with spec spellings

The spec's `image-label` object uses keys that are not Python identifiers
(`label-value`) and marks most keys OPTIONAL: an absent key is meaningful,
a JSON `null` is not. `AttributesBase` handles both: aliases are used on
input and output, and an optional key is declared with the
[`pydantic.MISSING`](https://docs.pydantic.dev/latest/concepts/experimental/#missing-sentinel)
sentinel — a field left at `MISSING` stays out of the document, and since
`None` appears in no annotation, an explicit `None` is a validation error
rather than a stored `null`.

```python
--8<-- "examples/ome_zarr_05.py:attrs"
```

## A multiscale image group

The `ome` block of an image group holds `multiscales`: axes, and one dataset
entry per resolution level, each carrying `scale`/`translation`
transformations (the only types the spec allows there — a closed discriminated
union). The group model narrows `members` to a `TypedDict`: the fixed name
`labels` may hold a subgroup, and every other member must be an array
(`extra_items=ArraySpec`), so a stray subgroup is refused by name. Validators
then cross-check the metadata against the members: every dataset `path` must
resolve to an array whose dimensionality matches the axes. The hierarchy
round-trips through a Zarr store, and realistic invalid layouts — a dangling
dataset path, a stray subgroup member — are refused.

```python
--8<-- "examples/ome_zarr_05.py:image"
```

## Matching a reference implementation

The model is only as good as its agreement with what real OME-Zarr writers
produce. The script's final section generates the same image with
[ome-zarr-py](https://github.com/ome/ome-zarr-py) (the reference writer,
whose output decides the dataset paths `s0`/`s1`, the per-dataset
`scale` + `translation` transformations, and the codec choices), describes
that image with a `MultiscaleImage` instance, writes both to stores, and
asserts that every `zarr.json` document in the two stores is identical.

```python
--8<-- "examples/ome_zarr_05.py:reference"
```

A complete model of the layout would also type the `labels` group and its
label images (integer data types, `image-label` attributes as above, and the
rule that label paths listed in the labels group resolve to members), using
exactly the same ingredients: `AttributesBase` subclasses for the `ome`
blocks, narrowed `members`, and `get_path` in validators. The reference
comparison covers the multiscale image only: ome-zarr-py writes labels
through a separate API, so the labels model above is exercised model-side.
