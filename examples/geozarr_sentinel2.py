"""A GeoZarr-style model of a Sentinel-2 L2A resolution dataset.

Shows the patterns for modeling a metadata convention with pydantic-zarr:
attribute models with non-identifier keys (``_ARRAY_DIMENSIONS``) on
``AttributesBase``, closed-``TypedDict`` members for a hierarchy whose member
names are fixed by a spec (the four native 10 m Sentinel-2 bands), a
cross-member validator (every CF ``grid_mapping`` reference resolves), a store
round trip, the invalid layouts the model refuses, and a check that the model
writes the same metadata documents as xarray writing the equivalent CF
dataset. GeoZarr: https://github.com/zarr-developers/geozarr-spec

Run standalone with:
uv run --no-project --with-editable <path to pydantic-zarr checkout> \
    --with "zarr>=3.3" --with xarray --python 3.12 -- python examples/geozarr_sentinel2.py
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-zarr", "zarr", "xarray"]
# ///
# NOTE: needs the pydantic-zarr release carrying PR #183
# (zarr-metadata-backed v3 document validation and the APIs used below).

# --8<-- [start:attrs]
from typing import Self

import zarr
import zarr.storage
from pydantic import Field, ValidationError, model_validator
from typing_extensions import TypedDict

from pydantic_zarr.core import MISSING, AttributesBase, get_attribute
from pydantic_zarr.v3 import ArraySpec, CodecLike, GroupSpec, is_array_spec


class BandAttrs(AttributesBase):
    """CF attributes of a reflectance band (GeoZarr data variable)."""

    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")
    grid_mapping: str
    standard_name: str = "toa_bidirectional_reflectance"
    units: str = "1"


class BandArray(ArraySpec[BandAttrs]):
    """A Sentinel-2 band: uint16 digital numbers on the (y, x) grid."""


b02 = BandArray.from_shape_and_data_type(
    (180, 180),
    "uint16",
    attributes={"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref"},
    dimension_names=("y", "x"),
)
# serialization uses the alias spelling ...
assert b02.attributes.model_dump() == {
    "_ARRAY_DIMENSIONS": ("y", "x"),
    "grid_mapping": "spatial_ref",
    "standard_name": "toa_bidirectional_reflectance",
    "units": "1",
}
# ... and the python spelling validates too, to an equal model
assert BandAttrs(array_dimensions=("y", "x"), grid_mapping="spatial_ref") == b02.attributes
# --8<-- [end:attrs]


# --8<-- [start:dataset]
class GridMappingAttrs(AttributesBase):
    """CF grid-mapping (CRS) variable attributes."""

    grid_mapping_name: str
    crs_wkt: str | MISSING = MISSING


class GridMappingArray(ArraySpec[GridMappingAttrs]): ...


class GriddedMembers(TypedDict):
    """Members every resolution dataset shares (non-closed base)."""

    x: ArraySpec
    y: ArraySpec
    spatial_ref: GridMappingArray


class R10mMembers(GriddedMembers, closed=True):
    """The 10 m dataset: exactly the four native 10 m bands."""

    b02: BandArray
    b03: BandArray
    b04: BandArray
    b08: BandArray


class GeoZarrDataset(GroupSpec):
    """A group of data variables over one grid, with GeoZarr rules."""

    @model_validator(mode="after")
    def check_grid_mapping_refs(self) -> Self:
        """Every grid_mapping attribute resolves to a grid-mapping member."""
        for name, member in self.members.items():
            if not is_array_spec(member):
                continue
            ref = get_attribute(member.attributes, "grid_mapping")
            if not isinstance(ref, str):
                continue
            target = self.get_path(ref)
            if target is None or not is_array_spec(target):
                raise ValueError(f"{name!r} declares grid_mapping={ref!r}: no such member")
            if get_attribute(target.attributes, "grid_mapping_name") is None:
                raise ValueError(f"{ref!r} is not a grid-mapping variable")
        return self


class R10mDataset(GeoZarrDataset):
    # sound at runtime: pydantic validates members against the TypedDict;
    # the ignore is for checkers that treat the field override as unsound
    members: R10mMembers  # type: ignore[assignment]


def band(grid_mapping: str = "spatial_ref") -> BandArray:
    return BandArray.from_shape_and_data_type(
        (180, 180),
        "uint16",
        attributes={"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": grid_mapping},
        dimension_names=("y", "x"),
    )


def coord(axis: str) -> ArraySpec:
    # plain-dict attributes work too; tuples match what a store read returns
    return ArraySpec.from_shape_and_data_type(
        (180,),
        "float64",
        attributes={"_ARRAY_DIMENSIONS": (axis,)},
        dimension_names=(axis,),
    )


crs = GridMappingArray.from_shape_and_data_type(
    (), "int32", attributes={"grid_mapping_name": "transverse_mercator"}
)
r10m = R10mDataset(
    members={
        "b02": band(),
        "b03": band(),
        "b04": band(),
        "b08": band(),
        "x": coord("x"),
        "y": coord("y"),
        "spatial_ref": crs,
    }
)
assert sorted(r10m.members) == ["b02", "b03", "b04", "b08", "spatial_ref", "x", "y"]
b02 = r10m["b02"]
assert is_array_spec(b02)
assert b02.dimension_names == ("y", "x")

# Round trip: write the dataset to a store and read it back. `from_zarr` on
# the typed subclass re-validates everything it reads.
store = zarr.storage.MemoryStore()
r10m.to_zarr(store, path="measurements/r10m")
read_back = R10mDataset.from_zarr(zarr.open_group(store, path="measurements/r10m", mode="r"))
assert read_back.like(r10m)
assert read_back == r10m
print("round trip through a MemoryStore preserved the dataset")

# the stored attribute documents use the alias spellings
stored = zarr.open_group(store, path="measurements/r10m", mode="r")
assert dict(stored["b02"].attrs)["_ARRAY_DIMENSIONS"] == ["y", "x"]
# --8<-- [end:dataset]


# --8<-- [start:refusals]
# b05 is a 20 m band: it does not belong in the 10 m dataset
try:
    R10mDataset(members={**dict(r10m.members), "b05": band()})
except ValidationError as e:
    assert "Extra inputs are not permitted" in e.errors()[0]["msg"]
    print("refused: b05 is not a member of the 10 m dataset")
else:
    raise AssertionError("expected refusal")

# a grid_mapping reference must name a member of the same dataset
try:
    R10mDataset(members={**dict(r10m.members), "b02": band(grid_mapping="crs")})
except ValidationError as e:
    assert "declares grid_mapping='crs'" in e.errors()[0]["msg"]
    print("refused: grid_mapping='crs' does not resolve")
else:
    raise AssertionError("expected refusal")
# --8<-- [end:refusals]


# --8<-- [start:reference]
# Reference check: xarray writing the equivalent CF dataset is the nearest
# reference implementation for this layer (one resolution dataset with
# grid_mapping/spatial_ref and 1-D x/y coords — the CF layer, not the full
# Sentinel-2 product tree). The model, told the reference's codecs, must
# write byte-for-byte the same metadata documents.
import asyncio
import json

import numpy as np
import xarray as xr

# what zarr-python (and so xarray) writes by default: bytes + zstd
REF_CODECS: tuple[CodecLike, ...] = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
)
N = 18
BAND_ATTRS = {
    "_ARRAY_DIMENSIONS": ["y", "x"],
    "grid_mapping": "spatial_ref",
    "standard_name": "toa_bidirectional_reflectance",
    "units": "1",
}

reference = xr.Dataset(
    data_vars={
        **{
            name: (("y", "x"), np.zeros((N, N), dtype="uint16"), dict(BAND_ATTRS))
            for name in ("b02", "b03", "b04", "b08")
        },
        "spatial_ref": ((), np.int32(0), {"grid_mapping_name": "transverse_mercator"}),
    },
    coords={
        "x": ("x", np.arange(N, dtype="float64"), {"_ARRAY_DIMENSIONS": ["x"]}),
        "y": ("y", np.arange(N, dtype="float64"), {"_ARRAY_DIMENSIONS": ["y"]}),
    },
)
ref_store = zarr.storage.MemoryStore()
# suppress xarray's CF _FillValue attribute on the float coordinates; the
# zarr-level fill_value ("NaN") is still written
reference.to_zarr(  # type: ignore[no-matching-overload]
    ref_store,
    zarr_format=3,
    consolidated=False,
    encoding={"x": {"_FillValue": None}, "y": {"_FillValue": None}},
)


def ref_band() -> BandArray:
    return BandArray.from_shape_and_data_type(
        (N, N), "uint16", attributes=dict(BAND_ATTRS), dimension_names=("y", "x"), codecs=REF_CODECS
    )


def ref_coord(axis: str) -> ArraySpec:
    return ArraySpec.from_shape_and_data_type(
        (N,),
        "float64",
        attributes={"_ARRAY_DIMENSIONS": [axis]},
        dimension_names=(axis,),
        codecs=REF_CODECS,
    )


model = R10mDataset(
    members={
        "b02": ref_band(),
        "b03": ref_band(),
        "b04": ref_band(),
        "b08": ref_band(),
        "x": ref_coord("x"),
        "y": ref_coord("y"),
        "spatial_ref": GridMappingArray.from_shape_and_data_type(
            (), "int32", attributes={"grid_mapping_name": "transverse_mercator"}, codecs=REF_CODECS
        ),
    }
)
model_store = zarr.storage.MemoryStore()
model.to_zarr(model_store, path="")


def metadata_documents(store: zarr.storage.MemoryStore) -> dict[str, dict[str, object]]:
    """Every zarr.json in the store, parsed (chunk payloads ignored)."""

    async def keys(s: zarr.storage.MemoryStore) -> list[str]:
        return [k async for k in s.list()]

    proto = zarr.core.buffer.default_buffer_prototype()
    documents: dict[str, dict[str, object]] = {}
    for key in asyncio.run(keys(store)):
        if not key.endswith("zarr.json"):
            continue
        buffer = asyncio.run(store.get(key, prototype=proto))
        assert buffer is not None
        documents[key] = json.loads(buffer.to_bytes())
    return documents


ref_docs = metadata_documents(ref_store)
model_docs = metadata_documents(model_store)
assert ref_docs.keys() == model_docs.keys()
for key in ref_docs:
    assert ref_docs[key].keys() == model_docs[key].keys(), key
    for field in ref_docs[key]:
        assert ref_docs[key][field] == model_docs[key][field], (key, field)
print("model output matches the xarray-written reference, document for document")
# --8<-- [end:reference]

print("All GeoZarr Sentinel-2 checks passed.")
