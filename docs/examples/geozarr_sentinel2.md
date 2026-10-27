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

## CF attributes and arrays

CF and GeoZarr attribute keys are often not Python identifiers
(`_ARRAY_DIMENSIONS`, `proj:code`). On `AttributesBase`, a `Field(alias=...)`
is the whole story: both spellings validate, and serialization — including the
metadata written by `to_zarr` — uses the alias. `ArraySpec.from_shape_and_data_type` builds
the array metadata around the attributes; we pass an explicit `chunk_grid`
here to keep the example small, but when it is omitted the method guesses
size-appropriate chunks for real shapes (a 10980×10980 Sentinel-2 band gets
chunks of (687, 1373)).

```python
from pydantic import Field

from pydantic_zarr.core import AttributesBase
from pydantic_zarr.v3 import ArraySpec


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
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": (90, 90)}},
    attributes={"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref"},
    dimension_names=("y", "x"),
)
print(b02.attributes.model_dump())
"""
{
    '_ARRAY_DIMENSIONS': ('y', 'x'),
    'grid_mapping': 'spatial_ref',
    'standard_name': 'toa_bidirectional_reflectance',
    'units': '1',
}
"""

# the python spelling validates too, and compares equal
print(BandAttrs(array_dimensions=("y", "x"), grid_mapping="spatial_ref") == b02.attributes)
#> True
```

## A typed resolution dataset

A Sentinel-2 product fixes its member names: the 10 m dataset holds exactly
the four native 10 m bands plus the coordinate and CRS variables. A closed
`TypedDict` turns that into a validated contract — unknown names and missing
bands are refused. The shared `x`/`y`/`spatial_ref` trio lives in a non-closed
base that each closed resolution dict inherits. Cross-member rules go in
`model_validator`s on a `GroupSpec` subclass: `get_attribute` reads an
attribute key from member attributes whether they are a model or a plain
mapping, and `get_path` resolves a relative member path without raising.

```python
import zarr
import zarr.storage
from pydantic import Field, model_validator
from typing_extensions import Self, TypedDict

from pydantic_zarr.core import AttributesBase, get_attribute
from pydantic_zarr.v3 import ArraySpec, GroupSpec, is_array_spec


class BandAttrs(AttributesBase):
    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")
    grid_mapping: str
    standard_name: str = "toa_bidirectional_reflectance"
    units: str = "1"


class GridMappingAttrs(AttributesBase):
    """CF grid-mapping (CRS) variable attributes."""

    grid_mapping_name: str
    crs_wkt: str | None = None


class BandArray(ArraySpec[BandAttrs]): ...


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
        for name, member in (self.members or {}).items():
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
    members: R10mMembers


def band() -> BandArray:
    return BandArray.from_shape_and_data_type(
        (180, 180),
        "uint16",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (90, 90)}},
        attributes={"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref"},
        dimension_names=("y", "x"),
    )


def coord(axis: str) -> ArraySpec:
    return ArraySpec.from_shape_and_data_type(
        (180,),
        "float64",
        attributes={"_ARRAY_DIMENSIONS": [axis]},
        dimension_names=(axis,),
    )


r10m = R10mDataset(
    members={
        "b02": band(),
        "b03": band(),
        "b04": band(),
        "b08": band(),
        "x": coord("x"),
        "y": coord("y"),
        "spatial_ref": GridMappingArray.from_shape_and_data_type(
            (), "int32", attributes={"grid_mapping_name": "transverse_mercator"}
        ),
    }
)
print(sorted(r10m.members))
#> ['b02', 'b03', 'b04', 'b08', 'spatial_ref', 'x', 'y']
print(r10m["b02"].dimension_names)
#> ('y', 'x')

# Round trip: write the dataset to a store and read it back. `from_zarr` on
# the typed subclass re-validates everything it reads.
store = zarr.storage.MemoryStore()
r10m.to_zarr(store, path="measurements/r10m")
read_back = R10mDataset.from_zarr(zarr.open_group(store, path="measurements/r10m", mode="r"))
print(read_back.like(r10m))
#> True
# the stored attributes use the alias spellings
stored = zarr.open_group(store, path="measurements/r10m", mode="r")
print(dict(stored["b02"].attrs)["_ARRAY_DIMENSIONS"])
#> ['y', 'x']
```

## Invalid layouts are refused

The closed `TypedDict` refuses member names outside the spec, and the
cross-member validator refuses a `grid_mapping` reference that does not
resolve. (Full pydantic error messages are long; here we print one stable
line per refusal.)

```python
from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self, TypedDict

from pydantic_zarr.core import AttributesBase, get_attribute
from pydantic_zarr.v3 import ArraySpec, GroupSpec, is_array_spec


class BandAttrs(AttributesBase):
    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")
    grid_mapping: str


class BandArray(ArraySpec[BandAttrs]): ...


class R10mMembers(TypedDict, closed=True):
    b02: BandArray
    spatial_ref: ArraySpec


class GeoZarrDataset(GroupSpec):
    @model_validator(mode="after")
    def check_grid_mapping_refs(self) -> Self:
        for name, member in (self.members or {}).items():
            if not is_array_spec(member):
                continue
            ref = get_attribute(member.attributes, "grid_mapping")
            if isinstance(ref, str) and self.get_path(ref) is None:
                raise ValueError(f"{name!r} declares grid_mapping={ref!r}: no such member")
        return self


class R10mDataset(GeoZarrDataset):
    members: R10mMembers


def band(grid_mapping: str) -> BandArray:
    return BandArray.from_shape_and_data_type(
        (180, 180),
        "uint16",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (90, 90)}},
        attributes={"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": grid_mapping},
        dimension_names=("y", "x"),
    )


crs = ArraySpec.from_shape_and_data_type((), "int32", attributes={"grid_mapping_name": "transverse_mercator"})

# b05 is a 20 m band: it does not belong in the 10 m dataset
try:
    R10mDataset(members={"b02": band("spatial_ref"), "b05": band("spatial_ref"), "spatial_ref": crs})
except ValidationError as e:
    print(f"refused: {e.errors()[0]['type']} for member {e.errors()[0]['loc'][-1]!r}")
    #> refused: extra_forbidden for member 'b05'

# a grid_mapping reference must name a member of the same dataset
try:
    R10mDataset(members={"b02": band("crs"), "spatial_ref": crs})
except ValidationError as e:
    print(f"refused: {e.errors()[0]['msg']}")
    #> refused: Value error, 'b02' declares grid_mapping='crs': no such member
```

The full version of this model — all three resolution datasets, the quality
masks, the multiscales layout with per-level `spatial:shape` checks, and
coordinate-length validation — follows exactly these patterns, scaled up.
