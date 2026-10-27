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
base64 of the value's bytes (four bytes here, for a `float32` variable). This page builds a typed model of that layout: CF attribute
models with reserved-name aliases, `ArraySpec` subclasses for variables and
coordinates, a cross-member dimension-consistency validator, and a closed
member set for one concrete dataset. The same model classes validate real
xarray-written stores read back with `from_zarr`. (xarray also writes a
`consolidated_metadata` block into the root `zarr.json` by default;
pydantic-zarr does not model consolidated metadata yet.)

## CF attributes with reserved names

CF attribute keys are not all legal pydantic field names: `_FillValue`
starts with an underscore, which pydantic treats as private. Declare it as a
field alias on [`AttributesBase`][pydantic_zarr.core.AttributesBase], which
serializes by alias, accepts both spellings on input, preserves unknown
keys, and omits optional fields that were left unset:

```python
from pydantic import Field

from pydantic_zarr.core import AttributesBase


class VariableAttrs(AttributesBase):
    """CF attributes of a netCDF-style variable."""

    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    calendar: str | None = None
    FillValue: str | float | None = Field(default=None, alias="_FillValue")


stored = {
    "units": "degC",
    "standard_name": "air_temperature",
    "_FillValue": "ADwcxg==",
    "comment": "unknown keys are preserved",
}
attrs = VariableAttrs.model_validate(stored)
print(attrs.FillValue)
#> ADwcxg==
print(attrs.model_dump() == stored)
#> True
print(VariableAttrs(units="degrees_north").model_dump())
#> {'units': 'degrees_north'}
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
xarray appends a `zstd` codec to the default `bytes` codec.

```python
from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self, TypedDict
from zarr.storage import MemoryStore

from pydantic_zarr.core import AttributesBase, get_attribute
from pydantic_zarr.v3 import ArraySpec, GroupSpec, is_array_spec


class CFAttrs(AttributesBase):
    """CF attributes (in a real model, one subclass per variable family)."""

    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    calendar: str | None = None
    coordinates: str | None = None
    FillValue: str | float | None = Field(default=None, alias="_FillValue")


class VariableSpec(ArraySpec[CFAttrs]):
    """A netCDF-style variable: every dimension must be named."""

    dimension_names: tuple[str, ...]


class DimensionCoordinateSpec(VariableSpec):
    """A netCDF dimension coordinate: exactly 1-D, named after its dimension."""

    shape: tuple[int]
    dimension_names: tuple[str]


class DatasetSpec(GroupSpec):
    """A Dataset-like group: members with consistent dimensions."""

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if self.members is None:  # members unknown: nothing to check
            return self
        sizes: dict[str, int] = {}
        for name, member in self.members.items():
            if not is_array_spec(member) or member.dimension_names is None:
                continue
            for dim, size in zip(member.dimension_names, member.shape, strict=True):
                if sizes.setdefault(dim, size) != size:
                    msg = (
                        f"Dimension {dim!r} has length {sizes[dim]} elsewhere, "
                        f"but length {size} on {name!r}."
                    )
                    raise ValueError(msg)
            if name in member.dimension_names and member.dimension_names != (name,):
                msg = (
                    f"{name!r} is named after a dimension it uses, so it must be "
                    f"that dimension's 1-D coordinate; got dimensions "
                    f"{member.dimension_names}."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_coordinates_attribute(self) -> Self:
        for name, member in (self.members or {}).items():
            if not is_array_spec(member):
                continue
            listed = get_attribute(member.attributes, "coordinates")
            if isinstance(listed, str):
                for coordinate in listed.split():
                    if self.members is None or coordinate not in self.members:
                        msg = (
                            f"{name!r} lists coordinate {coordinate!r}, "
                            "which is not a member."
                        )
                        raise ValueError(msg)
        return self


class ClimateMembers(TypedDict, closed=True):
    time: DimensionCoordinateSpec
    lat: VariableSpec  # 2-D auxiliary coordinate over (y, x)
    lon: VariableSpec  # 2-D auxiliary coordinate over (y, x)
    temperature: VariableSpec


class ClimateDataset(DatasetSpec):
    members: ClimateMembers


# xarray's default Zarr V3 codec pipeline
CODECS = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
)

# a curvilinear grid: the data dimensions are (time, y, x), and latitude and
# longitude are 2-D auxiliary coordinates over (y, x)
dataset = ClimateDataset(
    attributes={"Conventions": "CF-1.8", "title": "Example climate dataset"},
    members={
        "time": DimensionCoordinateSpec.from_shape_and_data_type(
            (4,),
            "int64",
            dimension_names=("time",),
            codecs=CODECS,
            attributes=CFAttrs(
                units="days since 2024-01-01", calendar="proleptic_gregorian"
            ),
        ),
        "lat": VariableSpec.from_shape_and_data_type(
            (3, 5),
            "float64",
            dimension_names=("y", "x"),
            codecs=CODECS,
            attributes=CFAttrs(units="degrees_north", standard_name="latitude"),
        ),
        "lon": VariableSpec.from_shape_and_data_type(
            (3, 5),
            "float64",
            dimension_names=("y", "x"),
            codecs=CODECS,
            attributes=CFAttrs(units="degrees_east", standard_name="longitude"),
        ),
        "temperature": VariableSpec.from_shape_and_data_type(
            (4, 3, 5),
            "float32",
            dimension_names=("time", "y", "x"),
            codecs=CODECS,
            attributes=CFAttrs(
                units="degC",
                standard_name="air_temperature",
                coordinates="lat lon",
                FillValue="ADwcxg==",
            ),
        ),
    },
)
temperature = dataset.members["temperature"]
print(temperature.dimension_names)
#> ('time', 'y', 'x')
print(temperature.fill_value, temperature.attributes.FillValue)
#> NaN ADwcxg==

# round trip through an in-memory Zarr V3 store
store = MemoryStore()
group = dataset.to_zarr(store, path="")
back = ClimateDataset.from_zarr(group)
print(back.like(dataset), back == dataset)
#> True True

# a dimension length that disagrees between members is refused
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {
            "temperature": VariableSpec.from_shape_and_data_type(
                (4, 2, 5),
                "float32",
                dimension_names=("time", "y", "x"),
                codecs=CODECS,
            )
        },
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    """
    Value error, Dimension 'y' has length 3 elsewhere, but length 2 on 'temperature'.
    """

# an array named after a dimension it uses must be that dimension's
# 1-D coordinate
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {
            "lat": VariableSpec.from_shape_and_data_type(
                (3, 5),
                "float64",
                dimension_names=("lat", "x"),
                codecs=CODECS,
            )
        },
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    """
    Value error, 'lat' is named after a dimension it uses, so it must be that dimension's 1-D coordinate; got dimensions ('lat', 'x').
    """

# a `coordinates` attribute referencing a missing member is refused
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {
            "temperature": temperature.update(
                attributes=CFAttrs(coordinates="lat lon elevation")
            )
        },
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    #> Value error, 'temperature' lists coordinate 'elevation', which is not a member.
```

Reading a real xarray-written store back is the same `from_zarr` call on the
typed class: members are re-validated against the narrowed annotations, so
`ClimateDataset.from_zarr(zarr.open_group(...))` returns typed attributes
and members — or raises `ValidationError` if the store does not follow the
layout this page encodes.
