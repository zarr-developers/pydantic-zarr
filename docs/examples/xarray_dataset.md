# xarray-style datasets

When [xarray](https://xarray.dev) writes a netCDF-style `Dataset` with
`Dataset.to_zarr(..., zarr_format=3)`, it produces a flat layout: a root
group whose attributes are the dataset's global attributes (`Conventions`,
`title`, ...), with one array member per data variable and per coordinate.
Dimensions use the native Zarr V3 `dimension_names` field, and each
*coordinate variable* is a 1-D array named after its own dimension (the
netCDF convention). Every array carries CF attributes such as `units`,
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
  xarray always names every dimension.
- `CoordinateSpec` is exactly 1-D with a single dimension name.
- `DatasetSpec` holds the rules an array cannot state about itself, because
  a spec never knows its own member name: every dimension name used by any
  member must belong to a *coordinate member* (a 1-D member whose single
  dimension name is its own name), and member shapes must agree with
  coordinate lengths.

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

from pydantic_zarr.core import AttributesBase
from pydantic_zarr.v3 import ArraySpec, GroupSpec, is_array_spec


class CFAttrs(AttributesBase):
    """CF attributes (in a real model, one subclass per variable family)."""

    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    calendar: str | None = None
    FillValue: str | float | None = Field(default=None, alias="_FillValue")


class VariableSpec(ArraySpec[CFAttrs]):
    """A netCDF-style variable: every dimension must be named."""

    dimension_names: tuple[str, ...]


class CoordinateSpec(VariableSpec):
    """A netCDF coordinate variable: exactly 1-D."""

    shape: tuple[int]
    dimension_names: tuple[str]


class DatasetSpec(GroupSpec):
    """A Dataset-like group: members with consistent dimensions."""

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if self.members is None:  # members unknown: nothing to check
            return self
        coord_sizes = {
            name: member.shape[0]
            for name, member in self.members.items()
            if is_array_spec(member) and member.dimension_names == (name,)
        }
        for name, member in self.members.items():
            if not is_array_spec(member) or member.dimension_names is None:
                continue
            for dim, size in zip(member.dimension_names, member.shape, strict=True):
                if dim not in coord_sizes:
                    msg = f"Dimension {dim!r} of {name!r} has no coordinate array."
                    raise ValueError(msg)
                if coord_sizes[dim] != size:
                    msg = (
                        f"{name!r} has length {size} along {dim!r}, but the "
                        f"coordinate array has length {coord_sizes[dim]}."
                    )
                    raise ValueError(msg)
        return self


class ClimateMembers(TypedDict, closed=True):
    time: CoordinateSpec
    lat: CoordinateSpec
    lon: CoordinateSpec
    temperature: VariableSpec


class ClimateDataset(DatasetSpec):
    members: ClimateMembers


# xarray's default Zarr V3 codec pipeline
CODECS = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
)

dataset = ClimateDataset(
    attributes={"Conventions": "CF-1.8", "title": "Example climate dataset"},
    members={
        "time": CoordinateSpec.from_shape_and_data_type(
            (4,),
            "int64",
            dimension_names=("time",),
            codecs=CODECS,
            attributes=CFAttrs(
                units="days since 2024-01-01", calendar="proleptic_gregorian"
            ),
        ),
        "lat": CoordinateSpec.from_shape_and_data_type(
            (3,),
            "float64",
            dimension_names=("lat",),
            codecs=CODECS,
            attributes=CFAttrs(units="degrees_north", standard_name="latitude"),
        ),
        "lon": CoordinateSpec.from_shape_and_data_type(
            (5,),
            "float64",
            dimension_names=("lon",),
            codecs=CODECS,
            attributes=CFAttrs(units="degrees_east", standard_name="longitude"),
        ),
        "temperature": VariableSpec.from_shape_and_data_type(
            (4, 3, 5),
            "float32",
            dimension_names=("time", "lat", "lon"),
            codecs=CODECS,
            attributes=CFAttrs(
                units="degC",
                standard_name="air_temperature",
                FillValue="ADwcxg==",
            ),
        ),
    },
)
temperature = dataset.members["temperature"]
print(temperature.dimension_names)
#> ('time', 'lat', 'lon')
print(temperature.fill_value, temperature.attributes.FillValue)
#> NaN ADwcxg==

# round trip through an in-memory Zarr V3 store
store = MemoryStore()
group = dataset.to_zarr(store, path="")
back = ClimateDataset.from_zarr(group)
print(back.like(dataset), back == dataset)
#> True True

# a dimension name with no corresponding coordinate array is refused
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {
            "temperature": temperature.update(
                dimension_names=("time", "latitude", "lon")
            )
        },
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    #> Value error, Dimension 'latitude' of 'temperature' has no coordinate array.

# a variable shape that disagrees with a coordinate length is refused
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {
            "temperature": VariableSpec.from_shape_and_data_type(
                (4, 2, 5),
                "float32",
                dimension_names=("time", "lat", "lon"),
                codecs=CODECS,
            )
        },
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    """
    Value error, 'temperature' has length 2 along 'lat', but the coordinate array has length 3.
    """
```

Reading a real xarray-written store back is the same `from_zarr` call on the
typed class: members are re-validated against the narrowed annotations, so
`ClimateDataset.from_zarr(zarr.open_group(...))` returns typed attributes
and members — or raises `ValidationError` if the store does not follow the
layout this page encodes.
