"""
A typed model of the Zarr V3 layout xarray writes for a netCDF-style Dataset:
CF attribute models with reserved-name aliases, ArraySpec subclasses for
variables and dimension coordinates, a cross-member dimension-consistency
validator with 2-D auxiliary coordinates on a curvilinear grid, a closed
member set for one concrete dataset, a store round trip, the invalid
layouts the model refuses, and a document-for-document fidelity check
against a store written by real xarray. This script is the source of the
fenced code on the docs page ``docs/examples/xarray_dataset.md``. It needs
an environment with xarray; from a checkout, run it as::

    uv run --no-project --with-editable . --with xarray \
        --with 'zarr>=3.1' --python 3.12 -- python examples/xarray_dataset.py
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-zarr", "zarr", "xarray"]
# ///
# Needs the pydantic-zarr release carrying PR #183
# (document-level Zarr V3 metadata validation).

# --8<-- [start:attrs]
from pydantic import Field, ValidationError

from pydantic_zarr.core import MISSING, AttributesBase


class VariableAttrs(AttributesBase):
    """CF attributes of a netCDF-style variable."""

    units: str | MISSING = MISSING
    standard_name: str | MISSING = MISSING
    long_name: str | MISSING = MISSING
    calendar: str | MISSING = MISSING
    FillValue: str | float | MISSING = Field(default=MISSING, alias="_FillValue")


stored = {
    "units": "degC",
    "standard_name": "air_temperature",
    "_FillValue": "AAAAAICHw8A=",
    "comment": "unknown keys are preserved",
}
attrs = VariableAttrs.model_validate(stored)
assert attrs.FillValue == "AAAAAICHw8A="
assert attrs.long_name is MISSING
assert attrs.model_dump() == stored
assert VariableAttrs(units="degrees_north").model_dump() == {"units": "degrees_north"}

# a CF attribute is a string or absent, never null: `None` is in no
# annotation, so an explicit None is a validation error, not a stored null
try:
    VariableAttrs(long_name=None)
except ValidationError as exc:
    assert exc.errors()[0]["type"] == "string_type"
else:
    raise AssertionError("expected a null long_name to be refused")
# --8<-- [end:attrs]

print("CF attributes: alias round trip and unset-key omission hold")

# --8<-- [start:dataset]
from typing import Self

from pydantic import Field, ValidationError, model_validator
from typing_extensions import TypedDict
from zarr.storage import MemoryStore

from pydantic_zarr.core import MISSING, AttributesBase, get_attribute
from pydantic_zarr.v3 import ArraySpec, CodecLike, GroupSpec, is_array_spec


class CFAttrs(AttributesBase):
    """CF attributes (in a real model, one subclass per variable family)."""

    units: str | MISSING = MISSING
    standard_name: str | MISSING = MISSING
    long_name: str | MISSING = MISSING
    calendar: str | MISSING = MISSING
    coordinates: str | MISSING = MISSING
    FillValue: str | float | MISSING = Field(default=MISSING, alias="_FillValue")


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
        sizes: dict[str, int] = {}
        for name, member in self.members.items():
            if not is_array_spec(member) or member.dimension_names is None:
                continue
            for dim, size in zip(member.dimension_names, member.shape, strict=True):
                if dim is None:  # unnamed dimensions have no coordinates
                    continue
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
        for name, member in self.members.items():
            if not is_array_spec(member):
                continue
            listed = get_attribute(member.attributes, "coordinates")
            if isinstance(listed, str):
                for coordinate in listed.split():
                    if coordinate not in self.members:
                        msg = f"{name!r} lists coordinate {coordinate!r}, which is not a member."
                        raise ValueError(msg)
        return self


class ClimateMembers(TypedDict, closed=True):
    time: DimensionCoordinateSpec
    lat: VariableSpec  # 2-D auxiliary coordinate over (y, x)
    lon: VariableSpec  # 2-D auxiliary coordinate over (y, x)
    temperature: VariableSpec


class ClimateDataset(DatasetSpec):
    # sound at runtime: pydantic validates members against the TypedDict;
    # the ignore is for checkers that treat the field override as unsound
    members: ClimateMembers  # type: ignore[assignment]


# xarray's default Zarr V3 codec pipeline
CODECS: tuple[CodecLike, ...] = (
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
            attributes=CFAttrs(units="days since 2024-01-01", calendar="proleptic_gregorian"),
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
                FillValue="AAAAAICHw8A=",
            ),
        ),
    },
)
temperature = dataset.members["temperature"]
assert temperature.dimension_names == ("time", "y", "x")
assert temperature.fill_value == "NaN"
assert temperature.attributes.FillValue == "AAAAAICHw8A="

# round trip through an in-memory Zarr V3 store
store = MemoryStore()
group = dataset.to_zarr(store, path="")
back = ClimateDataset.from_zarr(group)
assert back.like(dataset)
assert back == dataset
print("store round trip: from_zarr(to_zarr(dataset)) is the same model")

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
    raise AssertionError("accepted a mismatched dimension length")
except ValidationError as exc:
    assert "Dimension 'y' has length 3 elsewhere" in exc.errors()[0]["msg"]

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
    raise AssertionError("accepted a non-coordinate array named after a dimension")
except ValidationError as exc:
    assert "'lat' is named after a dimension it uses" in exc.errors()[0]["msg"]

# unnamed dimensions carry no coordinates, so their lengths are not
# reconciled across members: two unnamed 1-D arrays of different lengths
# coexist, while the named-dimension mismatch above is still refused
DatasetSpec(
    members={
        "a": ArraySpec.from_shape_and_data_type((3,), "uint8", dimension_names=(None,)),
        "b": ArraySpec.from_shape_and_data_type((5,), "uint8", dimension_names=(None,)),
    },
)

# a `coordinates` attribute referencing a missing member is refused
try:
    ClimateDataset(
        attributes=dataset.attributes,
        members=dict(dataset.members)
        | {"temperature": temperature.update(attributes=CFAttrs(coordinates="lat lon elevation"))},
    )
    raise AssertionError("accepted a dangling `coordinates` reference")
except ValidationError as exc:
    assert "lists coordinate 'elevation'" in exc.errors()[0]["msg"]
# --8<-- [end:dataset]

print("refusals: all three invalid layouts were rejected")

# --8<-- [start:reference]
import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.storage import LocalStore

# the same curvilinear dataset, built with real xarray: assign_coords makes
# lat/lon auxiliary coordinates, so xarray itself writes the CF
# `coordinates` attribute on the data variable
times = np.datetime64("2024-01-01") + np.arange(4).astype("timedelta64[D]")
ds = xr.Dataset(
    {
        "temperature": (
            ("time", "y", "x"),
            np.zeros((4, 3, 5), dtype="float32"),
            {"units": "degC", "standard_name": "air_temperature"},
        )
    },
    coords={"time": ("time", times)},
    attrs={"Conventions": "CF-1.8", "title": "Example climate dataset"},
)
ds = ds.assign_coords(
    lat=(
        ("y", "x"),
        np.add.outer(np.linspace(40.0, 42.0, 3), np.zeros(5)),
        {"units": "degrees_north", "standard_name": "latitude"},
    ),
    lon=(
        ("y", "x"),
        np.add.outer(np.zeros(3), np.linspace(-110.0, -106.0, 5)),
        {"units": "degrees_east", "standard_name": "longitude"},
    ),
)


def metadata_documents(root: Path) -> dict[str, object]:
    """Every zarr.json document in the store, parsed, keyed by relative path."""
    return {
        path.relative_to(root).as_posix(): json.loads(path.read_text())
        for path in root.rglob("zarr.json")
    }


with tempfile.TemporaryDirectory() as tmp:
    reference_path = Path(tmp) / "reference.zarr"
    model_path = Path(tmp) / "model.zarr"
    # consolidated=False: pydantic-zarr does not model the consolidated
    # metadata block xarray writes by default
    ds.to_zarr(
        reference_path,
        zarr_format=3,
        consolidated=False,
        encoding={
            "time": {"units": "days since 2024-01-01", "dtype": "int64"},
            "lat": {"_FillValue": None},
            "lon": {"_FillValue": None},
            "temperature": {"_FillValue": np.float32(-9999.0)},
        },
    )
    dataset.to_zarr(LocalStore(model_path), path="")

    reference_documents = metadata_documents(reference_path)
    model_documents = metadata_documents(model_path)
    assert reference_documents.keys() == model_documents.keys()
    for key in reference_documents:
        assert model_documents[key] == reference_documents[key], key
print("fidelity: the model writes exactly the documents xarray writes")
# --8<-- [end:reference]

print("SUCCESS: xarray_dataset example ran to completion")
