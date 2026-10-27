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
Each code block below is self-contained, so the second one redeclares the
handful of classes it needs.

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
from typing import Annotated, ClassVar, Literal, Self, Union

from pydantic import ConfigDict, Field, ValidationError, model_validator

from pydantic_zarr.core import AttributesBase


class OmeBase(AttributesBase):
    """OME-Zarr 0.6 metadata: optional keys are absent, never null."""

    model_config = ConfigDict(frozen=True)
    explicit_none: ClassVar[Literal["keep", "omit", "refuse"]] = "refuse"


class Axis(OmeBase):
    name: str = Field(min_length=1)
    type: str | None = None
    unit: str | None = None


class CoordinateSystem(OmeBase):
    name: str = Field(min_length=1)
    axes: tuple[Axis, ...] = Field(min_length=1)


class CoordinateReference(AttributesBase):
    """A transformation's input/output: 0.6 allows an explicit "path": null
    here, so this model keeps the default explicit_none="keep" policy."""

    model_config = ConfigDict(frozen=True)
    name: str | None = None
    path: str | None = None


class TransformBase(OmeBase):
    input: CoordinateReference | None = None
    output: CoordinateReference | None = None


class Identity(TransformBase):
    type: Literal["identity"] = "identity"


class Scale(TransformBase):
    type: Literal["scale"] = "scale"
    scale: tuple[float, ...] = Field(min_length=1)


class Translation(TransformBase):
    type: Literal["translation"] = "translation"
    translation: tuple[float, ...] = Field(min_length=1)


Transform = Annotated[Union[Identity, Scale, Translation], Field(discriminator="type")]


class SequenceTransform(TransformBase):
    type: Literal["sequence"] = "sequence"
    transformations: tuple[Transform, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _members_dims_agree(self) -> Self:
        dims = {
            len(t.scale) if isinstance(t, Scale) else len(t.translation)
            for t in self.transformations
            if isinstance(t, (Scale, Translation))
        }
        if len(dims) > 1:
            raise ValueError(f"sequence members disagree on dimensionality: {sorted(dims)}")
        return self


# raw 0.6 metadata parses into the tagged union
level_1 = SequenceTransform.model_validate(
    {
        "type": "sequence",
        "input": {"path": "s1"},
        "output": {"name": "physical"},
        "transformations": [
            {"type": "scale", "scale": [2.0, 2.0]},
            {"type": "translation", "translation": [0.5, 0.5]},
        ],
    }
)
print(type(level_1.transformations[0]).__name__)
#> Scale
print(level_1.model_dump()["transformations"])
"""
(
    {'type': 'scale', 'scale': (2.0, 2.0)},
    {'type': 'translation', 'translation': (0.5, 0.5)},
)
"""

# a 3D scale cannot be followed by a 2D translation
try:
    SequenceTransform(
        transformations=(
            Scale(scale=(2.0, 2.0, 2.0)),
            Translation(translation=(0.5, 0.5)),
        )
    )
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    #> Value error, sequence members disagree on dimensionality: [2, 3]

# absent-vs-null: a null unit is refused, a null reference path is kept
try:
    Axis(name="x", type="space", unit=None)
except ValidationError:
    print("explicit null unit refused")
    #> explicit null unit refused
print(CoordinateReference(name="physical", path=None).model_dump())
#> {'name': 'physical', 'path': None}
```

## The image group

An image is a group whose `ome.multiscales` metadata must agree with the
group's actual members: every `datasets[*].path` resolves (via `get_path`) to
an array whose dimensionality matches the coordinate systems, every level
maps into the same intrinsic system, and the scale vector length equals the
image dimensionality. The resolution-level arrays are metadata-only
`ArraySpec.from_shape` documents, the hierarchy round-trips through an
in-memory Zarr v3 store, and invalid layouts are refused at construction.
(The full spec allows identity and scale-then-translation sequences per
dataset and ten more transformation types; this trimmed model fixes each
level to a single scale.)

```python
from typing import Any, ClassVar, Literal, Self, Union

import zarr
import zarr.storage
from pydantic import ConfigDict, Field, ValidationError, model_validator

from pydantic_zarr.core import AttributesBase
from pydantic_zarr.v3 import ArraySpec, GroupSpec


class OmeBase(AttributesBase):
    model_config = ConfigDict(frozen=True)
    explicit_none: ClassVar[Literal["keep", "omit", "refuse"]] = "refuse"


class Axis(OmeBase):
    name: str = Field(min_length=1)
    type: str | None = None
    unit: str | None = None


class CoordinateSystem(OmeBase):
    name: str = Field(min_length=1)
    axes: tuple[Axis, ...] = Field(min_length=1)


class CoordinateReference(AttributesBase):
    model_config = ConfigDict(frozen=True)
    name: str | None = None
    path: str | None = None


class Scale(OmeBase):
    type: Literal["scale"] = "scale"
    scale: tuple[float, ...] = Field(min_length=1)
    input: CoordinateReference
    output: CoordinateReference


class Dataset(OmeBase):
    """One resolution level: spec "multiscales > datasets"."""

    path: str = Field(min_length=1)
    coordinateTransformations: tuple[Scale, ...] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def _references_valid(self) -> Self:
        transform = self.coordinateTransformations[0]
        if transform.input.path != self.path:
            raise ValueError(f"transform input must reference {self.path!r} by path")
        if transform.output.name is None:
            raise ValueError("transform output must name the intrinsic coordinate system")
        return self


class Multiscale(OmeBase):
    coordinateSystems: tuple[CoordinateSystem, ...] = Field(min_length=1)
    datasets: tuple[Dataset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _intrinsic_system_and_dims(self) -> Self:
        outputs = {ds.coordinateTransformations[0].output.name for ds in self.datasets}
        if len(outputs) != 1:
            raise ValueError(f"datasets must share one intrinsic output, got {sorted(outputs)}")
        known = {cs.name for cs in self.coordinateSystems}
        if not outputs <= known:
            raise ValueError(f"intrinsic system {outputs} is not declared in {sorted(known)}")
        ndim = len(self.coordinateSystems[0].axes)
        for ds in self.datasets:
            if len(ds.coordinateTransformations[0].scale) != ndim:
                raise ValueError(f"dataset {ds.path!r}: scale length must be {ndim}")
        return self


class OmeAttrs(OmeBase):
    version: Literal["0.6"]
    multiscales: tuple[Multiscale, ...] = Field(min_length=1)


class ImageAttributes(OmeBase):
    ome: OmeAttrs


class Image(GroupSpec[ImageAttributes, Union[ArraySpec[Any], GroupSpec[Any, Any]]]):
    """A 0.6 image group: the metadata must match the member arrays."""

    @model_validator(mode="after")
    def _members_match_metadata(self) -> Self:
        if self.members is None:  # members unknown: nothing to check against
            return self
        for ms in self.attributes.ome.multiscales:
            ndim = len(ms.coordinateSystems[0].axes)
            for ds in ms.datasets:
                node = self.get_path(ds.path)
                if node is None:
                    raise ValueError(f"dataset path {ds.path!r} does not resolve to a member")
                if not isinstance(node, ArraySpec) or len(node.shape) != ndim:
                    raise ValueError(f"dataset path {ds.path!r} must be a {ndim}D array")
        return self


def level(path: str, pixel_size: float) -> Dataset:
    scale = Scale(
        scale=(pixel_size, pixel_size),
        input=CoordinateReference(path=path),
        output=CoordinateReference(name="physical"),
    )
    return Dataset(path=path, coordinateTransformations=(scale,))


micrometer_yx = (
    Axis(name="y", type="space", unit="micrometer"),
    Axis(name="x", type="space", unit="micrometer"),
)
attributes = ImageAttributes(
    ome=OmeAttrs(
        version="0.6",
        multiscales=(
            Multiscale(
                coordinateSystems=(CoordinateSystem(name="physical", axes=micrometer_yx),),
                datasets=(level("s0", 0.5), level("s1", 1.0)),
            ),
        ),
    )
)
members = {
    "s0": ArraySpec.from_shape((16, 16), "uint16"),
    "s1": ArraySpec.from_shape((8, 8), "uint16"),
}
image = Image(attributes=attributes, members=members)
print(image["s0"].shape)
#> (16, 16)

# round trip through an in-memory Zarr v3 store
store = zarr.storage.MemoryStore()
image.to_zarr(store, path="image.ome.zarr")
readback = Image.from_zarr(zarr.open_group(store, path="image.ome.zarr", mode="r"))
print(image.like(readback))
#> True
print(readback.attributes == attributes)
#> True

# refused: metadata names a level the hierarchy does not hold
try:
    Image(attributes=attributes, members={"s0": members["s0"]})
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    #> Value error, dataset path 's1' does not resolve to a member

# refused: a 3-element scale vector on a 2-dimensional image
bad = attributes.model_dump()
bad["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] = [
    0.5,
    0.5,
    0.5,
]
try:
    Image(attributes=bad, members=members)
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
    #> Value error, dataset 's0': scale length must be 2
```
