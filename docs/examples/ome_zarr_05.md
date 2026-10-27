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

## Attribute models with spec spellings

The spec's `image-label` object uses keys that are not Python identifiers
(`label-value`) and marks most keys OPTIONAL: an absent key is meaningful,
a JSON `null` is not. `AttributesBase` handles both: aliases are used on
input and output, a field left unset is omitted, and the
`explicit_none = "refuse"` policy rejects an explicit `None` outright.

```python
from typing import ClassVar, Literal

from pydantic import Field, ValidationError
from pydantic_zarr.core import AttributesBase


class LabelColor(AttributesBase):
    """One entry of "image-label".colors: a label value and its RGBA color."""

    label_value: int = Field(alias="label-value")
    rgba: tuple[int, int, int, int] | None = None


class ImageLabel(AttributesBase):
    """The "image-label" object of a label image (OME-Zarr 0.5, section 2.6)."""

    explicit_none: ClassVar[Literal["keep", "omit", "refuse"]] = "refuse"

    version: str | None = None
    colors: tuple[LabelColor, ...] | None = None


image_label = ImageLabel.model_validate(
    {"version": "0.5", "colors": [{"label-value": 1, "rgba": [0, 128, 0, 128]}]}
)
print(image_label.colors[0].label_value)
#> 1

# serialization writes the alias, and omits the unset optional key "colors"
print(ImageLabel(version="0.5", colors=(LabelColor(label_value=2),)).model_dump())
#> {'version': '0.5', 'colors': ({'label-value': 2},)}
print(ImageLabel(version="0.5").model_dump())
#> {'version': '0.5'}

# the spec has no null-valued keys, so an explicit None is refused
try:
    ImageLabel(version=None)
except ValidationError as e:
    print(e.errors()[0]["msg"])
    """
    Value error, Field 'version' is None, and ImageLabel refuses an explicit None (explicit_none="refuse"): leave the field unset for an absent key.
    """
```

## A multiscale image group

The `ome` block of an image group holds `multiscales`: axes, and one dataset
entry per resolution level, each carrying `scale`/`translation`
transformations (the only types the spec allows there — a closed discriminated
union). The group model narrows `members` to a `TypedDict`: the fixed name
`labels` may hold a subgroup, and every other member must be an array
(`extra_items=ArraySpec`), so a stray subgroup is refused by name. Validators
then cross-check the metadata against the members: every dataset `path` must
resolve to an array whose dimensionality matches the axes.

```python
import copy
from typing import Annotated, Literal, Self, Union

import zarr
import zarr.storage
from pydantic import ConfigDict, Field, ValidationError, model_validator
from typing_extensions import NotRequired, TypedDict

from pydantic_zarr.core import AttributesBase
from pydantic_zarr.v3 import ArraySpec, GroupSpec


class Axis(AttributesBase):
    """One axis (section 2.1): "name" MUST be given; "type" and "unit" SHOULD."""

    name: str
    type: str | None = None
    unit: str | None = None


class ScaleTransform(AttributesBase):
    type: Literal["scale"] = "scale"
    scale: tuple[float, ...]


class TranslationTransform(AttributesBase):
    type: Literal["translation"] = "translation"
    translation: tuple[float, ...]


# section 2.4: inside "multiscales" only scale and translation are allowed
CoordinateTransformation = Annotated[
    Union[ScaleTransform, TranslationTransform], Field(discriminator="type")
]


class Dataset(AttributesBase):
    """One resolution level: a path and its data-to-physical transformations."""

    path: str
    coordinateTransformations: tuple[CoordinateTransformation, ...]


class Multiscale(AttributesBase):
    """One multiscale image description (section 2.4)."""

    axes: tuple[Axis, ...] = Field(min_length=2, max_length=5)
    datasets: tuple[Dataset, ...] = Field(min_length=1)
    name: str | None = None

    @model_validator(mode="after")
    def _transform_vectors_match_axes(self) -> Self:
        for dataset in self.datasets:
            for transform in dataset.coordinateTransformations:
                vector = getattr(transform, transform.type)
                if len(vector) != len(self.axes):
                    msg = (
                        f"dataset {dataset.path!r}: {transform.type} vector has "
                        f"length {len(vector)}, but there are {len(self.axes)} axes"
                    )
                    raise ValueError(msg)
        return self


class OmeImageBlock(AttributesBase):
    """attributes["ome"] of an image group: versioned, with multiscales."""

    version: Literal["0.5"]
    multiscales: tuple[Multiscale, ...] = Field(min_length=1)


class ImageAttributes(AttributesBase):
    ome: OmeImageBlock


class MultiscaleImageMembers(TypedDict, extra_items=ArraySpec):
    """Dynamically named resolution arrays, plus at most one "labels" group."""

    # keep the enclosing model's extra="forbid" config off this TypedDict
    __pydantic_config__ = ConfigDict()

    labels: NotRequired[GroupSpec]


class MultiscaleImage(GroupSpec):
    """An OME-Zarr 0.5 multiscale image group (sections 1.1 and 2.4)."""

    attributes: ImageAttributes
    members: MultiscaleImageMembers

    @model_validator(mode="after")
    def _datasets_resolve_to_arrays(self) -> Self:
        for multiscale in self.attributes.ome.multiscales:
            for dataset in multiscale.datasets:
                node = self.get_path(dataset.path)
                if not isinstance(node, ArraySpec):
                    msg = f"dataset path {dataset.path!r} is not an array member"
                    raise ValueError(msg)
                if len(node.shape) != len(multiscale.axes):
                    msg = (
                        f"array at {dataset.path!r} has {len(node.shape)} "
                        f"dimensions, but there are {len(multiscale.axes)} axes"
                    )
                    raise ValueError(msg)
        return self


multiscale = Multiscale(
    name="example",
    axes=(
        Axis(name="c", type="channel"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ),
    datasets=(
        Dataset(
            path="0",
            coordinateTransformations=(ScaleTransform(scale=(1.0, 0.25, 0.25)),),
        ),
        Dataset(
            path="1",
            coordinateTransformations=(ScaleTransform(scale=(1.0, 0.5, 0.5)),),
        ),
    ),
)
image = MultiscaleImage(
    attributes=ImageAttributes(ome=OmeImageBlock(version="0.5", multiscales=(multiscale,))),
    members={
        "0": ArraySpec.from_shape((2, 32, 32), "uint16", dimension_names=("c", "y", "x")),
        "1": ArraySpec.from_shape((2, 16, 16), "uint16", dimension_names=("c", "y", "x")),
        "labels": GroupSpec(attributes={"ome": {"version": "0.5", "labels": []}}),
    },
)
print(sorted(image.members))
#> ['0', '1', 'labels']
print(image["0"].shape)
#> (2, 32, 32)

# write the hierarchy to a store and read it back
store = zarr.storage.MemoryStore()
image.to_zarr(store, path="image.zarr")
round_tripped = MultiscaleImage.from_zarr(zarr.open_group(store, path="image.zarr", mode="r"))
print(image.like(round_tripped), image == round_tripped)
#> True True

# a dataset path that no longer resolves to a member is refused
bad = copy.deepcopy(image.model_dump(mode="json"))
bad["members"]["00"] = bad["members"].pop("0")
try:
    MultiscaleImage.model_validate(bad)
except ValidationError as e:
    print(e.errors()[0]["msg"])
    #> Value error, dataset path '0' is not an array member

# any member other than "labels" must be an array, so a stray subgroup is refused
bad = copy.deepcopy(image.model_dump(mode="json"))
bad["members"]["scratch"] = {"zarr_format": 3, "node_type": "group", "attributes": {}}
try:
    MultiscaleImage.model_validate(bad)
except ValidationError as e:
    print(e.errors()[0]["loc"][:2], e.errors()[0]["msg"])
    #> ('members', 'scratch') Input should be 'array'
```

A complete model of the layout would also type the `labels` group and its
label images (integer data types, `image-label` attributes as above, and the
rule that label paths listed in the labels group resolve to members), using
exactly the same ingredients: `AttributesBase` subclasses for the `ome`
blocks, narrowed `members`, and `get_path` in validators.
