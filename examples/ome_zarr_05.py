"""
A typed model of the OME-Zarr 0.5 image layout (https://ngff.openmicroscopy.org/0.5/):
attribute models built on `AttributesBase` (hyphenated spec keys via aliases, optional
keys omitted rather than written as `null`), a discriminated union for the spec's closed
set of coordinate transformations, typed group members via a PEP 728 `TypedDict` with
`extra_items`, cross-field validators tying the metadata to the members with `get_path`,
and a store round trip. The final section proves the model writes exactly what the
reference implementation (ome-zarr-py) writes for the same image. Rendered in the docs
at docs/examples/ome_zarr_05.md. The dev environment does not carry ome-zarr; run with:
uv run --no-project --with-editable . --with ome-zarr --with 'zarr>=3.1' \
    --python 3.12 -- python examples/ome_zarr_05.py
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-zarr", "zarr", "ome-zarr"]
# ///
# NOTE: needs the pydantic-zarr release carrying PR #183
# (AttributesBase, ArraySpec.from_shape_and_data_type, GroupSpec.get_path).

import copy
from typing import Annotated, ClassVar, Literal, NotRequired, Self, Union

import zarr
import zarr.storage
from pydantic import ConfigDict, Field, ValidationError, model_validator
from typing_extensions import TypedDict

from pydantic_zarr.core import AttributesBase
from pydantic_zarr.v3 import ArraySpec, CodecLike, GroupSpec, is_array_spec

# --8<-- [start:attrs]


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
assert image_label.colors is not None
assert image_label.colors[0].label_value == 1

# serialization writes the alias, and omits the unset optional key "colors"
assert ImageLabel(version="0.5", colors=(LabelColor(label_value=2),)).model_dump() == {
    "version": "0.5",
    "colors": ({"label-value": 2},),
}
assert ImageLabel(version="0.5").model_dump() == {"version": "0.5"}

# the spec has no null-valued keys, so an explicit None is refused
try:
    ImageLabel(version=None)
except ValidationError as e:
    assert "refuses an explicit None" in e.errors()[0]["msg"]
else:
    raise AssertionError("expected refusal")
# --8<-- [end:attrs]

# --8<-- [start:image]


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
    """Attributes["ome"] of an image group: versioned, with multiscales."""

    version: Literal["0.5"]
    multiscales: tuple[Multiscale, ...] = Field(min_length=1)


class ImageAttributes(AttributesBase):
    ome: OmeImageBlock


class MultiscaleImageMembers(TypedDict, extra_items=ArraySpec):
    """Dynamically named resolution arrays, plus at most one "labels" group."""

    # keep the enclosing model's extra="forbid" config off this TypedDict;
    # checkers reject assignments in TypedDict bodies, but pydantic documents
    # this configuration mechanism, hence the ignore
    __pydantic_config__ = ConfigDict()  # type: ignore[misc]

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
                    # pydantic converts ValueError, not TypeError, into a validation error
                    raise ValueError(msg)  # noqa: TRY004
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
        "0": ArraySpec.from_shape_and_data_type(
            (2, 32, 32), "uint16", dimension_names=("c", "y", "x")
        ),
        "1": ArraySpec.from_shape_and_data_type(
            (2, 16, 16), "uint16", dimension_names=("c", "y", "x")
        ),
        "labels": GroupSpec(attributes={"ome": {"version": "0.5", "labels": []}}),
    },
)
assert sorted(image.members) == ["0", "1", "labels"]
level0 = image["0"]
assert is_array_spec(level0)
assert level0.shape == (2, 32, 32)
print("built a 2-level OME-Zarr 0.5 image:", sorted(image.members))

# write the hierarchy to a store and read it back
store = zarr.storage.MemoryStore()
image.to_zarr(store, path="image.zarr")
round_tripped = MultiscaleImage.from_zarr(zarr.open_group(store, path="image.zarr", mode="r"))
assert image.like(round_tripped)
assert image == round_tripped
print("store round trip compares equal")

# a dataset path that no longer resolves to a member is refused
bad = copy.deepcopy(image.model_dump(mode="json"))
bad["members"]["00"] = bad["members"].pop("0")
try:
    MultiscaleImage.model_validate(bad)
except ValidationError as e:
    assert "dataset path '0' is not an array member" in e.errors()[0]["msg"]
else:
    raise AssertionError("expected refusal")

# any member other than "labels" must be an array, so a stray subgroup is refused
bad = copy.deepcopy(image.model_dump(mode="json"))
bad["members"]["scratch"] = {"zarr_format": 3, "node_type": "group", "attributes": {}}
try:
    MultiscaleImage.model_validate(bad)
except ValidationError as e:
    assert e.errors()[0]["loc"][:2] == ("members", "scratch")
    assert e.errors()[0]["msg"] == "Input should be 'array'"
else:
    raise AssertionError("expected refusal")
# --8<-- [end:image]

# --8<-- [start:reference]
# The model should write exactly what a reference OME-Zarr 0.5 implementation
# writes. ome-zarr-py generates a multiscale image from synthetic data; a
# MultiscaleImage instance then describes that same image, writes it to a
# second store, and every zarr.json in the two stores must be identical.
import json
import pathlib
import tempfile

import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

reference_dir = pathlib.Path(tempfile.mkdtemp()) / "reference.zarr"
write_image(
    image=np.zeros((2, 32, 32), dtype="uint16"),
    group=zarr.group(store=parse_url(str(reference_dir), mode="w").store),
    axes=[
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ],
    scale_factors=[2],
)

# the same image, described by the model: ome-zarr-py's layout and choices
# (paths s0/s1, per-dataset scale + translation, whole-shape chunks, zstd)
reference_codecs: tuple[CodecLike, ...] = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
)


def reference_level(shape: tuple[int, ...]) -> ArraySpec:
    return ArraySpec.from_shape_and_data_type(
        shape,
        "uint16",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": shape}},
        codecs=reference_codecs,
        dimension_names=("c", "y", "x"),
    )


modeled = MultiscaleImage(
    attributes=ImageAttributes(
        ome=OmeImageBlock(
            version="0.5",
            multiscales=(
                Multiscale(
                    name="image",
                    axes=(
                        Axis(name="c", type="channel"),
                        Axis(name="y", type="space"),
                        Axis(name="x", type="space"),
                    ),
                    datasets=(
                        Dataset(
                            path="s0",
                            coordinateTransformations=(
                                ScaleTransform(scale=(1.0, 1.0, 1.0)),
                                TranslationTransform(translation=(0.0, 0.0, 0.0)),
                            ),
                        ),
                        Dataset(
                            path="s1",
                            coordinateTransformations=(
                                ScaleTransform(scale=(1.0, 2.0, 2.0)),
                                TranslationTransform(translation=(0.0, 0.5, 0.5)),
                            ),
                        ),
                    ),
                ),
            ),
        )
    ),
    members={"s0": reference_level((2, 32, 32)), "s1": reference_level((2, 16, 16))},
)
modeled_dir = pathlib.Path(tempfile.mkdtemp()) / "modeled.zarr"
modeled.to_zarr(zarr.storage.LocalStore(modeled_dir), path="")


def metadata_documents(root: pathlib.Path) -> dict[str, object]:
    """Every zarr.json under `root`, parsed, keyed by its relative path."""
    return {str(p.relative_to(root)): json.loads(p.read_text()) for p in root.rglob("zarr.json")}


reference_docs = metadata_documents(reference_dir)
modeled_docs = metadata_documents(modeled_dir)
assert reference_docs.keys() == modeled_docs.keys(), (
    sorted(reference_docs),
    sorted(modeled_docs),
)
for key, reference_doc in reference_docs.items():
    assert modeled_docs[key] == reference_doc, f"{key} differs"
print("model output matches the ome-zarr-py reference:", sorted(reference_docs))
# --8<-- [end:reference]

print("ome_zarr_05.py: all assertions passed")
