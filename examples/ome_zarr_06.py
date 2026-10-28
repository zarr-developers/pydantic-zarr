"""A trimmed, typed model of the OME-Zarr 0.6 image layout
(https://ngff.openmicroscopy.org/specifications/0.6/): named coordinate
systems, a discriminated union of coordinate transformations with cross-field
validators, spec-style attribute models on `AttributesBase` (absent-vs-null
told apart by the `pydantic.MISSING` sentinel), an image `GroupSpec` whose metadata is
checked against its member arrays with `get_path`, and a round trip through
an in-memory Zarr v3 store. The final section proves the model reproduces the
metadata a real 0.6 writer emits: it is checked, zarr.json by zarr.json,
against the output of ngff-zarr (the implementation that writes 0.6 today).
Rendered into docs/examples/ome_zarr_06.md via pymdownx-snippets sections.
Self-contained run (from the repo root; ngff-zarr does not itself depend on
zarr-python): uv run --no-project --with ngff-zarr --with zarr --with . \
    --python 3.12 -- python examples/ome_zarr_06.py
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-zarr", "zarr", "ngff-zarr"]
# ///
# NOTE: this needs the pydantic-zarr release carrying PR #183
# (zarr v3 metadata-document validation). ngff-zarr (>= 0.50) is only needed
# for the final reference-comparison section, which skips when it is absent.

# --8<-- [start:transforms]
from typing import Annotated, Any, Literal, Self, Union

from pydantic import ConfigDict, Field, ValidationError, model_validator

from pydantic_zarr.core import MISSING, AttributesBase


class OmeBase(AttributesBase):
    """OME-Zarr 0.6 metadata: optional keys are absent (`MISSING`), never null,
    so `None` appears in an annotation only where the spec allows a null."""

    model_config = ConfigDict(frozen=True)


class Axis(OmeBase):
    name: str = Field(min_length=1)
    type: str | MISSING = MISSING
    unit: str | MISSING = MISSING


class CoordinateSystem(OmeBase):
    name: str = Field(min_length=1)
    axes: tuple[Axis, ...] = Field(min_length=1)


class CoordinateReference(OmeBase):
    """A transformation's input/output: 0.6 allows an explicit "path": null
    here, so `path` alone admits `None` alongside the `MISSING` sentinel."""

    name: str | MISSING = MISSING
    path: str | None | MISSING = MISSING


class TransformBase(OmeBase):
    input: CoordinateReference | MISSING = MISSING
    output: CoordinateReference | MISSING = MISSING


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
print("sequence members:", [type(t).__name__ for t in level_1.transformations])
assert isinstance(level_1.transformations[0], Scale)
# inner transforms left their input/output unset, so the keys are absent
assert level_1.model_dump()["transformations"] == (
    {"type": "scale", "scale": (2.0, 2.0)},
    {"type": "translation", "translation": (0.5, 0.5)},
)

# a 3D scale cannot be followed by a 2D translation
try:
    SequenceTransform(
        transformations=(
            Scale(scale=(2.0, 2.0, 2.0)),
            Translation(translation=(0.5, 0.5)),
        )
    )
    raise AssertionError("expected refusal")
except ValidationError as exc:
    assert "sequence members disagree on dimensionality: [2, 3]" in exc.errors()[0]["msg"]

# absent-vs-null: `unit` has no None in its annotation, so a null unit is a
# validation error, while `path` admits None and a null path keeps its key
try:
    Axis(name="x", type="space", unit=None)
    raise AssertionError("expected refusal")
except ValidationError as exc:
    assert exc.errors()[0]["type"] == "string_type"
assert CoordinateReference(name="physical", path=None).model_dump() == {
    "name": "physical",
    "path": None,
}
# --8<-- [end:transforms]


# --8<-- [start:image]
import zarr
import zarr.storage

from pydantic_zarr.v3 import ArraySpec, GroupSpec, is_array_spec


class Dataset(OmeBase):
    """One resolution level: spec "multiscales > datasets". The level's single
    transformation is a scale, or a sequence of one scale followed by one
    translation; its input references the dataset by path and its output names
    the intrinsic coordinate system."""

    path: str = Field(min_length=1)
    coordinateTransformations: tuple[Union[Scale, SequenceTransform], ...] = Field(
        min_length=1, max_length=1
    )

    @model_validator(mode="after")
    def _references_valid(self) -> Self:
        transform = self.coordinateTransformations[0]
        if isinstance(transform, SequenceTransform) and not (
            len(transform.transformations) == 2
            and isinstance(transform.transformations[0], Scale)
            and isinstance(transform.transformations[1], Translation)
        ):
            raise ValueError("a dataset sequence must be one scale then one translation")
        if transform.input is MISSING or transform.input.path != self.path:
            raise ValueError(f"transform input must reference {self.path!r} by path")
        if transform.output is MISSING or transform.output.name is MISSING:
            raise ValueError("transform output must name the intrinsic coordinate system")
        return self

    @property
    def scale_vector(self) -> tuple[float, ...]:
        """The level's pixel sizes, wherever its transformation keeps them."""
        transform = self.coordinateTransformations[0]
        if isinstance(transform, Scale):
            return transform.scale
        return next(t.scale for t in transform.transformations if isinstance(t, Scale))


class Multiscale(OmeBase):
    coordinateSystems: tuple[CoordinateSystem, ...] = Field(min_length=1)
    datasets: tuple[Dataset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _intrinsic_system_and_dims(self) -> Self:
        outputs: set[str] = set()
        for ds in self.datasets:
            output = ds.coordinateTransformations[0].output
            if output is MISSING or output.name is MISSING:
                # Dataset validation guarantees this; a model smuggled in via
                # model_construct would not be re-validated, so fail loudly
                raise ValueError("each dataset's first transform needs a named output")
            outputs.add(output.name)
        if len(outputs) != 1:
            raise ValueError(f"datasets must share one intrinsic output, got {sorted(outputs)}")
        known = {cs.name for cs in self.coordinateSystems}
        if not outputs <= known:
            raise ValueError(f"intrinsic system {outputs} is not declared in {sorted(known)}")
        ndim = len(self.coordinateSystems[0].axes)
        for ds in self.datasets:
            if len(ds.scale_vector) != ndim:
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
    "s0": ArraySpec.from_shape_and_data_type((16, 16), "uint16"),
    "s1": ArraySpec.from_shape_and_data_type((8, 8), "uint16"),
}
image = Image(attributes=attributes, members=members)
s0 = image["s0"]
assert is_array_spec(s0)
assert s0.shape == (16, 16)

# round trip through an in-memory Zarr v3 store
store = zarr.storage.MemoryStore()
image.to_zarr(store, path="image.ome.zarr")
readback = Image.from_zarr(zarr.open_group(store, path="image.ome.zarr", mode="r"))
print("round trip like():", image.like(readback))
assert image.like(readback)
assert readback.attributes == attributes

# refused: metadata names a level the hierarchy does not hold
try:
    Image(attributes=attributes, members={"s0": members["s0"]})
    raise AssertionError("expected refusal")
except ValidationError as exc:
    assert "dataset path 's1' does not resolve to a member" in exc.errors()[0]["msg"]

# refused: a 3-element scale vector on a 2-dimensional image
bad = attributes.model_dump()
bad["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] = [
    0.5,
    0.5,
    0.5,
]
try:
    Image(attributes=bad, members=members)
    raise AssertionError("expected refusal")
except ValidationError as exc:
    assert "dataset 's0': scale length must be 2" in exc.errors()[0]["msg"]
# --8<-- [end:image]


# --8<-- [start:reference]
# Reference check against a real 0.6 writer. ngff-zarr (checked at 0.50.0) is,
# at the time of writing, the installable implementation that writes OME-Zarr
# 0.6 (`to_ngff_zarr(..., version="0.6")`), so this section uses it as ground
# truth instead of the spec repository's example document: it converts
# synthetic data to a two-level 0.6 image, rebuilds the same image with the
# models above, and asserts that every zarr.json in the two stores is
# identical as parsed JSON (chunk payloads are not compared). The section is
# skipped, loudly, when ngff-zarr is not installed.
import importlib.util

if importlib.util.find_spec("ngff_zarr") is None:
    print("reference check SKIPPED: ngff-zarr is not installed")
else:
    import json
    import tempfile
    from pathlib import Path

    import ngff_zarr
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        # --- the reference store, written by ngff-zarr ---
        ref_path = Path(tmp) / "reference.ome.zarr"
        pixels = np.arange(16 * 16, dtype="uint16").reshape(16, 16)
        ngff_image = ngff_zarr.to_ngff_image(
            pixels,
            dims=("y", "x"),
            scale={"y": 0.5, "x": 0.5},
            axes_units={"y": "micrometer", "x": "micrometer"},
        )
        reference = ngff_zarr.to_multiscales(ngff_image, scale_factors=[2], chunks=8)
        ngff_zarr.to_ngff_zarr(ref_path, reference, version="0.6", consolidate_metadata=False)

        ref_root = json.loads((ref_path / "zarr.json").read_text())
        ref_multiscale = ref_root["attributes"]["ome"]["multiscales"][0]

        # --- the same image, described by the models above ---
        # The structure is typed by hand; only free-form provenance values
        # ("@type", name, type, metadata) are adopted from the reference.
        def ref_level(path: str, scale: list, translation: list, name: str) -> dict:
            sequence = {
                "type": "sequence",
                "name": name,
                "input": {"path": path},
                "output": {"name": "intrinsic"},
                "transformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
            return {"path": path, "coordinateTransformations": [sequence]}

        ref_axes = [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        multiscale = Multiscale.model_validate(
            {
                "@type": ref_multiscale["@type"],
                "name": ref_multiscale["name"],
                "type": ref_multiscale["type"],
                "metadata": ref_multiscale["metadata"],
                "coordinateSystems": [{"name": "intrinsic", "axes": ref_axes}],
                "datasets": [
                    ref_level("scale0/image", [0.5, 0.5], [0.0, 0.0], "scale0_to_intrinsic"),
                    ref_level("scale1/image", [1.0, 1.0], [0.25, 0.25], "scale1_to_intrinsic"),
                ],
            }
        )

        def ref_array(shape: tuple[int, ...]) -> ArraySpec[Any]:
            return ArraySpec.from_shape_and_data_type(
                shape,
                "uint16",
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
                codecs=(
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
                ),
                dimension_names=("y", "x"),
            )

        def ref_group(shape: tuple[int, ...]) -> GroupSpec[Any, Any]:
            # ngff-zarr nests each level's array as <level>/image and labels
            # the level group with xarray-style _ARRAY_DIMENSIONS
            return GroupSpec(
                attributes={"_ARRAY_DIMENSIONS": ["y", "x"]},
                members={"image": ref_array(shape)},
            )

        mine = Image(
            attributes=ImageAttributes(ome=OmeAttrs(version="0.6", multiscales=(multiscale,))),
            members={"scale0": ref_group((16, 16)), "scale1": ref_group((8, 8))},
        )
        my_path = Path(tmp) / "model.ome.zarr"
        mine.to_zarr(zarr.storage.LocalStore(str(my_path)), path="")

        # --- compare every metadata document between the two stores ---
        ref_docs = {
            str(p.relative_to(ref_path)): json.loads(p.read_text())
            for p in ref_path.rglob("zarr.json")
        }
        my_docs = {
            str(p.relative_to(my_path)): json.loads(p.read_text())
            for p in my_path.rglob("zarr.json")
        }
        assert my_docs.keys() == ref_docs.keys(), (my_docs.keys(), ref_docs.keys())
        for doc_path in sorted(ref_docs):
            assert my_docs[doc_path] == ref_docs[doc_path], f"{doc_path} differs"
        print(f"reference check: {len(ref_docs)} zarr.json documents match ngff-zarr exactly")
# --8<-- [end:reference]

print("OME-Zarr 0.6 example: all checks passed")
