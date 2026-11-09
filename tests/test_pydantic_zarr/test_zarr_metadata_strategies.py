"""
Property tests fed by ``zarr-metadata``'s own Hypothesis strategies.

``zarr_metadata.testing.strategies`` draws valid Zarr V3 metadata: whole
array and group documents whose members fit each other, from every data
type, chunk grid, chunk key encoding and codec the package defines. That
is the space the readers in `pydantic_zarr.v3` claim to accept, so the
property is agreement: what ``zarr-metadata`` draws as valid, a spec reads,
and what the spec writes back, ``zarr-metadata`` reads without a problem.

Where the two still disagree, a strict expected failure records it, so a
fix on either side turns the test into a failure that says to drop the
mark. The strategies ship with the ``zarr-metadata`` release after 0.6.0,
behind its ``testing`` extra; this module is skipped without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pydantic_zarr.v3 import ArraySpec, BaseGroupSpec

zs = pytest.importorskip(
    "zarr_metadata.testing.strategies",
    reason="zarr-metadata's testing extra (its next release) provides the strategies",
)

from zarr_metadata.model import read_array_metadata_v3, read_group_metadata_v3  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Mapping

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

ARRAY_FIELDS = frozenset(
    {
        "zarr_format",
        "node_type",
        "shape",
        "data_type",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
        "attributes",
        "dimension_names",
        "storage_transformers",
    }
)
GROUP_FIELDS = frozenset({"zarr_format", "node_type", "attributes", "consolidated_metadata"})


def _without(*names: str) -> dict[Any, Any]:
    """`DEFAULT_STRATEGIES` without the definitions called `names` (a `Strategies` mapping)."""
    return {
        definition: strategy
        for definition, strategy in zs.DEFAULT_STRATEGIES.items()
        if definition.name not in names
    }


# every document the strategies draw, in every envelope spelling the spec
# allows, except one with a `cast_value` codec (see
# `test_cast_value_keeps_bytes_endian`)
READABLE_ARRAYS = zs.array_documents(strategies=_without("cast_value"))


@PROPERTY_SETTINGS
@given(document=READABLE_ARRAYS)
def test_array_documents_agree(document: Mapping[str, Any]) -> None:
    """
    Every array document zarr-metadata draws, in any envelope spelling the
    spec allows, `ArraySpec` reads, as the same spec it reads from the
    document's canonical spelling; the document it writes back, zarr-metadata
    reads without a problem, as the same array; and the spec round-trips
    through its own document.
    """
    spec: ArraySpec[Any] = ArraySpec.model_validate(document)
    drawn = read_array_metadata_v3(document)
    assert drawn.metadata is not None
    assert ArraySpec.model_validate(drawn.metadata.to_json()) == spec
    written = spec.to_metadata_json()
    reading = read_array_metadata_v3(written)
    assert reading.problems == ()
    assert reading.metadata is not None
    assert reading.metadata.shape == tuple(document["shape"])
    assert ArraySpec.model_validate(written) == spec
    assert spec.like(spec)
    # extra fields the spec does not define are kept as drawn
    for key in set(document) - ARRAY_FIELDS:
        assert key in spec.extra_fields


@PROPERTY_SETTINGS
@given(document=zs.group_documents())
def test_group_documents_agree(document: Mapping[str, Any]) -> None:
    """
    Every group document zarr-metadata draws, `BaseGroupSpec` reads, keeping
    the attributes and the extra fields as drawn; the document it writes back,
    zarr-metadata reads without a problem; and the spec round-trips through
    its own document.
    """
    spec: BaseGroupSpec[Any] = BaseGroupSpec.model_validate(document)
    assert spec.attributes == document.get("attributes", {})
    assert set(spec.extra_fields) == set(document) - GROUP_FIELDS
    written = spec.to_metadata_json()
    assert read_group_metadata_v3(written).problems == ()
    assert BaseGroupSpec.model_validate(written) == spec


@PROPERTY_SETTINGS
@given(document=zs.consolidated_group_documents(strategies=_without("cast_value")))
def test_consolidated_group_documents_kept_as_written(document: Mapping[str, Any]) -> None:
    """
    A group document carrying ``consolidated_metadata`` reads as a
    `BaseGroupSpec` that keeps the record as written, in its extra fields,
    and writes it back unchanged.
    """
    spec: BaseGroupSpec[Any] = BaseGroupSpec.model_validate(document)
    assert "consolidated_metadata" in spec.extra_fields
    written = spec.to_metadata_json()
    assert read_group_metadata_v3(written).problems == ()
    assert written["consolidated_metadata"] == document["consolidated_metadata"]


# ---------------------------------------------------------------------------
# Disagreements between zarr-metadata's strategies and the specs, each kept as
# a strict expected failure until one side changes.


@pytest.mark.xfail(
    strict=True,
    reason="canonicalize_declarations drops the bytes codec's endian for a single-byte array "
    "data type, though a cast_value codec before it hands bytes a multi-byte chunk",
)
@PROPERTY_SETTINGS
@given(
    document=zs.array_documents(spellings="canonical").filter(
        lambda document: any(codec.get("name") == "cast_value" for codec in document["codecs"])
    )
)
def test_cast_value_keeps_bytes_endian(document: Mapping[str, Any]) -> None:
    """An array document whose pipeline casts the chunk reads as an `ArraySpec`."""
    ArraySpec.model_validate(document)


_TIME_TYPES = frozenset({"numpy.datetime64", "numpy.timedelta64"})


@PROPERTY_SETTINGS
@given(data=st.data())
def test_fill_value_spellings_alike(data: st.DataObject) -> None:
    """
    Two spellings zarr-metadata draws for one data type's fill value, placed
    in the same array document, give specs that are alike exactly when
    zarr-metadata reads them as the same value.
    """
    from zarr_metadata.definition._definition import kind_of
    from zarr_metadata.v3.definition import DataTypeDefinition

    # structs and the variable-length types have no fill value of their own to
    # respell; the numpy time types disagree, see `test_numpy_time_nat_alike`
    definitions = [
        definition
        for definition in zs.DEFAULT_STRATEGIES
        if (kind := kind_of(definition)) is not None
        and issubclass(kind, DataTypeDefinition)
        and definition.name not in _TIME_TYPES
        and definition.name not in {"struct", "variable_length_utf8", "variable_length_bytes"}
    ]
    definition = data.draw(st.sampled_from(definitions))
    configuration = data.draw(zs.data_type_configurations(definition))
    first = data.draw(zs.fill_values(definition, configuration))
    second = data.draw(zs.fill_values(definition, configuration))
    data_type: object = (
        definition.name
        if configuration == {}
        else {
            "name": definition.name,
            "configuration": configuration,
        }
    )
    base = data.draw(READABLE_ARRAYS)
    documents = [{**base, "data_type": data_type, "fill_value": fill} for fill in (first, second)]
    readings = [read_array_metadata_v3(document) for document in documents]
    first, second = readings[0].metadata, readings[1].metadata
    if first is None or second is None:
        # the base document's codecs were drawn for another data type
        return
    specs = [ArraySpec.model_validate(document) for document in documents]
    same_value = first.fill_value == second.fill_value
    assert specs[0].like(specs[1]) == same_value


@pytest.mark.xfail(
    strict=True,
    reason='zarr-metadata reads a numpy time type\'s "NaT" and the int64 minimum as two fill '
    "values; ArraySpec.like equates them, as numpy stores NaT as that integer",
)
@pytest.mark.parametrize("name", sorted(_TIME_TYPES))
def test_numpy_time_nat_alike(name: str) -> None:
    """
    Specs whose fill values are `"NaT"` and the int64 minimum are alike exactly
    when zarr-metadata reads them as the same value.
    """
    base = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (),
        "data_type": {"name": name, "configuration": {"unit": "s", "scale_factor": 1}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": ()}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    documents = [{**base, "fill_value": fill} for fill in ("NaT", -(2**63))]
    readings = [read_array_metadata_v3(document) for document in documents]
    first, second = readings[0].metadata, readings[1].metadata
    assert first is not None
    assert second is not None
    specs = [ArraySpec.model_validate(document) for document in documents]
    same_value = first.fill_value == second.fill_value
    assert specs[0].like(specs[1]) == same_value
