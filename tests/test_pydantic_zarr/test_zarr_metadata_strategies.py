"""
Property tests fed by ``zarr-metadata``'s own Hypothesis strategies.

``zarr_metadata.testing.strategies`` draws valid Zarr V3 metadata: whole
array and group documents whose members fit each other, from every data
type, chunk grid, chunk key encoding and codec the package defines. That
is the space the readers in `pydantic_zarr.v3` claim to accept, so the
property is agreement: what ``zarr-metadata`` draws as valid, a spec reads,
and what the spec writes back, ``zarr-metadata`` reads without a problem.

The strategies ship with ``zarr-metadata`` 0.7.0 and later, behind its
``testing`` extra; this module is skipped without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pydantic_zarr.v3 import ArraySpec, BaseGroupSpec

zs = pytest.importorskip(
    "zarr_metadata.testing.strategies",
    reason="zarr-metadata's testing extra (0.7.0 and later) provides the strategies",
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


# every document the strategies draw, from every definition they hold, in
# every envelope spelling the spec allows
READABLE_ARRAYS = zs.array_documents()


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
@given(document=zs.consolidated_group_documents())
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


@PROPERTY_SETTINGS
@given(data=st.data())
def test_fill_value_spellings_alike(data: st.DataObject) -> None:
    """
    Two spellings zarr-metadata draws for one data type's fill value, placed
    in the same array document, give specs that are alike exactly when
    zarr-metadata reads the two documents as equal models. Model equality is
    zarr-metadata's judgment of sameness: a fill value's spellings (a float
    and its hex bits, `"NaN"` and its canonical bits, `"NaT"` and the int64
    minimum) compare equal, where the values as written do not.
    """
    from zarr_metadata.definition._definition import kind_of
    from zarr_metadata.v3.definition import DataTypeDefinition

    # structs and the variable-length types have no fill value of their own to respell
    definitions = [
        definition
        for definition in zs.DEFAULT_STRATEGIES
        if (kind := kind_of(definition)) is not None
        and issubclass(kind, DataTypeDefinition)
        and definition.name not in {"struct", "variable_length_utf8", "variable_length_bytes"}
    ]
    definition = data.draw(st.sampled_from(definitions))
    configuration = data.draw(zs.data_type_configurations(definition))
    first = data.draw(zs.fill_values(definition, configuration))
    second = data.draw(zs.fill_values(definition, configuration))
    data_type: object = (
        definition.name
        if configuration == {}
        else {"name": definition.name, "configuration": configuration}
    )
    base = data.draw(READABLE_ARRAYS)
    documents = [{**base, "data_type": data_type, "fill_value": fill} for fill in (first, second)]
    models = [read_array_metadata_v3(document).metadata for document in documents]
    # the base document's codecs may have been drawn for another data type;
    # such examples are discarded, so Hypothesis can tell how many checked anything
    assume(models[0] is not None and models[1] is not None)
    specs = [ArraySpec.model_validate(document) for document in documents]
    assert specs[0].like(specs[1]) == (models[0] == models[1])


@pytest.mark.parametrize(
    ("data_type", "fills", "alike"),
    [
        ("float16", (0.0, "0x0000"), True),
        ("float16", (0.0, -0.0), False),
        ("float16", ("NaN", "0x7e00"), True),
        ("float16", ("NaN", "0x7e01"), False),
        ("float32", ("Infinity", "0x7f800000"), True),
        ("complex64", ((0.0, "NaN"), (0.0, "0x7fc00000")), True),
        (
            {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
            ("NaT", -(2**63)),
            True,
        ),
        (
            {"name": "numpy.timedelta64", "configuration": {"unit": "s", "scale_factor": 1}},
            ("NaT", -(2**63)),
            True,
        ),
    ],
    ids=str,
)
def test_fill_value_spellings_alike_examples(
    data_type: object, fills: tuple[object, object], *, alike: bool
) -> None:
    """
    The spellings `test_fill_value_spellings_alike` may not draw together:
    signed zeros and NaN payloads differ, a value and its hex bits agree, and
    a numpy time type's `"NaT"` is the int64 minimum, as numpy stores it. Each
    pair is alike exactly when zarr-metadata reads the documents as equal models.
    """
    base = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (),
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": ()}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    documents = [{**base, "fill_value": fill} for fill in fills]
    models = [read_array_metadata_v3(document).metadata for document in documents]
    assert models[0] is not None
    assert models[1] is not None
    assert (models[0] == models[1]) is alike
    specs = [ArraySpec.model_validate(document) for document in documents]
    assert specs[0].like(specs[1]) is alike
