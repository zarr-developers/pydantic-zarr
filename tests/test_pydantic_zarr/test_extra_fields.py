"""
Extra fields: metadata document members the Zarr spec does not define.

Zarr V3 admits an extra field a reader recognizes (one a definition in the
scope claims, such as a group's `consolidated_metadata` in `CORE`), and one
it does not recognize only as an extension object that sets
`"must_understand": false`. Zarr V2 admits any other key in array metadata
(it "SHOULD be ignored") and none in group metadata.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, NotRequired

import pytest
from packaging.version import Version
from pydantic import ValidationError
from typing_extensions import TypedDict
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ArrayExtraFieldDefinition,
    ZarrV3Context,
)

from pydantic_zarr import v2, v3
from pydantic_zarr.core import tuplify_json

from .conftest import ZARR_PYTHON_VERSION

if TYPE_CHECKING:
    from pydantic_zarr.v3 import AnyArraySpec, AnyBaseGroupSpec

ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None

CONSOLIDATED: dict[str, Any] = {"kind": "inline", "must_understand": False, "metadata": {}}
"""An empty `consolidated_metadata` member, as zarr-python writes it."""

WAIVED: dict[str, Any] = {
    "name": "acme.notes",
    "configuration": {"tags": ["a", "b"]},
    "must_understand": False,
}
"""An extension object a reader that does not recognize it may ignore."""


class _AcmeConfiguration(TypedDict, closed=True):
    level: int
    must_understand: NotRequired[bool]


ACME_SCOPE = CORE.update(
    array_extra_fields=[ArrayExtraFieldDefinition(name="acme", configuration=_AcmeConfiguration)]
)
"""The core scope, plus a definition claiming an array extra field `acme`."""


def _array_document(**extra: object) -> dict[str, Any]:
    return {**v3.ArraySpec.from_shape_and_data_type((4,), "uint8").model_dump(), **extra}


def _group_document(**extra: object) -> dict[str, Any]:
    return {"zarr_format": 3, "node_type": "group", "attributes": {}, **extra}


class _StrictCoreGroupSpec(v3.GroupSpec):
    zarr_scope = CORE
    zarr_scope_strict = True


class _AcmeArraySpec(v3.ArraySpec):
    zarr_scope = ACME_SCOPE


@pytest.mark.parametrize(
    ("cls", "document", "context"),
    [
        # zarr-python's consolidated metadata, claimed by the core scope
        (v3.GroupSpec, _group_document(consolidated_metadata=CONSOLIDATED), None),
        (v3.BaseGroupSpec, _group_document(consolidated_metadata=CONSOLIDATED), CORE),
        # a strict class accepts it too, since its scope claims the key
        (_StrictCoreGroupSpec, _group_document(consolidated_metadata=CONSOLIDATED), None),
        # an unrecognized extension object that waives understanding
        (v3.GroupSpec, _group_document(notes=WAIVED), None),
        (v3.ArraySpec, _array_document(notes=WAIVED), CORE_AND_EXTENSIONS),
        (v3.ArraySpec, _array_document(notes={"name": "acme", "must_understand": False}), None),
        # an extra field the scope claims, which then need not waive understanding
        (_AcmeArraySpec, _array_document(acme={"level": 1, "must_understand": True}), None),
        # the same extra fields given as the `extra_fields` mapping
        (v3.ArraySpec, _array_document(extra_fields={"notes": WAIVED}), None),
        # v2 array metadata admits any other key
        (
            v2.ArraySpec,
            {"shape": (4,), "chunks": (4,), "dtype": "|u1", "notes": [1, {"a": None}]},
            None,
        ),
    ],
    ids=str,
)
def test_extra_fields_accepted(cls: type[Any], document: dict[str, Any], context: object) -> None:
    """
    A document holding extra fields the spec permits is accepted, keeps them
    in `extra_fields`, and writes them back as top-level members through
    every operation that writes the document.
    """
    spec = cls.model_validate(document, context=context)
    given = document.get("extra_fields") or {
        key: value for key, value in document.items() if key not in cls.model_fields
    }
    assert spec.extra_fields == {key: tuplify_json(value) for key, value in given.items()}
    dumped = spec.model_dump()
    assert "extra_fields" not in dumped
    assert {key: dumped[key] for key in given} == spec.extra_fields
    assert cls.model_validate(dumped, context=context) == spec
    assert cls.model_validate_json(spec.model_dump_json(), context=context) == spec
    assert spec.update(attributes={"x": 1}).extra_fields == spec.extra_fields
    assert spec.update(extra_fields={}).extra_fields == {}
    assert spec.like(spec)
    assert not spec.like(spec.update(extra_fields={}))
    if isinstance(spec, v3.GroupSpec):
        flat = v3.to_flat(spec)
        assert flat[""].extra_fields == spec.extra_fields
        assert v3.from_flat(flat).extra_fields == spec.extra_fields


@pytest.mark.skipif(
    not ZARR_AVAILABLE or ZARR_PYTHON_VERSION < Version("3.1.0"),
    reason="zarr-python stores extra fields on arrays from 3.1",
)
@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part")
def test_extra_fields_round_trip_zarr() -> None:
    """
    `to_zarr` writes a v3 node's extra fields, and `from_zarr` reads them
    back, where zarr-python stores them (arrays from 3.1, groups from 3.4).
    `from_zarr` leaves out a group's `consolidated_metadata`, which zarr-python
    writes when it consolidates the hierarchy below the group.
    """
    import zarr

    store = zarr.storage.MemoryStore()
    array: AnyArraySpec = v3.ArraySpec.model_validate(_array_document(notes=WAIVED))
    written_array = array.to_zarr(store, "a")
    assert v3.ArraySpec.from_zarr(written_array) == array
    assert array.like(written_array)
    if ZARR_PYTHON_VERSION < Version("3.4.0"):
        return
    group = v3.GroupSpec.model_validate({**_group_document(notes=WAIVED), "members": {"a": array}})
    written = group.to_zarr(store, "g")
    assert v3.GroupSpec.from_zarr(written) == group
    assert group.like(written)
    # an identical spec written again finds its own node in place
    assert group.to_zarr(store, "g").metadata == written.metadata
    base: AnyBaseGroupSpec = v3.BaseGroupSpec.from_zarr(written)
    assert base.extra_fields == group.extra_fields
    zarr.consolidate_metadata(store, "g")
    assert v3.GroupSpec.from_zarr(zarr.open_group(store, path="g")) == group


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_extra_fields_v2_array_not_stored() -> None:
    """
    zarr-python neither writes nor reads a v2 array's extra fields (other
    keys "SHOULD NOT be present"), so `to_zarr` writes the array without
    them, and finds it in place when written again.
    """
    import zarr

    store = zarr.storage.MemoryStore()
    spec = v2.ArraySpec(shape=(4,), chunks=(4,), dtype="|u1", extra_fields={"notes": 1})
    written = spec.to_zarr(store, "a")
    assert v2.ArraySpec.from_zarr(written) == spec.update(extra_fields={})
    assert spec.to_zarr(store, "a").metadata == written.metadata


@pytest.mark.parametrize(
    "value", [{"name": "acme"}, {"name": "acme", "must_understand": True}], ids=str
)
def test_extra_field_must_understand_unrecognized(value: dict[str, Any]) -> None:
    """
    An extra field no definition in the scope recognizes, and that does not
    set `"must_understand": false` (absent means true), is refused: the spec
    requires a reader to refuse the document.
    """
    with pytest.raises(ValidationError, match='notes: no array extra field.*"must_understand"'):
        v3.ArraySpec.model_validate(_array_document(notes=value))


@pytest.mark.parametrize("value", [1, "acme", ["acme"], None], ids=str)
def test_extra_field_not_an_object(value: object) -> None:
    """
    An unrecognized extra field that is not an object is refused, which is
    also what a misspelled field name meets.
    """
    with pytest.raises(ValidationError, match="dimensions_names: not a field of this spec"):
        v3.ArraySpec.model_validate(_array_document(dimensions_names=value))


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        ({"must_understand": False}, r"notes\.name: missing required key"),
        ({"name": "Not/A/Name", "must_understand": False}, r"notes\.name: expected an extension"),
        ({"name": "acme", "configuration": 1, "must_understand": False}, r"notes\.configuration"),
        ({"name": "acme", "kind": "x", "must_understand": False}, r"notes\.kind: unexpected key"),
    ],
    ids=str,
)
def test_extra_field_not_an_extension_object(value: dict[str, Any], problem: str) -> None:
    """
    An unrecognized extra field that waives understanding is still refused
    unless it is an extension object: a `name` that is an extension name, an
    optional `configuration` object, and no other members.
    """
    with pytest.raises(ValidationError, match=problem):
        v3.GroupSpec.model_validate(_group_document(notes=value))


def test_consolidated_metadata_on_array() -> None:
    """
    `consolidated_metadata` is exempt from the extension-object rules on a
    group only: on an array no definition claims it, and it has no `name`.
    """
    with pytest.raises(ValidationError, match=r"consolidated_metadata\.name: missing"):
        v3.ArraySpec.model_validate(_array_document(consolidated_metadata=CONSOLIDATED))


@pytest.mark.parametrize(
    "value", [None, {**CONSOLIDATED, "kind": "elsewhere"}, {"kind": "inline"}], ids=str
)
def test_claimed_extra_field_refused_by_definition(value: object) -> None:
    """
    An extra field a definition in the scope claims is judged by that
    definition: `consolidated_metadata` must be the object the spec gives.
    """
    with pytest.raises(ValidationError, match="Invalid Zarr V3 group metadata"):
        v3.GroupSpec.model_validate(_group_document(consolidated_metadata=value))


def test_strict_refuses_unclaimed_extra_field() -> None:
    """
    A class with `zarr_scope_strict` set refuses an extra field nothing in
    its scope claims, even one that waives understanding.
    """
    with pytest.raises(ValidationError, match="notes: no group extra field in the scope claims"):
        _StrictCoreGroupSpec.model_validate(_group_document(notes=WAIVED))


def test_empty_scope_refuses_consolidated_metadata() -> None:
    """
    In a scope that does not claim `consolidated_metadata`, it is an
    unrecognized extra field, and without a `name` not an extension object.
    """
    with pytest.raises(ValidationError, match=r"consolidated_metadata\.name"):
        v3.GroupSpec.model_validate(
            _group_document(consolidated_metadata=CONSOLIDATED), context=ZarrV3Context()
        )


def test_extra_field_given_twice() -> None:
    """An extra field given both as a member and in `extra_fields` is refused."""
    with pytest.raises(ValidationError, match="given both as members and in `extra_fields`"):
        v3.ArraySpec.model_validate(_array_document(notes=WAIVED, extra_fields={"notes": WAIVED}))


def test_extra_field_named_after_a_field() -> None:
    """An `extra_fields` key naming a declared field is refused."""
    with pytest.raises(ValidationError, match=r"\['shape'\] name fields of this spec"):
        v3.ArraySpec.model_validate(_array_document(extra_fields={"shape": WAIVED}))


def test_extra_field_not_json() -> None:
    """An extra field's value must be JSON."""
    with pytest.raises(ValidationError, match=r"extra_fields\.notes"):
        v2.ArraySpec.model_validate(
            {"shape": (4,), "chunks": (4,), "dtype": "|u1", "notes": object()}
        )


def test_v2_group_extra_field() -> None:
    """Zarr V2 group metadata admits no other keys ("Other keys MUST NOT be present")."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        v2.GroupSpec.model_validate({"zarr_format": 2, "attributes": {}, "notes": WAIVED})


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_to_zarr_refuses_consolidated_metadata() -> None:
    """
    `to_zarr` refuses a group holding `consolidated_metadata` before writing
    anything: written as held, it could list members other than the ones
    written.
    """
    import zarr

    store = zarr.storage.MemoryStore()
    spec = v3.GroupSpec.model_validate(_group_document(consolidated_metadata=CONSOLIDATED))
    with pytest.raises(ValueError, match="`consolidated_metadata`, which `to_zarr` does not write"):
        spec.to_zarr(store, "g")
    assert not list(store._store_dict)


@pytest.mark.skipif(
    not ZARR_AVAILABLE or ZARR_PYTHON_VERSION >= Version("3.4.0"),
    reason="requires a zarr-python that cannot store extra fields on a group",
)
def test_to_zarr_group_extra_fields_unsupported() -> None:
    """zarr-python before 3.4 cannot store a group's extra fields, so `to_zarr` refuses them."""
    import zarr

    spec = v3.GroupSpec.model_validate(_group_document(notes=WAIVED))
    with pytest.raises(ValueError, match="zarr-python 3.4 and later can"):
        spec.to_zarr(zarr.storage.MemoryStore(), "g")


@pytest.mark.skipif(
    not ZARR_AVAILABLE or ZARR_PYTHON_VERSION >= Version("3.1.0"),
    reason="requires a zarr-python that cannot store extra fields on an array",
)
def test_to_zarr_array_extra_fields_unsupported() -> None:
    """zarr-python before 3.1 cannot store an array's extra fields, so `to_zarr` refuses them."""
    import zarr

    spec = v3.ArraySpec.model_validate(_array_document(notes=WAIVED))
    with pytest.raises(ValueError, match="zarr-python 3.1 and later can"):
        spec.to_zarr(zarr.storage.MemoryStore(), "a")
