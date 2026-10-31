"""
`AttributesBase` and the experimental v3 `NodeSpec` keep undeclared keys apart
from their fields rather than as pydantic extras, so their openness does not
reach the TypedDicts they hold: each nested TypedDict says its own.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from pydantic import ConfigDict, Field, ValidationError
from typing_extensions import TypedDict

from pydantic_zarr.core import MISSING, AttributesBase, _declared_keys, get_attribute
from pydantic_zarr.experimental.v3 import ArraySpec as ExperimentalArraySpec
from pydantic_zarr.experimental.v3 import GroupSpec as ExperimentalGroupSpec


class Closed(TypedDict, closed=True):
    a: int


class Plain(TypedDict):
    a: int


class Open(TypedDict, extra_items=Any):
    a: int


# Each model holding a closed TypedDict is built with warnings as errors: before
# this fix, pydantic >= 2.14 warned that the inherited `extra="allow"` was ignored.
with warnings.catch_warnings():
    warnings.simplefilter("error")

    class Attrs(AttributesBase):
        fill: float | MISSING = Field(default=MISSING, alias="_FillValue")
        closed: Closed | MISSING = MISSING
        plain: Plain | MISSING = MISSING
        open: Open | MISSING = MISSING

    class FrozenAttrs(AttributesBase):
        model_config = ConfigDict(frozen=True)

        fill: float | MISSING = Field(default=MISSING, alias="_FillValue")

    class IgnoringAttrs(Attrs):
        model_config = ConfigDict(extra="ignore")

    class ByAliasAttrs(Attrs):
        model_config = ConfigDict(populate_by_name=False, validate_by_name=False)

    class ClosedAttrs(AttributesBase):
        model_config = ConfigDict(extra="forbid")

        fill: float | MISSING = Field(default=MISSING, alias="_FillValue")
        closed: Closed | MISSING = MISSING

    class ClosedAttributesGroup(ExperimentalGroupSpec):
        attributes: Closed  # type: ignore[assignment]


_DOCUMENT = {
    "_FillValue": 1.5,
    "closed": {"a": 1},
    "plain": {"a": 1, "zz": 2},
    "open": {"a": 1, "zz": 2},
    "proj:code": "EPSG:32632",
    # an attribute key is any string, the container's own name included
    "extra_fields": {"a": 1},
}


def test_attributes_keep_unknown_keys_and_nested_typeddicts_say_their_own_openness() -> None:
    model = Attrs.model_validate(_DOCUMENT)
    # An aliased key is a field; an undeclared key, whatever its name, is an extra field.
    assert model.extra_fields == {"proj:code": "EPSG:32632", "extra_fields": {"a": 1}}
    assert get_attribute(model, "proj:code") == "EPSG:32632"
    assert model.model_dump() == {
        "_FillValue": 1.5,
        "closed": {"a": 1},
        # a plain TypedDict drops unknown keys, as it does under any model
        "plain": {"a": 1},
        "open": {"a": 1, "zz": 2},
        "proj:code": "EPSG:32632",
        "extra_fields": {"a": 1},
    }
    assert Attrs.model_validate_json(model.model_dump_json()) == model
    # A subclass that ignores unknown keys keeps none.
    assert IgnoringAttrs.model_validate(_DOCUMENT).extra_fields == {}


def test_attributes_dump_filters_name_extra_fields_by_key() -> None:
    model = Attrs.model_validate(_DOCUMENT)
    assert "proj:code" not in model.model_dump(exclude={"proj:code"})
    assert '"proj:code"' not in model.model_dump_json(exclude={"proj:code": True})
    # `include` names declared fields by field name, as pydantic does
    assert model.model_dump(include={"fill", "proj:code"}) == {
        "_FillValue": 1.5,
        "proj:code": "EPSG:32632",
    }
    assert Attrs(**{"proj:code": None}).model_dump(exclude_none=True) == {}


def test_attributes_copy_keeps_undeclared_update_keys() -> None:
    model = Attrs.model_validate(_DOCUMENT)
    copied = model.model_copy(update={"units": "m", "fill": 2.5})
    assert copied.fill == 2.5
    assert copied.extra_fields == {**model.extra_fields, "units": "m"}
    assert copied.model_dump()["units"] == "m"
    # the copy does not share its extra fields with the original
    assert "units" not in model.extra_fields


def test_frozen_attributes_hash_by_their_fields() -> None:
    document = {"_FillValue": 1.5, "proj:code": "EPSG:32632", "extra_fields": {"a": 1}}
    a = FrozenAttrs.model_validate(document)
    b = FrozenAttrs.model_validate(document)
    assert a == b
    assert hash(a) == hash(b)
    assert FrozenAttrs.model_validate({**document, "proj:code": "other"}) != a


def test_declared_keys_are_the_keys_pydantic_reads_into_a_field() -> None:
    assert _declared_keys(Attrs) == {"_FillValue", "fill", "closed", "plain", "open"}
    assert _declared_keys(ByAliasAttrs) == {"_FillValue", "closed", "plain", "open"}
    # computed once per class, not on every validation
    assert _declared_keys(Attrs) is _declared_keys(Attrs)


def test_experimental_v3_specs_hold_closed_attributes_and_keep_metadata_members() -> None:
    document = {"attributes": {"a": 1}, "members": {}, "foo": {"must_understand": False}}
    group = ClosedAttributesGroup.model_validate(document)
    assert group.attributes == {"a": 1}
    assert group.extra_fields == {"foo": {"must_understand": False}}
    assert get_attribute(group, "foo") == {"must_understand": False}
    assert group.model_dump()["foo"] == {"must_understand": False}
    assert "foo" not in group.model_dump(exclude={"foo"})
    other = ClosedAttributesGroup.model_validate(
        {**document, "foo": {"must_understand": False, "x": 1}}
    )
    assert not group.like(other)
    assert group.like(other, exclude={"foo"})
    array = ExperimentalArraySpec.from_array(
        np.arange(4),
        attributes={},
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}, "must_understand": False},
        ),
    )
    # The spec's own metadata objects keep members they do not declare.
    assert array.model_dump()["codecs"][0]["must_understand"] is False
    assert array.model_copy(deep=True) == array


def test_error_closed_attributes_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="proj:code"):
        ClosedAttrs.model_validate({"proj:code": "EPSG:32632"})


def test_error_attributes_field_name_where_names_are_not_read() -> None:
    # Where pydantic reads a field only by its alias, its name is not a field
    # key, and an extra field under that name would be dumped over it.
    with pytest.raises(ValidationError, match=r"\['fill'\] name fields of this model"):
        ByAliasAttrs.model_validate({"fill": 1.5})


def test_error_attributes_setting_extra_allow() -> None:
    # `allow` would reach the TypedDicts the model holds, which is what keeping
    # extra fields apart avoids.
    with pytest.raises(TypeError, match="leave `extra` unset"):

        class AllowingAttrs(AttributesBase):
            model_config = ConfigDict(extra="allow")


def test_error_attributes_assigning_an_undeclared_key() -> None:
    model = Attrs()
    with pytest.raises(ValueError, match="no field"):
        model.units = "m"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "model",
    [
        Attrs(**{"proj:code": "EPSG:32632"}),
        ExperimentalGroupSpec.model_validate(
            {"attributes": {}, "members": {}, "foo": {"must_understand": False}}
        ),
    ],
    ids=["attributes", "experimental-group"],
)
def test_error_extra_fields_are_read_only(model: Attrs | ExperimentalGroupSpec) -> None:
    with pytest.raises(TypeError):
        model.extra_fields["foo"] = {"must_understand": True}  # type: ignore[index]
