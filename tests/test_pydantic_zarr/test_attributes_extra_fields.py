"""
`AttributesBase` and the experimental v3 `NodeSpec` keep undeclared keys in
`extra_fields` rather than as pydantic extras, so their openness does not
reach the TypedDicts they hold: each nested TypedDict says its own.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from pydantic import ConfigDict, Field, ValidationError
from typing_extensions import TypedDict

from pydantic_zarr.core import MISSING, AttributesBase
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


def test_attributes_keep_unknown_keys_and_nested_typeddicts_say_their_own_openness() -> None:
    document = {
        "_FillValue": 1.5,
        "closed": {"a": 1},
        "plain": {"a": 1, "zz": 2},
        "open": {"a": 1, "zz": 2},
        "proj:code": "EPSG:32632",
    }
    model = Attrs.model_validate(document)
    # An aliased key is a field; an undeclared key is an extra field.
    assert model.extra_fields == {"proj:code": "EPSG:32632"}
    assert model.model_dump() == {
        "_FillValue": 1.5,
        "closed": {"a": 1},
        # a plain TypedDict drops unknown keys, as it does under any model
        "plain": {"a": 1},
        "open": {"a": 1, "zz": 2},
        "proj:code": "EPSG:32632",
    }
    assert Attrs.model_validate(model.model_dump()) == model
    # A subclass that ignores unknown keys keeps none.
    assert IgnoringAttrs.model_validate(document).extra_fields == {}


def test_experimental_v3_specs_hold_closed_attributes_and_keep_metadata_members() -> None:
    group = ClosedAttributesGroup.model_validate(
        {"attributes": {"a": 1}, "members": {}, "foo": {"must_understand": False}}
    )
    assert group.attributes == {"a": 1}
    assert group.extra_fields == {"foo": {"must_understand": False}}
    assert group.model_dump()["foo"] == {"must_understand": False}
    array = ExperimentalArraySpec.from_array(
        np.arange(4),
        attributes={},
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}, "must_understand": False},
        ),
    )
    # The spec's own metadata objects keep members they do not declare.
    assert array.model_dump()["codecs"][0]["must_understand"] is False


def test_error_closed_attributes_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="proj:code"):
        ClosedAttrs.model_validate({"proj:code": "EPSG:32632"})


def test_error_closed_attributes_refuse_extra_fields_given_directly() -> None:
    with pytest.raises(ValidationError, match="keeps no extra fields"):
        ClosedAttrs.model_validate({"extra_fields": {"proj:code": "EPSG:32632"}})


def test_error_attributes_extra_field_named_after_an_alias() -> None:
    with pytest.raises(ValidationError, match=r"\['_FillValue'\] name fields of this model"):
        Attrs.model_validate({"extra_fields": {"_FillValue": 1.5}})


def test_error_attributes_extra_field_given_twice() -> None:
    with pytest.raises(ValidationError, match="given both as members and in `extra_fields`"):
        Attrs.model_validate({"proj:code": "a", "extra_fields": {"proj:code": "b"}})


def test_error_attributes_field_name_where_names_are_not_read() -> None:
    # Where pydantic reads a field only by its alias, its name is not a field
    # key, and an extra field under that name would be dumped over it.
    with pytest.raises(ValidationError, match=r"\['fill'\] name fields of this model"):
        ByAliasAttrs.model_validate({"fill": 1.5})
