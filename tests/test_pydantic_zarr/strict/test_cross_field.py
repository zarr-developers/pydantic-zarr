import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.strict.v3._cross_field import check_array_consistency
from pydantic_zarr.v3 import AnyCoreArraySpec, AnyExtraArraySpec


def _grid(cs: tuple[int, ...]) -> dict:
    return {"name": "regular", "configuration": {"chunk_shape": cs}}


def test_chunk_grid_ndim_mismatch() -> None:
    errs = check_array_consistency(
        shape=(8, 8), chunk_grid=_grid((4,)), codecs=(), dimension_names=None
    )
    assert any("chunk_grid" in e or "ndim" in e for e in errs)


def test_dimension_names_mismatch() -> None:
    errs = check_array_consistency(
        shape=(8, 8), chunk_grid=_grid((4, 4)), codecs=(), dimension_names=("x",)
    )
    assert errs


def test_multiple_violations_collected() -> None:
    errs = check_array_consistency(
        shape=(8, 8), chunk_grid=_grid((4,)), codecs=(), dimension_names=("x",)
    )
    assert len(errs) >= 2


def test_sharding_indivisible() -> None:
    shard = {
        "name": "sharding_indexed",
        "configuration": {"chunk_shape": (3, 3), "codecs": (), "index_codecs": ()},
    }
    errs = check_array_consistency(
        shape=(8, 8), chunk_grid=_grid((8, 8)), codecs=(shard,), dimension_names=None
    )
    assert any("divide" in e.lower() for e in errs)


def test_valid_passes() -> None:
    assert (
        check_array_consistency(
            shape=(8, 8), chunk_grid=_grid((4, 4)), codecs=(), dimension_names=("y", "x")
        )
        == []
    )


# ---------------------------------------------------------------------------
# Regression: bare-string codecs must not raise TypeError
# ---------------------------------------------------------------------------

_BC = {"name": "bytes", "configuration": {"endian": "little"}}
_BASE_DOC = {
    "zarr_format": 3,
    "node_type": "array",
    "data_type": "float64",
    "shape": (4,),
    "fill_value": 0.0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "attributes": {},
}


def test_bare_string_transpose_no_typeerror_check_array_consistency() -> None:
    """Check_array_consistency must return a list (never raise TypeError) for bare-string 'transpose'."""
    result = check_array_consistency(
        shape=(4,),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        codecs=("transpose", _BC),
        dimension_names=None,
    )
    assert isinstance(result, list)


def test_bare_string_sharding_no_typeerror_check_array_consistency() -> None:
    """Check_array_consistency must return a list (never raise TypeError) for bare-string 'sharding_indexed'."""
    result = check_array_consistency(
        shape=(4,),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        codecs=("sharding_indexed",),
        dimension_names=None,
    )
    assert isinstance(result, list)


@pytest.mark.parametrize("adapter_cls", [AnyCoreArraySpec, AnyExtraArraySpec])
def test_bare_string_transpose_no_typeerror_validate(adapter_cls) -> None:
    """TypeAdapter must not raise TypeError for bare-string 'transpose' codec."""
    doc = {**_BASE_DOC, "codecs": ("transpose", _BC)}
    try:
        TypeAdapter(adapter_cls).validate_python(doc)
    except TypeError as exc:
        pytest.fail(f"Unexpected TypeError for bare-string 'transpose': {exc}")
    except ValidationError:
        pass  # clean rejection is fine


@pytest.mark.parametrize("adapter_cls", [AnyCoreArraySpec, AnyExtraArraySpec])
def test_bare_string_sharding_no_typeerror_validate(adapter_cls) -> None:
    """TypeAdapter must not raise TypeError for bare-string 'sharding_indexed' codec."""
    doc = {**_BASE_DOC, "codecs": ("sharding_indexed",)}
    try:
        TypeAdapter(adapter_cls).validate_python(doc)
    except TypeError as exc:
        pytest.fail(f"Unexpected TypeError for bare-string 'sharding_indexed': {exc}")
    except ValidationError:
        pass  # clean rejection is fine
