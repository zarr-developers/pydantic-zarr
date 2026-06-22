from pydantic_zarr.strict.v3._pipeline import validate_pipeline

_BYTES = {"name": "bytes", "configuration": {"endian": "little"}}


def test_pipeline_valid() -> None:
    assert validate_pipeline("int32", (_BYTES,)) == []


def test_pipeline_requires_one_array_bytes() -> None:
    assert any(
        "array" in e.lower() and "bytes" in e.lower() for e in validate_pipeline("int32", ())
    )
    two = validate_pipeline("int32", (_BYTES, _BYTES))
    assert any("one" in e.lower() or "array" in e.lower() for e in two)


def test_cast_value_scalar_input_validated_against_current_dtype() -> None:
    # array is int32; cast_value input scalar "NaN" is INVALID for int32
    cast = {
        "name": "cast_value",
        "configuration": {"data_type": "float64", "scalar_map": {"encode": (("NaN", 0.0),)}},
    }
    errs = validate_pipeline("int32", (cast, _BYTES))
    assert any("scalar" in e.lower() for e in errs)


def test_cast_value_valid_scalars() -> None:
    # array is float64; input scalar "NaN" valid for float64; output 0 valid for int32 target
    cast = {
        "name": "cast_value",
        "configuration": {"data_type": "int32", "scalar_map": {"encode": (("NaN", 0),)}},
    }
    assert validate_pipeline("float64", (cast, _BYTES)) == []
