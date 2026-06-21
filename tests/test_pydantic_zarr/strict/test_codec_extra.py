from pydantic_zarr.strict.v3.codec.cast_value import cast_value
from pydantic_zarr.strict.v3.codec.cast_value import dtype_out as cast_dtype_out
from pydantic_zarr.strict.v3.codec.cast_value import kind as cast_kind
from pydantic_zarr.strict.v3.codec.scale_offset import kind as so_kind
from pydantic_zarr.strict.v3.codec.scale_offset import scale_offset


def test_scale_offset_builds() -> None:
    m = scale_offset(scale=2.0, offset=1.0)
    assert m["configuration"] == {"scale": 2.0, "offset": 1.0}
    assert so_kind == "array_array"


def test_cast_value_dtype_out_is_target() -> None:
    m = cast_value("float64")
    assert m["configuration"]["data_type"] == "float64"
    assert cast_dtype_out(m, "int32") == "float64"  # cast changes the flowing dtype
    assert cast_kind == "array_array"
