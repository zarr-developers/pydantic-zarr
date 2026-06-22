from pydantic_zarr.strict.v3.chunk_key_encoding import default, v2


def test_default_builder() -> None:
    assert default() == {"name": "default", "configuration": {"separator": "/"}}
    assert default(".") == {"name": "default", "configuration": {"separator": "."}}


def test_v2_builder() -> None:
    assert v2() == {"name": "v2", "configuration": {"separator": "."}}
    assert v2("/") == {"name": "v2", "configuration": {"separator": "/"}}
