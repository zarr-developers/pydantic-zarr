import pytest

from pydantic_zarr.core import ensure_member_name


@pytest.mark.parametrize("data", ["/", "///", "a/b/", "a/b/vc"])
def test_parse_str_no_path(data: str) -> None:
    with pytest.raises(ValueError, match='Strings containing "/" are invalid.'):
        ensure_member_name(data)
