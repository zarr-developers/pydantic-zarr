from pydantic_zarr.strict.v3._cross_field import check_array_consistency


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
