from pydantic_zarr.strict.v3._registry import config_required
from pydantic_zarr.strict.v3.chunk_grid import GRIDS
from pydantic_zarr.strict.v3.codec import CODECS

# Ground-truth table from the spec.
# Columns: (kind, has_dtype_dependent_config, config_required)
# Note: cast_value has configuration as a required key in CastValueCodecObject,
# so config_required=True (the brief's table had False here, which was a typo).
_CODEC_FACTS = {
    "bytes": ("array_bytes", False, False),
    "crc32c": ("bytes_bytes", False, False),
    "gzip": ("bytes_bytes", False, True),
    "zstd": ("bytes_bytes", False, True),
    "blosc": ("bytes_bytes", False, True),
    "transpose": ("array_array", False, True),
    "sharding_indexed": ("array_bytes", False, True),
    "scale_offset": ("array_array", False, False),
    "cast_value": ("array_array", True, True),
}


def test_codecs_registry_complete() -> None:
    assert set(CODECS) == set(_CODEC_FACTS)


def test_codec_spec_facts_match_table() -> None:
    for name, (kind, has_dtype_dep, cfg_req) in _CODEC_FACTS.items():
        spec = CODECS[name]
        assert spec.name == name
        assert spec.kind == kind
        assert spec.has_dtype_dependent_config is has_dtype_dep
        # config_required derived from the metadata type must match the table
        assert config_required(spec.metadata_type) is cfg_req


def test_codec_metadata_type_is_typeddict() -> None:
    for spec in CODECS.values():
        # *Object TypedDicts expose __optional_keys__; unions do not
        assert hasattr(spec.metadata_type, "__optional_keys__")


_GRID_FACTS = {"regular": True, "rectilinear": True}  # config_required


def test_grids_registry_complete() -> None:
    assert set(GRIDS) == set(_GRID_FACTS)


def test_grid_spec_facts_match_table() -> None:
    for name, cfg_req in _GRID_FACTS.items():
        spec = GRIDS[name]
        assert spec.name == name
        assert config_required(spec.metadata_type) is cfg_req
        assert hasattr(spec.metadata_type, "__optional_keys__")
