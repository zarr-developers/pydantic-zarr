from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.strict.v3._registry import config_required
from pydantic_zarr.strict.v3.chunk_grid import GRIDS
from pydantic_zarr.strict.v3.chunk_key_encoding import KEY_ENCODINGS
from pydantic_zarr.strict.v3.codec import CODECS, _CoreCodec, _ExtraCodec

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


_CKE_FACTS = {"default": False, "v2": False}  # config_required


def test_key_encodings_registry_complete() -> None:
    assert set(KEY_ENCODINGS) == set(_CKE_FACTS)


def test_key_encoding_spec_facts_match_table() -> None:
    for name, cfg_req in _CKE_FACTS.items():
        spec = KEY_ENCODINGS[name]
        assert spec.name == name
        assert config_required(spec.metadata_type) is cfg_req
        assert hasattr(spec.metadata_type, "__optional_keys__")


# codec name -> (in_core, config_required)
_UNION_FACTS = {
    "bytes": (True, False),
    "crc32c": (True, False),
    "gzip": (True, True),
    "zstd": (True, True),
    "blosc": (True, True),
    "transpose": (True, True),
    "sharding_indexed": (True, True),
    "scale_offset": (False, False),
    "cast_value": (False, True),
}


def _accepts(union: object, value: object) -> bool:
    try:
        TypeAdapter(union).validate_python(value)
    except ValidationError:
        return False
    else:
        return True


def test_bare_string_accepted_iff_config_optional() -> None:
    # A bare codec name is accepted by a union IFF that codec is in the family
    # AND its configuration is optional (config_required is False).
    for name, (in_core, cfg_req) in _UNION_FACTS.items():
        bare_ok = not cfg_req
        # Extra includes every codec; bare acceptance tracks config-optionality.
        assert _accepts(_ExtraCodec, name) is bare_ok, f"extra bare {name}"
        # Core includes only core codecs; bare acceptance is (in_core AND config-optional).
        assert _accepts(_CoreCodec, name) is (in_core and bare_ok), f"core bare {name}"


def test_object_form_always_accepted_for_member_codecs() -> None:
    # The object form (a minimal valid dict) is accepted for every codec the family contains.
    # Build a minimal object per codec from its builder so required config is present.
    from pydantic_zarr.strict.v3.codec import (
        blosc_codec,
        bytes_codec,
        cast_value_codec,
        crc32c_codec,
        gzip_codec,
        scale_offset_codec,
        sharding_indexed_codec,
        transpose_codec,
        zstd_codec,
    )

    core_objs = [
        bytes_codec(),
        crc32c_codec(),
        gzip_codec(5),
        zstd_codec(3),
        blosc_codec("zstd", 5, "shuffle", 0),
        transpose_codec((1, 0)),
        sharding_indexed_codec((4, 4)),
    ]
    for obj in core_objs:
        assert _accepts(_CoreCodec, obj), obj["name"]
        assert _accepts(_ExtraCodec, obj), obj["name"]
    for obj in (scale_offset_codec(), cast_value_codec("int32")):
        assert _accepts(_ExtraCodec, obj), obj["name"]
