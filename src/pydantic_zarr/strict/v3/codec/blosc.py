from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zarr_metadata.v3.codec.blosc import BloscCodecObject

from pydantic_zarr.strict.v3.codec._spec import CodecSpec

if TYPE_CHECKING:
    from zarr_metadata import BloscCodecMetadata
    from zarr_metadata.v3.codec.blosc import BloscCName, BloscShuffle

kind: Literal["bytes_bytes"] = "bytes_bytes"


def blosc(
    cname: BloscCName,
    clevel: int,
    shuffle: BloscShuffle,
    blocksize: int,
    typesize: int | None = None,
) -> BloscCodecMetadata:
    config: dict = {
        "cname": cname,
        "clevel": clevel,
        "shuffle": shuffle,
        "blocksize": blocksize,
    }
    if typesize is not None:
        config["typesize"] = typesize
    meta: BloscCodecMetadata = {"name": "blosc", "configuration": config}  # type: ignore[typeddict-item]
    validate_blosc(meta)
    return meta


def validate_blosc(meta: BloscCodecMetadata) -> None:
    clevel = meta["configuration"]["clevel"]
    if not (0 <= clevel <= 9):
        raise ValueError(f"blosc clevel {clevel} out of range [0, 9]")
    # cname/shuffle are Literal-typed; no runtime re-check needed.


def ndim_of(meta: BloscCodecMetadata) -> int | None:
    return None


def dtype_out(meta: BloscCodecMetadata, input_dtype: str) -> str:
    return input_dtype


SPEC = CodecSpec(
    name="blosc",
    kind="bytes_bytes",
    metadata_type=BloscCodecObject,
    has_dtype_dependent_config=False,
    validate=validate_blosc,
    ndim_of=ndim_of,
    dtype_out=dtype_out,
)
