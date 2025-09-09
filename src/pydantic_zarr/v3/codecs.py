from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, PrivateAttr, field_validator, model_serializer


class Codec(BaseModel):
    """
    Base class for codec models.
    """

    name: str
    configuration: BaseModel
    _codec_type: Literal["array-array", "array-bytes", "bytes-bytes"] = PrivateAttr()


class BloscConfiguration(BaseModel):
    """
    Configuration for blosc codec.
    """

    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
    clevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"]
    typesize: PositiveInt
    blocksize: int


class Blosc(Codec):
    """
    Blosc codec.
    """

    name: Literal["blosc"] = "blosc"
    configuration: BloscConfiguration
    _codec_type: Literal["bytes-bytes"] = "bytes-bytes"


class BytesConfig(BaseModel):
    """
    Configuration for bytes codec.
    """

    endian: Literal["big", "little"] | None = None

    @model_serializer
    def ser_model(self) -> dict[str, Any]:
        if self.endian is None:
            return {}
        else:
            return super().model_dump()


class Bytes(Codec):
    """
    Bytes codec.
    """

    name: Literal["bytes"] = "bytes"
    configuration: BytesConfig
    _codec_type: Literal["array-bytes"] = "array-bytes"


class CRC32CConfig(BaseModel):
    """
    Configuration for crc32c codec.
    """


class CRC32C(Codec):
    """
    CRC32C codec.
    """

    name: Literal["crc32c"] = "crc32c"
    configuration: CRC32CConfig = Field(default=CRC32CConfig())

    _codec_type: Literal["bytes-bytes"] = "bytes-bytes"


class GzipConfig(BaseModel):
    """
    Configuration for gzip codec.
    """

    level: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Gzip(Codec):
    """
    Gzip codec.
    """

    name: Literal["gzip"] = "gzip"
    configuration: GzipConfig

    _codec_type: Literal["bytes-bytes"] = "bytes-bytes"


class ShardingConfig(BaseModel):
    """
    Configuration for sharding codec.
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    # Default is recommended in the specification
    index_codecs: tuple[Codec, ...] = Field(
        default=(Bytes(configuration=BytesConfig(endian="little")), CRC32C())
    )
    index_location: Literal["start", "end"] = "end"

    @field_validator("codecs", "index_codecs")
    @classmethod
    def check_single_array_bytes_codec(cls, codecs: tuple[Codec, ...]) -> tuple[Codec, ...]:
        if sum([(codec._codec_type == "array-bytes") for codec in codecs]) != 1:
            raise ValueError("Codec list must contain exactly one array-bytes codec")
        return codecs


class Sharding(Codec):
    """
    Sharding codec.
    """

    name: Literal["sharding_indexed"] = "sharding_indexed"
    configuration: ShardingConfig
    _codec_type: Literal["bytes-bytes"] = "bytes-bytes"


class TransposeConfig(BaseModel):
    """
    Configuration for transpose codec.
    """

    order: tuple[int, ...]

    @field_validator("order")
    @classmethod
    def check_order(cls, order: tuple[int, ...]) -> tuple[int, ...]:
        if set(range(len(order))) != set(order):
            raise ValueError("order must be a permutation of positive integers starting from 0")
        return order


class Transpose(Codec):
    """
    Transpose codec.
    """

    name: Literal["transpose"] = "transpose"
    configuration: TransposeConfig
    _codec_type: Literal["array-array"] = "array-array"
