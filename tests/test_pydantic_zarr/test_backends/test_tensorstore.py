import numpy as np
import pytest

import pydantic_zarr.v2 as v2
from pydantic_zarr.backends._tensorstore import create_array
from pydantic_zarr.backends._tensorstore.models import FileDriver, KVStore, MemoryDriver


def test_fake():
    pass


@pytest.fixture
def kvstore(request, tmp_path):
    if request.param == "file":
        return FileDriver(path=str(tmp_path))
    elif request.param == "memory":
        return MemoryDriver(path="foo")
    raise ValueError(f"Invalid request: {request.param}")


@pytest.mark.parametrize("driver", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
async def test_create_array(driver: KVStore, dtype: str, shape: tuple[int, ...]) -> None:
    template = np.zeros(shape, dtype=dtype)
    model = v2.ArrayMetadataSpec.from_array(template)
    ts = await create_array(model, kvstore=driver, open=False, create=True, delete_existing=True)
    assert ts.shape == shape
    assert ts.dtype.numpy_dtype == np.dtype(dtype)
    assert ts.schema.chunk_layout.read_chunk.shape == model.chunks
    assert ts.schema.chunk_layout.write_chunk.shape == model.chunks
