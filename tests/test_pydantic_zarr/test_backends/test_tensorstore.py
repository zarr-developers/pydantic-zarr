import numpy as np
import pytest

import pydantic_zarr.v2 as v2
from pydantic_zarr.backends._tensorstore import create_array


@pytest.mark.parametrize("driver", ["memory"])
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
async def test_create_array(driver: str, dtype: str, shape: tuple[int, ...]) -> None:
    template = np.zeros(shape, dtype=dtype)
    model = v2.ArrayMetadataSpec.from_array(template)
    ts = await create_array(model, driver=driver, open=False, create=True, delete_existing=True)
    assert ts.shape == shape
    assert ts.dtype.numpy_dtype == np.dtype(dtype)
    assert ts.schema.chunk_layout.read_chunk.shape == model.chunks
    assert ts.schema.chunk_layout.write_chunk.shape == model.chunks
