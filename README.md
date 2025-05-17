# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## ⚠️ Disclaimer ⚠️
This project is under flux -- I want to add [zarr version 3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) support to this project, but the [reference python implementation](https://github.com/zarr-developers/zarr-python) doesn't support version 3 yet. As the ecosystem evolves things will break so be advised!

## Installation

`pip install -U pydantic-zarr`

## Getting help

- Docs: see the [documentation](https://zarr.dev/pydantic-zarr/) for detailed information about this project.
- Chat: We use [Zulip](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr) for project-related chat.

## Example

```python
import zarr
from pydantic_zarr import GroupSpec

group = zarr.group(path='foo')
array = zarr.create(store = group.store, path='foo/bar', shape=10, dtype='uint8')
array.attrs.put({'metadata': 'hello'})

# this is a pydantic model
spec = GroupSpec.from_zarr(group)
print(spec.model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {},
    'members': {
        'bar': {
            'zarr_format': 2,
            'attributes': {'metadata': 'hello'},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '|u1',
            'fill_value': 0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': {
                'id': 'blosc',
                'cname': 'lz4',
                'clevel': 5,
                'shuffle': 1,
                'blocksize': 0,
            },
        }
    },
}
"""
```
## History
This project was developed at [HHMI / Janelia Research Campus](https://www.janelia.org/). It was originally written by Davis Bennett to solve problems he encountered while working on the [Cellmap Project team](https://www.janelia.org/project-team/cellmap/members). In December of 2024 this project was migrated from the [`janelia-cellmap`](https://github.com/janelia-cellmap) github organization to [`zarr-developers`](https://github.com/zarr-developers) organization.
