# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

[Pydantic](https://docs.pydantic.dev/latest/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

```sh
pip install -U pydantic-zarr
# or, with zarr i/o support
pip install -U "pydantic-zarr[zarr]"
```

## Getting help

- Docs: see the [documentation](https://pydantic-zarr.readthedocs.io/) for detailed information about this project.
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
