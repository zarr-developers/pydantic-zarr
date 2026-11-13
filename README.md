# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

[Pydantic](https://docs.pydantic.dev/latest/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

```sh
pip install -U pydantic-zarr
```

`pydantic-zarr` models Zarr metadata documents and does no IO; [zarr-python](https://github.com/zarr-developers/zarr-python) moves the documents in and out of stores, as the [usage guide](https://pydantic-zarr.readthedocs.io/en/latest/usage_zarr_v3/#reading-and-writing-with-zarr-python) shows.

## Getting help

- Docs: see the [documentation](https://pydantic-zarr.readthedocs.io/) for detailed information about this project.
- Chat: We use [Zulip]([https://ossci.zulipchat.com/#narrow/channel/423692-Zarr](https://ossci.zulipchat.com/#narrow/channel/649449-Pydantic-Zarr)) for project-related chat.

## Example

```python
import json

import zarr
from zarr.core.buffer import default_buffer_prototype

from pydantic_zarr.v2 import GroupSpec

group = zarr.create_group(store={}, path='foo', zarr_format=2)
array = zarr.create_array(
    store=group.store, name='foo/bar', shape=10, dtype='uint8', zarr_format=2
)
array.attrs.put({'metadata': 'hello'})

# collect the metadata documents zarr-python stored, keyed as the store keys them
prototype = default_buffer_prototype()
documents = {
    f"{path}/{name}" if path else name: json.loads(buffer.to_bytes())
    for path, node in [("", group), *group.members(max_depth=None)]
    for name, buffer in node.metadata.to_buffer_dict(prototype).items()
}

# this is a pydantic model
spec = GroupSpec.from_stored_metadata_json(documents)
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
            'compressor': {'id': 'zstd', 'level': 0},
        }
    },
}
"""
```

## Development

Development tasks are driven by [`just`](https://just.systems) — run
`just --list` for the available recipes — with environments managed by
[hatch](https://hatch.pypa.io). See the
[contributing guide](https://pydantic-zarr.readthedocs.io/en/latest/contributing/)
for setup, testing, and changelog conventions.

## History

This project was developed at [HHMI / Janelia Research Campus](https://www.janelia.org/). It was originally written by Davis Bennett to solve problems he encountered while working on the [Cellmap Project team](https://www.janelia.org/project-team/cellmap/members). In December of 2024 this project was migrated from the [`janelia-cellmap`](https://github.com/janelia-cellmap) github organization to [`zarr-developers`](https://github.com/zarr-developers) organization.
