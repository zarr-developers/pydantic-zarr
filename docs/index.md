# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

Type-safety for Zarr data.

## Overview

`pydantic-zarr` expresses data stored in the [Zarr](https://zarr.readthedocs.io/en/stable/) format with [Pydantic](https://docs.pydantic.dev/1.10/). Specifically, `pydantic-zarr` encodes Zarr groups and arrays as [Pydantic models](https://docs.pydantic.dev/1.10/usage/models/). These models are useful for formalizing the structure of Zarr hierarchies, type-checking Zarr hierarchies, and runtime validation for Zarr-based data.

```python
import zarr

from pydantic_zarr.v2 import GroupSpec

# create a Zarr group
group = zarr.create_group(store={}, path='foo', zarr_format=2)
# put an array inside the group
array = zarr.create_array(
    store=group.store, name='foo/bar', shape=10, dtype='uint8', zarr_format=2
)
array.attrs.put({'metadata': 'hello'})

# create a pydantic model to model the Zarr group
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
            'compressor': {'id': 'zstd', 'level': 0},
        }
    },
}
"""
```

More examples can be found in the [usage guide](usage_zarr_v2.md).

## Installation

`pip install -U pydantic-zarr`

### Limitations

#### No array data operations

This library only provides tools to represent the _layout_ of Zarr groups and arrays, and the structure of their attributes. `pydantic-zarr` performs no type checking or runtime validation of the multidimensional array data contained _inside_ Zarr arrays, and `pydantic-zarr` does not contain any tools for efficiently reading or writing Zarr arrays.

#### Supported Zarr versions

This library supports [version 2](https://zarr.readthedocs.io/en/stable/spec/v2.html) of the Zarr format, with partial support for [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html). Progress towards complete support for Zarr v3 is tracked by [this issue](https://github.com/d-v-b/pydantic-zarr/issues/3).

## Design

A Zarr group can be modeled as an object with two properties:

- `attributes`: A dict-like object, with keys that are strings, values that are JSON-serializable.
- `members`: A dict-like object, with keys that strings and values that are other Zarr groups, or Zarr arrays.

A Zarr array can be modeled similarly, but without the `members` property (because Zarr arrays cannot contain Zarr groups or arrays), and with a set of array-specific properties like `shape`, `dtype`, etc.

Note the use of the term "modeled": Zarr arrays are useful because they store N-dimensional array data, but `pydantic-zarr` does not treat that data as part of the "model" of a Zarr array.

In `pydantic-zarr`, Zarr groups are modeled by the `GroupSpec` class, which is a [`Pydantic model`](https://docs.pydantic.dev/latest/concepts/models/) with two fields:

- `attributes`: either a `Mapping` or a `pydantic.BaseModel`.
- `members`: either a mapping with string keys and values that must be `GroupSpec` or `ArraySpec` instances, or the value `Null`. The use of nullability is explained in its own [section](#nullable-members).

Zarr arrays are represented by the `ArraySpec` class, which has a similar `attributes` field, as well as fields for all the Zarr array properties (`dtype`, `shape`, `chunks`, etc).

`GroupSpec` and `ArraySpec` are both [generic models](https://docs.pydantic.dev/1.10/usage/models/#generic-models). `GroupSpec` takes two type parameters, the first specializing the type of `GroupSpec.attributes`, and the second specializing the type of the _values_ of `GroupSpec.members` (the keys of `GroupSpec.members` are always strings). `ArraySpec` only takes one type parameter, which specializes the type of `ArraySpec.attributes`.

Examples using this generic typing functionality can be found in the [usage guide](usage_zarr_v2.md#using-generic-types).

### Nullable `members`

When a Zarr group has no members, a `GroupSpec` model of that Zarr group will have its `members` attribute set to the empty dict `{}`. But there are scenarios where the members of a Zarr group are unknown:

- Some Zarr storage backends do not support directory listing, in which case it is possible to access a Zarr group and inspect its attributes, but impossible to discover its members. So the members of such a Zarr group are unknown.
- Traversing a deeply nested large Zarr group on high latency storage can be slow. This can be mitigated by only partially traversing the hierarchy, e.g. only inspecting the root group and N subgroups. This defines a sub-hierarchy of the full hierarchy; leaf groups of this subtree by definition did not have their members checked, and so their members are unknown.
- A Zarr hierarchy can be represented as a mapping `M` from paths to nodes (array or group). In this case, if `M["key"]` is a model of a Zarr group `G`, then `M["key/subkey"]` would encode a member of `G`. Since the key structure of the mapping `M` is doing the work of encoding the members of `G`, there is no value in `G` having a members attribute that claims anything about the members of `G`, and so `G.members` should be modeled as unknown.

To handle these cases, `pydantic-zarr` allows the `members` attribute of a `GroupSpec` to be `Null`.

## Standardization

The Zarr specifications do not define a model of the Zarr hierarchy. `pydantic-zarr` is an implementation of a particular model that can be found formalized in this [specification document](https://github.com/d-v-b/zeps/blob/zom/draft/ZEP0006.md), which has been proposed for inclusion in the Zarr specifications. You can find the discussion of that proposal in [this pull request](https://github.com/zarr-developers/zeps/pull/46).
