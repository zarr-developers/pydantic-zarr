# Usage (Zarr V2)

## Reading and writing a zarr hierarchy

### Reading

The `GroupSpec` and `ArraySpec` classes represent Zarr v2 groups and arrays, respectively. To create an instance of a `GroupSpec` or `ArraySpec` from an existing Zarr group or array, pass the Zarr group / array to the `.from_zarr` method defined on the `GroupSpec` / `ArraySpec` classes. This will result in a `pydantic-zarr` model of the Zarr object.

> By default `GroupSpec.from_zarr(zarr_group)` will traverse the entire hierarchy under `zarr_group`. This can be extremely slow if used on an extensive Zarr group on high latency storage. To limit the depth of traversal to a specific depth, use the `depth` keyword argument, e.g. `GroupSpec.from_zarr(zarr_group, depth=1)`

Note that `from_zarr` will _not_ read the data inside an array.

### Writing

To write a hierarchy to some zarr-compatible storage backend, `GroupSpec` and `ArraySpec` have `to_zarr` methods that take a Zarr store and a path and return a Zarr array or group created in the store at the given path.

Note that `to_zarr` will _not_ write any array data. You have to do this separately.

```python
from zarr import create_array, create_group

from pydantic_zarr.v2 import GroupSpec

# create an in-memory Zarr group + array with attributes
grp = create_group(store={}, path='foo', zarr_format=2)
grp.attrs.put({'group_metadata': 10})
arr = create_array(
    name='foo/bar', store=grp.store, shape=(10,), dtype="f8", compressors=None, zarr_format=2
)
arr.attrs.put({'array_metadata': True})

spec = GroupSpec.from_zarr(grp)
print(spec.model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {'group_metadata': 10},
    'members': {
        'bar': {
            'zarr_format': 2,
            'attributes': {'array_metadata': True},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '<f8',
            'fill_value': 0.0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': None,
        }
    },
}
"""

# convert the spec to a dict so we can modify it
spec_dict2 = spec.model_dump()

# change the group metadata
spec_dict2['attributes'] = {'a': 100, 'b': 'metadata'}

# change the properties of an array member
spec_dict2['members']['bar']['shape'] = (100,)

# serialize the spec to the store
group2 = GroupSpec(**spec_dict2).to_zarr(grp.store, path='foo2')

print(dict(group2.attrs))
#> {'a': 100, 'b': 'metadata'}

print(dict(group2['bar'].attrs))
#> {'array_metadata': True}
```

### Creating from an array

The `ArraySpec` class has a `from_array` static method that takes an array-like object and returns an `ArraySpec` with `shape` and `dtype` fields matching those of the array-like object.

```python
import numpy as np

from pydantic_zarr.v2 import ArraySpec

print(ArraySpec.from_array(np.arange(10)).model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {},
    'shape': (10,),
    'chunks': (10,),
    'dtype': '<i8',
    'fill_value': 0,
    'order': 'C',
    'filters': None,
    'dimension_separator': '/',
    'compressor': None,
}
"""
```

### Flattening and unflattening Zarr hierarchies

In the previous section we built a model of a Zarr hierarchy by defining `GroupSpec` and `ArraySpec`
instances, then providing those objects as `members` to the constructor of another `GroupSpec`. In
other words, with this approach we create "child nodes" and give those nodes to the "parent node",
recursively.

Constructing deeply nested hierarchies this way can be tedious.
For this reason, `pydantic-zarr` supports an alternative representation of the Zarr
hierarchy in the form of a dictionary with `str` keys and `ArraySpec` / `GroupSpec` values, and
methods to convert to / from these dictionaries.

#### Making a `GroupSpec` object from a flat hierarchy

This example demonstrates how to create a `GroupSpec` from a `dict` representation of a Zarr hierarchy.

```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec

# other than the key representing the root path "",
# the keys must be valid paths in the Zarr storage hierarchy
# note that the `members` attribute is `None` for the `GroupSpec` instances in this `dict`.
tree = {
    "": GroupSpec(members=None, attributes={"root": True}),
    "/a": GroupSpec(members=None, attributes={"root": False}),
    "/a/b": ArraySpec(shape=(10, 10), dtype="uint8", chunks=(1, 1), attributes={}),
}

print(GroupSpec.from_flat(tree).model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {'root': True},
    'members': {
        'a': {
            'zarr_format': 2,
            'attributes': {'root': False},
            'members': {
                'b': {
                    'zarr_format': 2,
                    'attributes': {},
                    'shape': (10, 10),
                    'chunks': (1, 1),
                    'dtype': '|u1',
                    'fill_value': 0,
                    'order': 'C',
                    'filters': None,
                    'dimension_separator': '/',
                    'compressor': None,
                }
            },
        }
    },
}
"""
```

#### flattening `GroupSpec` objects

This is similar to the example above, except that we are working in reverse -- we are making the
flat `dict` from the `GroupSpec` object.

```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec

# other than the key representing the root path "",
# the keys must be valid paths in the Zarr storage hierarchy
# note that the `members` attribute is `None` for the `GroupSpec` instances in this `dict`.

a_b = ArraySpec(shape=(10, 10), dtype="uint8", chunks=(1, 1), attributes={})
a = GroupSpec(members={'b': a_b}, attributes={"root": False})
root = GroupSpec(members={'a': a}, attributes={"root": True})

print(root.to_flat())
"""
{
    '': GroupSpec(zarr_format=2, attributes={'root': True}, members=None),
    '/a': GroupSpec(zarr_format=2, attributes={'root': False}, members=None),
    '/a/b': ArraySpec(
        zarr_format=2,
        attributes={},
        shape=(10, 10),
        chunks=(1, 1),
        dtype='|u1',
        fill_value=0,
        order='C',
        filters=None,
        dimension_separator='/',
        compressor=None,
    ),
}
"""
```

#### Implicit groups

`zarr-python` supports creating Zarr arrays or groups deep in the
hierarchy without explicitly creating the intermediate groups first.
`from_flat` models this behavior. For example, `{'/a/b/c': ArraySpec(...)}` implicitly defines the existence of a groups named `a` and `b` (which is contained in `a`). `from_flat` will create the expected `GroupSpec` object from such `dict` instances.

```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec

tree = {'/a/b/c': ArraySpec(shape=(1,), dtype='uint8', chunks=(1,), attributes={})}
print(GroupSpec.from_flat(tree).model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {},
    'members': {
        'a': {
            'zarr_format': 2,
            'attributes': {},
            'members': {
                'b': {
                    'zarr_format': 2,
                    'attributes': {},
                    'members': {
                        'c': {
                            'zarr_format': 2,
                            'attributes': {},
                            'shape': (1,),
                            'chunks': (1,),
                            'dtype': '|u1',
                            'fill_value': 0,
                            'order': 'C',
                            'filters': None,
                            'dimension_separator': '/',
                            'compressor': None,
                        }
                    },
                }
            },
        }
    },
}
"""
```

## Comparing `GroupSpec` and `ArraySpec` models

`GroupSpec` and `ArraySpec` both have `like` methods that take another `GroupSpec` or `ArraySpec` as an argument and return `True` (the models are like each other) or `False` (the models are not like each other).

The `like` method works by converting both input models to `dict` via `pydantic.BaseModel.model_dump`, and comparing the `dict` representation of the models. This means that instances of two different subclasses of `GroupSpec`, which would not be considered equal according to the `==` operator, will be considered `like` if and only if they serialize to identical `dict` instances.

The `like` method takes keyword arguments `include` and `exclude`, which determine the attributes included or excluded from the model comparison. So it's possible to use `like` to check if two `ArraySpec` instances have the same `shape`, `dtype` and `chunks` by calling `array_a.like(array_b, include={'shape', 'dtype', 'chunks'})`. This is useful if you don't care about the compressor or filters and just want to ensure that you can safely write an in-memory array to a Zarr array, which depends just on the two arrays having matching `shape`, `dtype`, and `chunks` attributes.

```python
import zarr
import zarr.storage

from pydantic_zarr.v2 import ArraySpec, GroupSpec

arr_a = ArraySpec(shape=(1,), dtype='uint8', chunks=(1,), attributes={})
# make an array with a different shape
arr_b = ArraySpec(shape=(2,), dtype='uint8', chunks=(1,), attributes={})

# Returns False, because of mismatched shape
print(arr_a.like(arr_b))
#> False

# Returns True, because we exclude shape.
print(arr_a.like(arr_b, exclude={'shape'}))
#> True

# `ArraySpec.like` will convert a zarr.Array to ArraySpec
store = zarr.storage.MemoryStore()
# This is a zarr.Array
arr_a_stored = arr_a.to_zarr(store, path='arr_a')

# arr_a is like the zarr.Array version of itself
print(arr_a.like(arr_a_stored))
#> True

# Returns False, because of mismatched shape
print(arr_b.like(arr_a_stored))
#> False

# Returns True, because we exclude shape.
print(arr_b.like(arr_a_stored, exclude={'shape'}))
#> True

# The same thing, but for groups
g_a = GroupSpec(attributes={'foo': 10}, members={'a': arr_a, 'b': arr_b})
g_b = GroupSpec(attributes={'foo': 11}, members={'a': arr_a, 'b': arr_b})

# g_a is like itself
print(g_a.like(g_a))
#> True

# Returns False, because of mismatched attributes
print(g_a.like(g_b))
#> False

# Returns True, because we ignore attributes
print(g_a.like(g_b, exclude={'attributes'}))
#> True

# g_a is like its zarr.Group counterpart
print(g_a.like(g_a.to_zarr(store, path='g_a')))
#> True
```

## Creating from an array

The `ArraySpec` class has a `from_array` static method that takes an array-like object and returns an `ArraySpec` with `shape` and `dtype` fields matching those of the array-like object.

```python
import numpy as np

from pydantic_zarr.v2 import ArraySpec

print(ArraySpec.from_array(np.arange(10)).model_dump())
"""
{
    'zarr_format': 2,
    'attributes': {},
    'shape': (10,),
    'chunks': (10,),
    'dtype': '<i8',
    'fill_value': 0,
    'order': 'C',
    'filters': None,
    'dimension_separator': '/',
    'compressor': None,
}
"""
```
