# Usage (Zarr V2)

## Reading and writing a zarr hierarchy

The `GroupSpec` and `ArraySpec` classes model Zarr v2 groups and arrays as their metadata documents. `pydantic-zarr` does no IO of its own: a spec gives the documents a store holds (`to_stored_metadata_json`) and reads such documents back (`from_stored_metadata_json`), and moving documents in and out of a store is the job of a Zarr implementation such as `zarr-python`. [Stored metadata documents](#stored-metadata-documents) describes the documents; [Reading and writing with zarr-python](#reading-and-writing-with-zarr-python) connects them to a `zarr-python` store. Neither direction reads or writes array data.

> The `to_zarr` and `from_zarr` methods, the module-level `to_zarr`, `from_zarr` and `from_zarr_flat`, and `like` given a `zarr.Array` or `zarr.Group` are deprecated and will be removed in `pydantic-zarr` 0.12. Each warns once, naming its replacement.

Reading a hierarchy `zarr-python` wrote means collecting the documents it stored. A `zarr.Group` lists the nodes under it, and each node's `metadata.to_buffer_dict` gives its documents as the store holds them (`.zarray` or `.zgroup`, and `.zattrs`):

```python
import json

import zarr
from zarr.core.buffer import default_buffer_prototype

from pydantic_zarr.v2 import GroupSpec

# create an in-memory Zarr group + array with attributes
grp = zarr.create_group(store={}, path='foo', zarr_format=2, attributes={'group_metadata': 10})
arr = grp.create_array(name='bar', shape=(10,), dtype="f8", compressors=None)
arr.attrs.put({'array_metadata': True})

# the documents zarr-python stored, keyed as the store keys them
prototype = default_buffer_prototype()
documents = {
    f"{path}/{name}" if path else name: json.loads(buffer.to_bytes())
    for path, node in [("", grp), *grp.members(max_depth=None)]
    for name, buffer in node.metadata.to_buffer_dict(prototype).items()
}
print(sorted(documents))
#> ['.zattrs', '.zgroup', 'bar/.zarray', 'bar/.zattrs']

spec = GroupSpec.from_stored_metadata_json(documents)
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

spec2 = GroupSpec(**spec_dict2)

# writing goes the other way: each node's document becomes one of zarr-python's
# metadata objects, and `zarr.create_hierarchy` writes them all under a path
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata

from pydantic_zarr.v2 import is_array_spec, to_flat

nodes = {
    f"foo2{path}": (ArrayV2Metadata if is_array_spec(node) else GroupMetadata).from_dict(
        node.to_metadata_json()
    )
    for path, node in to_flat(spec2).items()
}
written = dict(zarr.create_hierarchy(store=grp.store, nodes=nodes))

print(dict(written['foo2'].attrs))
#> {'a': 100, 'b': 'metadata'}

print(written['foo2/bar'].shape)
#> (100,)
```

Writing goes the other way from reading: each node's `to_metadata_json` document becomes one of `zarr-python`'s metadata objects, and `zarr.create_hierarchy` writes them all under a path. `to_flat` lists the nodes of a hierarchy with their paths.
[Reading and writing with zarr-python](#reading-and-writing-with-zarr-python) wraps both directions in two functions.

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
from pydantic_zarr.v2 import ArraySpec, BaseGroupSpec, GroupSpec

# other than the key representing the root path "",
# the keys must be valid paths in the Zarr storage hierarchy
# note that groups appear as `BaseGroupSpec` instances, without a `members` attribute.
tree = {
    "": BaseGroupSpec(attributes={"root": True}),
    "/a": BaseGroupSpec(attributes={"root": False}),
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
# note that groups appear as `BaseGroupSpec` instances, without a `members` attribute.

a_b = ArraySpec(shape=(10, 10), dtype="uint8", chunks=(1, 1), attributes={})
a = GroupSpec(members={'b': a_b}, attributes={"root": False})
root = GroupSpec(members={'a': a}, attributes={"root": True})

print(root.to_flat())
"""
{
    '': BaseGroupSpec(zarr_format=2, attributes={'root': True}),
    '/a': BaseGroupSpec(zarr_format=2, attributes={'root': False}),
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

## Metadata documents

`to_metadata_json` returns a node's metadata with a static type, unlike
`model_dump`, whose options change what it returns. A Zarr V2 node stores its
metadata in two documents, `.zarray` (or `.zgroup`) and `.zattrs`;
`to_metadata_json` returns them as one document, with the `.zattrs` content
under `attributes`, typed as `zarr_metadata.ZarrV2ArrayMetadataJSON` for an
array and `zarr_metadata.ZarrV2GroupMetadataJSON` for a group. A group's
members are not part of its metadata, so they are left out.

```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec

array = ArraySpec(shape=(10,), chunks=(5,), dtype="|u1", attributes={"foo": 42})
document = array.to_metadata_json()
zattrs = document.pop("attributes")
print(zattrs)
#> {'foo': 42}
print(document)
"""
{
    'zarr_format': 2,
    'shape': (10,),
    'chunks': (5,),
    'dtype': '|u1',
    'fill_value': 0,
    'order': 'C',
    'filters': None,
    'dimension_separator': '/',
    'compressor': None,
}
"""

group = GroupSpec(attributes={"bar": True}, members={"array": array})
print(group.to_metadata_json())
#> {'zarr_format': 2, 'attributes': {'bar': True}}
```

## Validation

An `ArraySpec` is validated as a whole Zarr V2 array metadata document, via
[`zarr-metadata`](https://pypi.org/project/zarr-metadata/), so a spec holds only
what a `.zarray` document can store. Among other things, its fill value must be
one its dtype accepts. Before that check, the fill value is put in its Zarr V2
JSON form: numpy scalars become Python scalars, complex numbers become
`(real, imaginary)` pairs, and non-finite floats become `"NaN"`, `"Infinity"`
or `"-Infinity"`. A spec built without a fill value gets 0 when its dtype
accepts 0, and `None` otherwise.

```python
import numpy as np
from pydantic import ValidationError

from pydantic_zarr.v2 import ArraySpec

print(ArraySpec(shape=(4,), chunks=(2,), dtype="<c8", fill_value=np.complex64(np.nan)).fill_value)
#> ('NaN', 0.0)
print(ArraySpec(shape=(4,), chunks=(2,), dtype="|b1").fill_value)
#> None
try:
    ArraySpec(shape=(4,), chunks=(2,), dtype="<i2", fill_value=1.5)
except ValidationError as e:
    print(e.errors()[0]["msg"].splitlines()[-1])
    #> fill_value: expected an integer or null, got 1.5
```

## Stored metadata documents

`to_stored_metadata_json` returns a node's metadata the way a Zarr store holds
it: a mapping from each store key to the JSON document stored there. For an
array, that is `.zarray` (the `to_metadata_json` document without
`attributes`) and `.zattrs` (the attributes), typed as
`pydantic_zarr.v2.ZarrV2ArrayStoredJSON`. `.zattrs` is included even when the
attributes are empty, as zarr-python writes it. A `GroupSpec`'s mapping holds
its own `.zgroup` and `.zattrs` and each member's, at any depth, prefixed by
the member's path: these are the metadata documents a Zarr implementation
writes. It is typed as `pydantic_zarr.v2.ZarrV2GroupStoredJSON`, which types the group's own
documents exactly and every other key as any of the three document types.

```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec

array = ArraySpec(shape=(10,), chunks=(5,), dtype="|u1", attributes={"foo": 42})
group = GroupSpec(attributes={"bar": True}, members={"array": array})
stored = group.to_stored_metadata_json()
print(list(stored))
#> ['.zgroup', '.zattrs', 'array/.zarray', 'array/.zattrs']
print(stored["array/.zattrs"])
#> {'foo': 42}
```

`from_stored_metadata_json` reads such a mapping back, with no store
involved; a missing `.zattrs` reads as empty attributes. A layout cannot tell
a group whose members were not read from one with none, so
`GroupSpec.from_stored_metadata_json` reads every group as a `GroupSpec` with
the members the layout holds, and `BaseGroupSpec.from_stored_metadata_json`
reads only the root's documents.

A key has no leading `/`: the Zarr V2 spec normalizes a node's logical path,
stripping any leading or trailing `/` and collapsing repeated `/`, before using
it as the prefix of the node's keys ([spec](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L250-L276)).
`from_stored_metadata_json` refuses a key with a leading `/` or an empty
segment, since `/.zattrs` would name the same node as `.zattrs`.

To write a hierarchy's metadata, write each document as JSON at its key
under the hierarchy's root. `json.dumps` writes the tuples in the documents as
JSON arrays. It writes a non-finite float in the attributes (`float("nan")`)
as the bare token `NaN`, as zarr-python does, which strict JSON parsers
reject; pass `allow_nan=False` to refuse such attributes instead. Fill values
need no care: a spec holds a non-finite fill value as a string, such as
`"NaN"` (see [Validation](#validation)).

```python
import json
import tempfile
from pathlib import Path

from pydantic_zarr.v2 import ArraySpec, GroupSpec

array = ArraySpec(shape=(10,), chunks=(5,), dtype="|u1", attributes={"foo": 42})
group = GroupSpec(attributes={"bar": True}, members={"array": array})
assert GroupSpec.from_stored_metadata_json(group.to_stored_metadata_json()) == group

with tempfile.TemporaryDirectory() as root:
    for key, document in group.to_stored_metadata_json().items():
        path = Path(root) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
    # the directory is now a Zarr store any implementation can open; read it back
    documents = {
        str(path.relative_to(root)): json.loads(path.read_text())
        for path in Path(root).rglob(".z*")
    }
    assert GroupSpec.from_stored_metadata_json(documents) == group
```

## Reading and writing with zarr-python

The two functions below write a spec's documents to a `zarr-python` store with
`zarr.create_hierarchy` (zarr-python 3.0.5 and later) and read a stored node's
documents back, for Zarr V2 and V3 specs alike. They live in
[`examples/zarr_io.py`](https://github.com/zarr-developers/pydantic-zarr/blob/main/examples/zarr_io.py),
which the example scripts import; copy them into a project that needs them.
The [Zarr V3 guide](usage_zarr_v3.md#reading-and-writing-with-zarr-python)
discusses them.

```python
--8<-- "examples/zarr_io.py:write"
```

```python
--8<-- "examples/zarr_io.py:read"
```

## Comparing `GroupSpec` and `ArraySpec` models

`GroupSpec` and `ArraySpec` both have `like` methods that take another `GroupSpec` or `ArraySpec` as an argument and return `True` (the models are like each other) or `False` (the models are not like each other).

The `like` method works by converting both input models to `dict` via `pydantic.BaseModel.model_dump`, and comparing the `dict` representation of the models. This means that instances of two different subclasses of `GroupSpec`, which would not be considered equal according to the `==` operator, will be considered `like` if and only if they serialize to identical `dict` instances.

The `like` method takes keyword arguments `include` and `exclude`, which determine the attributes included or excluded from the model comparison. So it's possible to use `like` to check if two `ArraySpec` instances have the same `shape`, `dtype` and `chunks` by calling `array_a.like(array_b, include={'shape', 'dtype', 'chunks'})`. This is useful if you don't care about the compressor or filters and just want to ensure that you can safely write an in-memory array to a Zarr array, which depends just on the two arrays having matching `shape`, `dtype`, and `chunks` attributes.

```python
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
```

To compare a spec with a stored node, read the node's documents into a spec first
(see [Reading and writing with zarr-python](#reading-and-writing-with-zarr-python)).

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

## Extra fields

The Zarr V2 spec treats keys it does not define differently in array and group
metadata. In `.zarray`, other keys "SHOULD NOT be present within the metadata
object and SHOULD be ignored by implementations", so `ArraySpec` accepts any
other key with a JSON value, gives it no meaning, and keeps it in the
`extra_fields` mapping, dumped back as a top-level member. `zarr-python` drops
these keys when it reads or writes an array, so a store written through it
holds none. In `.zgroup`, "Other keys MUST NOT be present", so the group specs
refuse them.

```python
from pydantic import ValidationError

from pydantic_zarr.v2 import ArraySpec, GroupSpec

array = ArraySpec.model_validate(
    {"shape": (10,), "chunks": (10,), "dtype": "<i8", "written_by": "acme"}
)
print(array.extra_fields)
#> {'written_by': 'acme'}

try:
    GroupSpec.model_validate({"zarr_format": 2, "attributes": {}, "written_by": "acme"})
except ValidationError as e:
    print(e.errors()[0]["type"])
    #> extra_forbidden
```
