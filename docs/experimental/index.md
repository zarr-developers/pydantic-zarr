# experimental features

## Improved `GroupSpec` and `ArraySpec` classes

We are developing a redesign of the `GroupSpec` and `ArraySpec` classes. These new classes can be found in the `experimental.v2` and `experimental.v3` modules for Zarr V2 and V3, respectively. Our
design goal for these new classes is to make the models simpler, more explicit, and more expressive.

Here's how we are doing that:

### Removing generic type parameters

In `pydantic_zarr`, the `GroupSpec` and `ArraySpec` classes take generic type parameters. `GroupSpec[A, B]` models a Zarr group with attributes that must be instances of `A` and child nodes that must be instances of `B`. The generic type parameters offer concise class definitions but complicate type checking for pydantic, and they are also not strictly necessary for the `GroupSpec` and `ArraySpec`
classes to do their jobs. So in `pydantic_zarr.experimental.v2` and `pydantic_zarr.experimental.v3` the `GroupSpec` and `ArraySpec` classes are not generic any more. They are just regular classes.

Code like this:

```python
from pydantic import BaseModel
from pydantic_zarr.v3 import ArraySpec

class AttrsType(BaseModel):
    a: int
    b: float

MyArray = ArraySpec[AttrsType]
print(MyArray)
#> <class 'pydantic_zarr.v3.ArraySpec[AttrsType]'>
```

becomes this:

```python
from pydantic import BaseModel
from pydantic_zarr.experimental.v3 import ArraySpec

class AttrsType(BaseModel):
    a: int
    b: float

class MyArray(ArraySpec):
    attributes: AttrsType

print(MyArray)
#> <class '__main__.MyArray'>
```

### A class hierarchy for Zarr Groups

In `pydantic_zarr.v2` and `pydantic_zarr.v3`, the `members` attribute of the `GroupSpec` class is
annotated as a union type with two variants: `Mapping` and `None`. `None` occurs in this union to handle the case where we want to model a Zarr group outside the context of any hierarchy, i.e. a situation where the `members` attribute would be undefined.

The main place where this occurs is when we create a flattened representation of a Zarr hierarchy with the `to_flat` functions. `to_flat` takes a Zarr hierarchy (a tree) and converts it to a `key: value` data structure where the hierarchy information is encoded in the structure of the keys. After this transformation, it is redundant for
the `GroupSpec` elements of the flattened Zarr hierarchy to carry their own representation of the hierarchy structure, as that information is completely specified by the keys. So when we flatten
a `GroupSpec`, we set all the `members` attributes to `None`.

But outside the context of flattening hierarchies, we need to handle the `None` variant in places where we are sure that the members are not `None`, and this is tedious.

To solve this problem, instead of defining the `members` attribute as a union over two possible types, in `pydantic_zarr.experimental.v2` and `pydantic_zarr.experimental.v3` we define two classes for modelling Zarr groups. One class, `BaseGroupSpec`,
narrowly models the structure of a Zarr Group. The `GroupSpec` class inherits from `BaseGroupSpec` and
defines a new `members` attribute, which allows it to model Zarr groups that have information
about the sub-groups and sub-arrays they contain. With this structure, `to_flat` can safely return a mapping from strings to `ArraySpec | BaseGroupSpec`, and `GroupSpec` instances don't need to handle the case where their `members` attribute is `None`.

With `BaseGroupSpec`, type checkers and Pydantic can distinguish at definition-time whether a group should have members, eliminating runtime None-checks in code that knows members must exist.

Ordinary `pydantic-zarr` usage should not be affected by the new class hierarchy for `GroupSpec` classes. The only time a user would create a `BaseGroupSpec` explicitly is when declaring a Zarr hierarchy in the flat form. Otherwise, `GroupSpec` is sufficient for all uses.

### More explicit modelling of Zarr groups

Since `pydantic-zarr` started, the Python type system has become significantly more expressive. One very useful development has been improvements in the `TypedDict` class for modelling mappings with known keys and typed values. `TypedDict` is a perfect fit for modelling Zarr groups where the names
of the members are part of the schema definition for that group.

The `GroupSpec` classes defined in `pydantic_zarr.experimental` accept `TypedDict` annotations for their `members` attribute. As `pydantic` can validate values against a `TypedDict` type annotation, we get a very concise type check on the names of the members of a Zarr group.

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

array = ArraySpec(
    shape=(1,),
    data_type='uint8',
    codecs=('bytes',),
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": (1,)}},
    chunk_key_encoding = {"name": "default"},
    fill_value = 0,
    )

class MyMembers(TypedDict):
    a: ArraySpec
    b: ArraySpec

class MyGroup(GroupSpec):
    members: MyMembers

# validation fails: missing array named "b"
try:
    MyGroup(members={"a": array}, attributes={})
except ValueError as e:
    print(e)
    """
    1 validation error for MyGroup
    members.b
      Field required [type=missing, input_value={'a': ArraySpec(zarr_form..., dimension_names=None)}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.11/v/missing
    """

# validation fails: extra array named "c"
try:
    MyGroup(members={"a": array, "b": array, "c": array}, attributes={})
except ValueError as e:
    print(e)
    """
    1 validation error for MyGroup
    members.c
      Extra inputs are not permitted [type=extra_forbidden, input_value=ArraySpec(zarr_format=3, ...), dimension_names=None), input_type=ArraySpec]
        For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden
    """

# validation succeeds
MyGroup(members={"a" : array, "b": array}, attributes={})
```