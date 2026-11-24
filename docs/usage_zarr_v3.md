# Usage (Zarr V3)

## Defining Zarr v3 hierarchies

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec, NamedConfig

array_attributes = {"baz": [1, 2, 3]}
group_attributes = {"foo": 42, "bar": False}

array_spec = ArraySpec(
    attributes=array_attributes,
    shape=[1000, 1000],
    dimension_names=["rows", "columns"],
    data_type="uint8",
    chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
    chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
    codecs=[
        NamedConfig(name="bytes"),
        NamedConfig(name="gzip", configuration={"level": 1}),
    ],
    storage_transformers=(),
    fill_value=0,
)

spec = GroupSpec(attributes=group_attributes, members={"array": array_spec})
print(spec.model_dump_json(indent=2))
"""
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "foo": 42,
    "bar": false
  },
  "members": {
    "array": {
      "zarr_format": 3,
      "node_type": "array",
      "attributes": {
        "baz": [
          1,
          2,
          3
        ]
      },
      "shape": [
        1000,
        1000
      ],
      "data_type": "uint8",
      "chunk_grid": {
        "name": "regular",
        "configuration": {
          "chunk_shape": [
            1000,
            100
          ]
        }
      },
      "chunk_key_encoding": {
        "name": "default",
        "configuration": {
          "separator": "/"
        }
      },
      "fill_value": 0,
      "codecs": [
        {
          "name": "bytes"
        },
        {
          "name": "gzip",
          "configuration": {
            "level": 1
          }
        }
      ],
      "storage_transformers": [],
      "dimension_names": [
        "rows",
        "columns"
      ]
    }
  }
}
"""
```

For a single-byte data type like `uint8`, the `bytes` codec needs no `endian`
configuration; a no-op `{"endian": ...}` configuration passed anyway is dropped
at validation, matching the spelling `zarr-python` stores. Which data types are
single-byte is what their definitions in the spec's scope declare, so in
`zarr-python`'s scope this covers its fixed-length and raw bytes types too.

## Typing a group's members

`GroupSpec` takes two type parameters: the type of its attributes, and the type
of its whole `members` mapping. The members type states what a group may hold:

- Left out, the group holds any arrays and groups.
- A `Mapping` constrains the type of every member:
  `GroupSpec[Attrs, Mapping[str, MyArraySpec]]`.
- A closed `TypedDict` fixes the member names, each with its own type. An
  undeclared member is refused, and so is a missing one.

```python
from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_zarr.v3 import ArraySpec, GroupSpec, TBaseAttr


class LevelMembers(TypedDict, closed=True):
    x: ArraySpec
    y: ArraySpec
    band: ArraySpec


Level = GroupSpec[TBaseAttr, LevelMembers]

axis = ArraySpec.create_default(shape=(4,), data_type="float64")
band = ArraySpec.create_default(shape=(4, 4), data_type="uint16")

level = Level(members={"x": axis, "y": axis, "band": band})
print(level.members["band"].shape)
#> (4, 4)

try:
    Level(members={"x": axis, "y": axis})
except ValidationError as e:
    print(e.errors()[0]["loc"], e.errors()[0]["type"])
    #> ('members', 'band') missing

try:
    Level(members={"x": axis, "y": axis, "band": band, "extra": band})
except ValidationError as e:
    print(e.errors()[0]["loc"], e.errors()[0]["type"])
    #> ('members', 'extra') extra_forbidden
```

A group subclass that adds validators can leave the members type as a
parameter, with `class Dataset(GroupSpec[TBaseAttr, TMembers], Generic[TMembers])`,
so that each concrete group names its own members: `Dataset[LevelMembers]`.

Notes for type checkers and for `TypedDict` members:

- **Closed `TypedDict`s only.** pyright, ty and pyrefly accept a `TypedDict`
  as the members type when it is closed (`closed=True`) or types its other
  members (`extra_items=...`), because only then is every member known to be
  an array or a group ([PEP 728](https://peps.python.org/pep-0728/)). mypy
  does not implement that rule yet and reports the argument as outside the
  parameter's bound.
- **`extra_items` and pydantic.** Decorate a `TypedDict` that has
  `extra_items` with `@with_config(ConfigDict())` (both from `pydantic`).
  Without its own configuration pydantic warns that the group's
  `extra="forbid"` setting does not apply to the `TypedDict`. Type checkers
  accept the decorator, where they refuse a `__pydantic_config__` assignment
  in the `TypedDict` body.
- **Member order.** A group keeps its members in the order they were given.
  pydantic builds a `TypedDict` in declaration order, so the group puts the
  members back in the given order afterwards. This also holds for a union of
  `TypedDict`s, a generic `TypedDict`, a type alias of a `TypedDict` and a
  type variable bound to one. Any other members type, such as one with
  validators of its own (`Annotated[..., AfterValidator(...)]`), keeps the
  order pydantic validates it in. A member given under an alias goes after
  the members given by name.
- **Read-only members.** A spec is frozen, but type checkers let code assign
  to a key of a `TypedDict`, so `group.members["band"] = other` type-checks.
  Declare the items `ReadOnly[...]` (from `typing_extensions`) for checkers to
  refuse that assignment. pydantic accepts `ReadOnly` items, with a warning
  that it does not enforce them at runtime.
- **Avoid `Any`.** `Any` as the members type turns off member validation:
  `GroupSpec[Any, Any]` accepts any mapping with string keys, does not check
  that its values are specs, and its JSON schema leaves `members`
  unconstrained. For a group
  of any arrays and groups, write `GroupSpec` (or `AnyGroupSpec`), whose
  members type defaults to `TBaseMembers`; for any attributes, give
  `TBaseAttr`, as in `GroupSpec[TBaseAttr, LevelMembers]`.
- **Building a group without a members type.** A checker infers the members
  type from the `members` argument. With arrays and groups mixed in one dict
  literal, pyright infers `dict[str, Unknown]`, and from `members={}` it
  infers `dict[Unknown, Unknown]`; annotate the result, as in
  `group: AnyGroupSpec = GroupSpec(members={...})`, to keep the members typed.
- **The type of one member.** Up to 0.11 the parameter was the type of one
  member. `GroupSpec[Attrs, MyArraySpec]` is now written
  `GroupSpec[Attrs, Mapping[str, MyArraySpec]]`. A class still written the old
  way validates its whole members mapping as one member, so validation
  fails; type checkers refuse it where it is written. The validation errors
  describe the members mapping as if it were one member, for example
  `members.shape  Field required` for an array type, so they do not mention
  the change of parameter.

## Creating array specs

`ArraySpec.create_default` builds a valid array spec from whichever fields are
given. Every other field takes the default `zarr-metadata` derives from them, as
its `ZarrV3ArrayMetadata.create_default` does: a shape brings one regular chunk
covering the array, and a data type brings its default fill value (zero,
`False`, the empty string, `"NaT"`) and its array -> bytes codec. With nothing
given, the array is a scalar `uint8`. `chunk_grid="auto"` guesses a chunk shape
from the array's size in bytes instead, as zarr-python does when no chunks are
given.

On a subclass, a field the class declares a default for counts as given, so a
class can pin values and leave the rest to be derived from them:

```python
from typing import Literal

from pydantic_zarr.v3 import ArraySpec

array = ArraySpec.create_default(shape=(4, 4), data_type="float32")
print(array.fill_value)
#> 0.0


class NaNFloat32(ArraySpec):
    data_type: Literal["float32"] = "float32"
    fill_value: Literal["NaN"] = "NaN"


pinned = NaNFloat32.create_default(shape=(4, 4))
print(pinned.data_type, pinned.fill_value)
#> float32 NaN

guessed = ArraySpec.create_default(shape=(10980, 10980), data_type="uint16", chunk_grid="auto")
print(guessed.chunk_grid["configuration"]["chunk_shape"])
#> (687, 1373)
```

A field the class requires and the data cannot provide, such as an attributes
model with required keys, has to be passed, as it does to the constructor. A
subclass whose construction needs more than defaults, such as attributes
computed from the shape, can override `create_default`. To stay a compatible
override, it keeps the keyword-only signature and passes the computed fields on
to `super().create_default(...)`.

## Metadata documents

`model_dump` takes options that change what it returns, so its result can only
be typed as `dict[str, Any]`. `to_metadata_json` takes none: it returns the
node's `zarr.json` document, typed as `zarr_metadata.ZarrV3ArrayMetadataJSON`
for an array and `zarr_metadata.ZarrV3GroupMetadataJSON` for a group. A group's
members are not part of its metadata document, so they are left out.

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec

array = ArraySpec.create_default(shape=(10,), data_type="uint8")
document = array.to_metadata_json()
print(document["shape"], document["data_type"])
#> (10,) uint8

group = GroupSpec(attributes={"foo": 42}, members={"array": array})
print(group.to_metadata_json())
#> {'zarr_format': 3, 'node_type': 'group', 'attributes': {'foo': 42}}
```

### Envelope spellings

A document's declarations (the data type, chunk grid, chunk key encoding,
codecs and storage transformers) are read in every envelope spelling the spec
allows, and held in the canonical one `zarr-metadata` writes: a bare name reads
as `{"name": ...}`, an explicit `"must_understand": true` and an empty
`configuration` are dropped, a data type object with nothing to configure reads
as its bare name, and the JSON arrays of a configuration become tuples. So specs
built from different spellings of one document compare equal. A declaration
that says `"must_understand": false` is refused, as `zarr-metadata` refuses it: a
reader may not ignore a codec or a chunk grid. `chunk_grid` and
`chunk_key_encoding` take any declaration the scope claims; the core spec
defines the regular grid, and `zarr-metadata` also defines the rectilinear one.

```python
from pydantic_zarr.v3 import ArraySpec

canonical = ArraySpec.create_default(shape=(5,), data_type="uint8")
spelled = ArraySpec.model_validate(
    {
        "shape": [5],
        "data_type": {"name": "uint8", "must_understand": True},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": ["bytes"],
    }
)
print(spelled == canonical)
#> True
```

## Stored metadata documents

`to_stored_metadata_json` returns a node's metadata the way a Zarr store holds
it: a mapping from each store key to the JSON document stored there. An array
stores one document, so its mapping has one key, `zarr.json`, typed as
`pydantic_zarr.v3.ZarrV3ArrayStoredJSON`. A `GroupSpec`'s mapping holds its own
`zarr.json` and each member's, at any depth, at the member's path:
these are the metadata documents a Zarr implementation writes. It is typed as
`pydantic_zarr.v3.ZarrV3GroupStoredJSON`, which types the group's own
`zarr.json` exactly and every other key as either document type.

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec

array = ArraySpec.create_default(shape=(10,), data_type="uint8")
group = GroupSpec(attributes={"foo": 42}, members={"sub": GroupSpec(members={"array": array})})
print(list(group.to_stored_metadata_json()))
#> ['zarr.json', 'sub/zarr.json', 'sub/array/zarr.json']
```

`from_stored_metadata_json` reads such a mapping back, with no store
involved. A layout cannot tell a group whose members were not read from one
with none, so `GroupSpec.from_stored_metadata_json` reads every group as a
`GroupSpec` with the members the layout holds, and
`BaseGroupSpec.from_stored_metadata_json` reads only the root's document.

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec

array = ArraySpec.create_default(shape=(10,), data_type="uint8")
group = GroupSpec(attributes={"foo": 42}, members={"sub": GroupSpec(members={"array": array})})
assert GroupSpec.from_stored_metadata_json(group.to_stored_metadata_json()) == group
```

A key has no leading `/`: the Zarr V3 spec forms a node's metadata key by
stripping the leading `/` from its hierarchy path and appending `/zarr.json`,
so the root's key is `zarr.json` and the key of the node at `/sub/array` is
`sub/array/zarr.json` ([spec](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L382-L388)).
`from_stored_metadata_json` refuses a key with a leading `/` or an empty
segment, since `/zarr.json` would name the same node as `zarr.json`.

To write a hierarchy's metadata, write each document as JSON at its key
under the hierarchy's root. `json.dumps` writes the tuples in the documents as
JSON arrays. It writes a non-finite float in the attributes (`float("nan")`)
as the bare token `NaN`, as zarr-python does, which strict JSON parsers
reject; pass `allow_nan=False` to refuse such attributes instead. Fill values
need no care: a spec holds a non-finite fill value as a string, such as
`"NaN"`.

```python
import json
import tempfile
from pathlib import Path

from pydantic_zarr.v3 import ArraySpec, GroupSpec

array = ArraySpec.create_default(shape=(10,), data_type="uint8")
group = GroupSpec(attributes={"foo": 42}, members={"sub": GroupSpec(members={"array": array})})

with tempfile.TemporaryDirectory() as root:
    for key, document in group.to_stored_metadata_json().items():
        path = Path(root) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
    # the directory is now a Zarr store any implementation can open; read it back
    documents = {
        str(path.relative_to(root)): json.loads(path.read_text())
        for path in Path(root).rglob("zarr.json")
    }
    assert GroupSpec.from_stored_metadata_json(documents) == group
```

## Reading and writing with zarr-python

pydantic-zarr does no IO. A spec is a metadata document, or a hierarchy of
them; moving documents in and out of a store is the job of a Zarr
implementation, and the two meet at the stored-documents mapping above. With
zarr-python, `zarr.create_hierarchy` (zarr-python 3.0.5 and later) writes a
hierarchy from zarr-python's metadata objects, which it builds from the
documents, and a stored node's `metadata.to_buffer_dict` gives its documents
back as the store holds them. Two functions connect them to the specs:

```python
--8<-- "examples/zarr_io.py:write"
```

```python
--8<-- "examples/zarr_io.py:read"
```

A stored document that breaks the spec is refused, as it should be.
zarr-python 3.0 writes `"consolidated_metadata": null` into every group,
which the spec does not allow. `read_documents` never passes it on, because it
leaves consolidated metadata out altogether (see below). A reader that takes a
group's `zarr.json` whole, such as one that reads the files of a store
directly, repairs it with
[`normalize_group_metadata`][pydantic_zarr.v3.normalize_group_metadata] before
`from_stored_metadata_json`. That routine takes a JSON value and gives one
back, and each repair it knows is asked for by a keyword:
`normalize_null_consolidated_metadata` (on by default) drops the null.

Together they round-trip a hierarchy through any zarr-python store:

```python
--8<-- "examples/zarr_io.py:round_trip"
```

Consolidated metadata (`zarr.consolidate_metadata`) is a copy of the
documents below a group: `.zmetadata` in Zarr V2, and the
`consolidated_metadata` member of a V3 group's `zarr.json`. `read_documents`
reads every node's own documents, so it leaves the copy out, and a
consolidated hierarchy reads back as the spec that was written.

Both live in
[`examples/zarr_io.py`](https://github.com/zarr-developers/pydantic-zarr/blob/main/examples/zarr_io.py),
which the example scripts import; copy them into a project that needs them.
They handle Zarr V2 specs the same way. Neither reads or writes array data:
`write_hierarchy` creates the metadata documents, and the arrays are then
filled through zarr-python as usual.

Up to pydantic-zarr 0.11, the `to_zarr` and `from_zarr` methods of the specs,
the module-level `to_zarr`, `from_zarr` and `from_zarr_flat`, and `like` or
`ArraySpec.from_array` given a `zarr.Array` or `zarr.Group` did this inside
pydantic-zarr through zarr-python internals. They were removed in 0.12: use
the two functions above instead.

## Validation scopes

Constructing an `ArraySpec` validates its fields as a whole Zarr V3 array
metadata document, via [`zarr-metadata`](https://pypi.org/project/zarr-metadata/):
the fill value is judged against the data type, the chunk grid against the
shape, and the codecs as a pipeline. Each extension point — data type, chunk
grid, chunk key encoding, codec, storage transformer — is judged by the
definition that claims its name in a *scope* (a `zarr_metadata`
`ZarrV3Context`). A scope is a vocabulary, not a whitelist: a name nothing in
the scope claims is left unjudged, so unknown extensions pass, while a claimed
name with an invalid configuration is refused. The default scope is
`zarr-metadata`'s `CORE_AND_EXTENSIONS_CONTEXT`.

### Choosing the scope per call

The scope travels through pydantic's validation context: pass a
`ZarrV3Context` as `context` to `model_validate` (or a mapping holding one
under `pydantic_zarr.v3.CONTEXT_KEY`, when the context dict is shared with
other keys). The `scale_offset` codec below is claimed by the default scope,
which refuses its unknown configuration key; the `CORE_CONTEXT` scope does not claim
it, so there the same document passes with the codec unjudged:

```python
from pydantic import ValidationError
from zarr_metadata.v3.definition import CORE_CONTEXT

from pydantic_zarr.v3 import ArraySpec

document = {
    "shape": (4, 4),
    "data_type": "float32",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0.0,
    "codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "scale_offset", "configuration": {"bogus": 1}},
    ],
}
try:
    ArraySpec.model_validate(document)
except ValidationError as e:
    print(e.error_count())
    #> 1

spec = ArraySpec.model_validate(document, context=CORE_CONTEXT)
print(spec.codecs[-1]["name"])
#> scale_offset
```

### Scope as a class property

A per-call context lasts one call: operations that build new specs
(`replace`, `create_default`, `with_sharding_codec`, `like`)
re-validate with no context, back in the default scope. To make the scope a
property of the model instead, set the `zarr_scope` class variable on a
subclass: it is the scope every validation of that class reads in when the
context carries none, so it holds at direct construction and through every
derived operation. An explicit validation context still wins. A group's
members are validated as their own classes, so a scoped tree narrows its
member types to scope-bearing classes (an explicit context, which pydantic
carries into nested validation, reaches the members either way).

### Strict scope conformance

Because unclaimed names pass unjudged, a smaller scope judges *less*, not
more. A class that also sets `zarr_scope_strict` flips that: every
extension-point name in the document — a sharding codec's inner and index
pipelines included — and every [extra field](#extra-fields) key must be
claimed in the operative scope, so
`zarr_scope = CORE_CONTEXT` plus strict accepts exactly the documents built from
names the core spec defines:

```python
from pydantic import ValidationError
from zarr_metadata.v3.definition import CORE_CONTEXT

from pydantic_zarr.v3 import ArraySpec


class CoreArraySpec(ArraySpec):
    """Validates in the core scope, and refuses names it does not claim."""

    zarr_scope = CORE_CONTEXT
    zarr_scope_strict = True


document = {
    "shape": (4, 4),
    "data_type": "float32",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0.0,
    "codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
    ],
}
# the zstd codec is a registered extension, not part of the core spec
try:
    CoreArraySpec.model_validate(document)
except ValidationError as e:
    print(e.errors()[0]["msg"])
    """
    Value error, Names the scope leaves unjudged, refused because this class sets `zarr_scope_strict`:
    codecs.1.name: 'zstd' names no codec in the scope
    """

core_document = {**document, "codecs": document["codecs"][:1]}
spec = CoreArraySpec.model_validate(core_document)
# derived operations re-validate as the class, so the contract sticks
sharded = spec.with_sharding_codec((2, 2))
print(sharded.codecs[0]["name"])
#> sharding_indexed
```

The variable-length data types and their codecs (`string`, `bytes`,
`vlen-utf8`, `vlen-bytes`) are registered extensions, not part of the core spec,
so a strict core class refuses variable-length arrays. `CORE_AND_EXTENSIONS_CONTEXT`
claims them, and `ZARR_PYTHON_CONTEXT` also claims the `"variable_length_bytes"`
spelling `zarr-python` has stored. No scope claims `"str"`, the spelling
`zarr-python` 3.0 stored for the string data type.

## Extra fields

A Zarr V3 metadata document may hold members the spec does not define. The
spec's rule for these [additional
fields](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#additional-fields)
is the `must_understand` rule for extensions: "An implementation MUST fail to
open Zarr groups or arrays if any metadata fields are present which (a) the
implementation does not recognize and (b) are not explicitly set to
`"must_understand": false`." Specs apply it in the same scope as the rest of
the document, to arrays and groups alike:

- An extra field whose key a definition in the scope claims is recognized,
  and judged by that definition. `CORE_CONTEXT` (and so `CORE_AND_EXTENSIONS_CONTEXT`) claims
  a group's `consolidated_metadata`, which `zarr-python` writes and which the
  spec allows on groups "for historical reasons".
- Any other extra field is unrecognized, and accepted only when it is an
  object that says `"must_understand": false`. It is kept as written, whatever
  else it holds: that is the reading `zarr-metadata` gives the rule, and what
  its strategies draw.

Every other unknown key is refused, which is also what a misspelled field name
meets. Accepted extra fields are kept in the `extra_fields` mapping and written
back as top-level members of the document:

```python
from pydantic import ValidationError

from pydantic_zarr.v3 import ArraySpec, GroupSpec

# a group document as zarr-python writes it after consolidating the hierarchy
group = GroupSpec.model_validate(
    {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": {}},
    }
)
print(list(group.extra_fields))
#> ['consolidated_metadata']
print(group.model_dump(exclude={"members"}))
"""
{
    'zarr_format': 3,
    'node_type': 'group',
    'attributes': {},
    'consolidated_metadata': {
        'kind': 'inline',
        'must_understand': False,
        'metadata': {},
    },
}
"""

array = ArraySpec.create_default(
    shape=(4, 4),
    data_type="uint8",
    extra_fields={"provenance": {"name": "acme.provenance", "must_understand": False}},
)
print(array.model_dump()["provenance"])
#> {'name': 'acme.provenance', 'must_understand': False}

# an extension object a reader must understand, which nothing here recognizes
try:
    ArraySpec.model_validate({**array.model_dump(), "provenance": {"name": "acme.provenance"}})
except ValidationError as e:
    print(e.errors()[0]["msg"])
    """
    Value error, Invalid Zarr V3 array metadata:
    provenance: no array extra field in the scope recognizes this field, and it does not set "must_understand": false, so the spec requires a reader to refuse it
    """

# a misspelled field name is an extra field the spec does not permit
try:
    ArraySpec.model_validate({**array.model_dump(), "dimensions_names": ["y", "x"]})
except ValidationError as e:
    print(e.errors()[0]["msg"])
    """
    Value error, Invalid Zarr V3 array metadata:
    dimensions_names: not a field of this spec, and not an extra field the spec permits: an extra field is an object that says "must_understand": false, got ('y', 'x')
    """
```

`replace(extra_fields=...)` replaces the extra fields as a whole. The stored
documents carry them, and `from_stored_metadata_json` reads them back.
Whether a store written through `zarr-python` holds them depends on
`zarr-python`, which keeps an array's extra fields from 3.1 and a group's from
3.4, and drops them before that. `consolidated_metadata` is a special case:
`zarr-python` writes it as a record of the hierarchy below the group, and the
specs keep its contents as written, not read as members. A hierarchy written
from a spec holding it would carry a record that need not match the members
written, so drop it first and consolidate the written hierarchy with
`zarr.consolidate_metadata`. For the same reason, `read_documents` leaves it
out when it reads a stored hierarchy (see [Reading and writing with
zarr-python](#reading-and-writing-with-zarr-python)).
