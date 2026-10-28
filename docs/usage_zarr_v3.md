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
at validation, matching the spelling `zarr-python` stores.

## Metadata documents

`model_dump` takes options that change what it returns, so its result can only
be typed as `dict[str, Any]`. `to_metadata_json` takes none: it returns the
node's `zarr.json` document, typed as `zarr_metadata.ZarrV3ArrayMetadataJSON`
for an array and `zarr_metadata.ZarrV3GroupMetadataJSON` for a group. A group's
members are not part of its metadata document, so they are left out.

```python
from pydantic_zarr.v3 import ArraySpec, GroupSpec

array = ArraySpec.from_shape_and_data_type((10,), "uint8")
document = array.to_metadata_json()
print(document["shape"], document["data_type"])
#> (10,) uint8

group = GroupSpec(attributes={"foo": 42}, members={"array": array})
print(group.to_metadata_json())
#> {'zarr_format': 3, 'node_type': 'group', 'attributes': {'foo': 42}}
```

## Stored metadata documents

`to_stored_metadata_json` returns a node's metadata the way a Zarr store holds
it: a mapping from each store key to the JSON document stored there. A Zarr V3
node stores one document, so the mapping has one key, `zarr.json`, typed as
`pydantic_zarr.v3.ZarrV3ArrayStoredJSON` for an array and
`pydantic_zarr.v3.ZarrV3GroupStoredJSON` for a group. Like `to_metadata_json`,
it leaves out a group's members.

```python
from pydantic_zarr.v3 import GroupSpec

group = GroupSpec(attributes={"foo": 42})
print(group.to_stored_metadata_json())
#> {'zarr.json': {'zarr_format': 3, 'node_type': 'group', 'attributes': {'foo': 42}}}
```

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
`zarr-metadata`'s `CORE_AND_EXTENSIONS`.

### Choosing the scope per call

The scope travels through pydantic's validation context: pass a
`ZarrV3Context` as `context` to `model_validate` (or a mapping holding one
under `pydantic_zarr.v3.CONTEXT_KEY`, when the context dict is shared with
other keys). The `scale_offset` codec below is claimed by the default scope,
which refuses its unknown configuration key; the `CORE` scope does not claim
it, so there the same document passes with the codec unjudged:

```python
from pydantic import ValidationError
from zarr_metadata.v3.definition import CORE

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

spec = ArraySpec.model_validate(document, context=CORE)
print(spec.codecs[-1]["name"])
#> scale_offset
```

### Scope as a class property

A per-call context lasts one call: operations that build new specs
(`update`, `from_shape_and_data_type`, `with_sharding_codec`, `like`)
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
`zarr_scope = CORE` plus strict accepts exactly the documents built from
names the core spec defines:

```python
from pydantic import ValidationError
from zarr_metadata.v3.definition import CORE

from pydantic_zarr.v3 import ArraySpec


class CoreArraySpec(ArraySpec):
    """Validates in the core scope, and refuses names it does not claim."""

    zarr_scope = CORE
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
`vlen-utf8`, `vlen-bytes`) are extensions that no current scope claims, as are
the legacy spellings `zarr-python` has stored (`"str"`,
`"variable_length_bytes"`), so a strict class refuses variable-length arrays.

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
  and judged by that definition. `CORE` (and so `CORE_AND_EXTENSIONS`) claims
  a group's `consolidated_metadata`, which `zarr-python` writes and which the
  spec allows on groups "for historical reasons".
- Any other extra field is unrecognized, and accepted only as an extension
  object — a `name` that is an extension name, an optional `configuration`
  object, no other members — that sets `"must_understand": false`.

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

array = ArraySpec.from_shape_and_data_type(
    (4, 4),
    "uint8",
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
    dimensions_names: not a field of this spec, and not an extra field the spec permits: an extra field is an extension object with "must_understand": false, got ('y', 'x')
    """
```

`update(extra_fields=...)` replaces the extra fields as a whole. `to_zarr`
writes them and `from_zarr` reads them back (`zarr-python` stores them on
arrays from 3.1 and on groups from 3.4; `to_zarr` refuses extra fields an
older `zarr-python` cannot store). `consolidated_metadata` is the exception:
`zarr-python` writes it as a record of the hierarchy below the group, so
`from_zarr` reads the members themselves and leaves it out, and `to_zarr`
refuses a group holding it rather than write a record that need not match the
members written; drop it first and consolidate the written hierarchy with
`zarr.consolidate_metadata`. Its contents are kept as written, not read as
members.
