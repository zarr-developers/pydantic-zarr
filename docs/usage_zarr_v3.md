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
pipelines included — must be claimed in the operative scope, so
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
