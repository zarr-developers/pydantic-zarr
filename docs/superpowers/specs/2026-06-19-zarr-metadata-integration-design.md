# zarr-metadata integration — design

**Date:** 2026-06-19
**Status:** Approved (brainstorm), pending implementation plan
**Scope:** Stable `pydantic_zarr.v2` / `pydantic_zarr.v3` only (not `experimental.*`)

## Goal

Adopt [`zarr-metadata`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
`>=0.3.0` as the source of truth for Zarr metadata document types, and use it to:

1. Add **typed serialization methods** (`to_json`, and for v2 `to_store_json`) that return
   spec-defined TypedDicts.
2. **Replace the hand-rolled internal field types** in `ArraySpec` / `GroupSpec` with
   `zarr-metadata` equivalents.
3. Introduce an opt-in **strict** spec variant whose validation is dtype- and codec-aware,
   alongside the existing **loose** spec (renamed semantics only — class names unchanged).

`zarr-metadata` is a pure-types package (only `typing-extensions`; no numpy/pydantic).
As of 0.3.0 its types are deliberately pydantic-`TypeAdapter`-resolvable (recursive `JSONValue`
built with `TypeAliasType`), and `fill_value`/`attributes`/config fields are typed `JSONValue`
rather than `object`.

## Key decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Target modules | Stable `v2`/`v3` only |
| Integration depth | Typed output **and** replace internal field types |
| Serialization names | `to_json` (faithful inline) + `to_store_json` (v2 `.zarray`/`.zattrs` split) |
| `to_json` impl | Wraps `model_dump`, casts to the zarr-metadata return type |
| Dependency | `zarr-metadata>=0.3.0` as a **core runtime** dependency |
| Strict vs loose | **Separate classes** (not a flag/parameter) |
| Loose semantics | Syntax only — `{name, configuration}` envelopes; `fill_value: JSONValue` |
| Strict semantics | dtype- and codec-aware — per-dtype `fill_value`, per-codec configs |
| Class structure | Sibling classes over a behavior-only base (Approach B) |
| Dtype coverage (strict) | Core numeric first: bool, int/uint 8–64, float16/32/64, complex64/128, raw `r<N>`. Defer string/datetime64/timedelta64/struct. |
| Group strictness | `GroupSpec` stays a single class (no `StrictGroupSpec`) |

## Why Approach B (siblings over a mixin), not a strict subclass

A strict subclass that **narrows** an inherited field annotation
(`StrictArraySpec(ArraySpec)` redeclaring `codecs` with a tighter type) works at runtime —
pydantic rebuilds the child validator and enforces the narrowed type — **but fails
`mypy --strict`**, which this project runs in CI and pre-commit:

```
error: Incompatible types in assignment (expression has type
"tuple[BloscCfg, ...]", base class "Loose" defined the type as
"tuple[str | NamedCfg, ...]")  [assignment]
```

(Verified directly with `mypy --strict` + the pydantic plugin.)

Therefore strict and loose are **siblings**, each declaring its own fields, sharing only
behavior via a non-field base. No inherited field is narrowed, so there is no LSP/override
error. This was verified to type-check clean and run:

```
class _BaseArraySpec(BaseModel, Generic[TAttr]):   # behavior only, no codec/dtype fields
    ...
    def to_json(self) -> ArrayMetadataV3: ...
class ArraySpec(_BaseArraySpec[TAttr], Generic[TAttr]):        # loose fields
    codecs: tuple[MetadataV3, ...]
class StrictArraySpec(...):                                     # strict fields (see below)
```

## Architecture

Per version (v2 and v3):

```
_BaseArraySpec[TAttr]   behavior-only: validators, from_array, from_zarr, to_zarr,
  │                     like, to_json, to_store_json. No codec/dtype fields.
  ├── ArraySpec[TAttr]          loose fields (envelope-only types, fill_value: JSONValue)
  ├── _BoolArraySpec            ┐ per-dtype strict member classes (private),
  ├── _Int8ArraySpec            │ each declaring data_type + matching fill_value
  ├── ... (one per core dtype)  │ + strict codecs/grid/encoding
  └── _RawArraySpec             ┘

StrictArraySpec = Annotated[_BoolArraySpec | ... | _RawArraySpec,
                            Field(discriminator="data_type")]   # public type alias, a union
```

Both `ArraySpec` and every `_<Dtype>ArraySpec` are direct subclasses of `_BaseArraySpec`
(siblings — none narrows another's fields). `StrictArraySpec` is not a class but a public type
alias for the discriminated union of the per-dtype members.

- `ArraySpec` keeps its name and **loose** semantics → backward compatible.
- `StrictArraySpec` is new and opt-in.
- `GroupSpec` keeps its single-class form; gains `to_json` (+ v2 `to_store_json`).

### Strict spec = discriminated union over per-dtype models

Strict mode's defining feature is that `data_type` and `fill_value` are **coupled**: e.g. a
`float64` array may have fill value `"NaN"` / `"Infinity"` / `"-Infinity"` / a `HexFloat64`
string / a number, while an `int64` array may not have a string fill value at all. The current
flat `FillValue` union cannot express this because it does not tie `fill_value` to `data_type`.

We model strict mode as one pydantic model **per core dtype**, then take their union,
discriminated on `data_type`:

```python
class _Float64ArraySpec(_BaseArraySpec[TAttr], Generic[TAttr]):
    data_type: Float64DataTypeName        # Literal["float64"]
    fill_value: Float64FillValue          # float | int | "NaN"|"Infinity"|"-Infinity" | HexFloat64
    codecs: tuple[_StrictCodec, ...]      # union of per-codec metadata types
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata
    ...
# one such class per core dtype: bool, int8..uint64, float16/32/64, complex64/128, raw

StrictArraySpec = Annotated[
    _BoolArraySpec | _Int8ArraySpec | ... | _RawArraySpec,
    Field(discriminator="data_type"),
]
```

- Per-dtype member classes are private (underscore-prefixed). `StrictArraySpec` is the public
  type alias.
- Verified: the discriminated union routes by `data_type`, accepts `float64`+`"NaN"`/hex,
  rejects `int64`+`"NaN"`. The bare union also works (pydantic smart-union), but
  `Field(discriminator="data_type")` is used for sharper error messages.
- Strict codec union is composed locally from zarr-metadata per-codec types
  (`BloscCodecMetadata | GzipCodecMetadata | BytesCodecMetadata | ...`); zarr-metadata does
  not ship a pre-built "any known codec" union.

## Internal type replacements (loose classes)

### v3 loose `ArraySpec`

| Field / local type | Today | Replace with |
| --- | --- | --- |
| `NamedConfig` / `AnyNamedConfig` | local generic TypedDict | `NamedConfigV3` |
| `chunk_grid` | `NamedConfig[Literal["regular"], ...]` | `MetadataV3` (loose) |
| `chunk_key_encoding` | `DefaultChunkKeyEncoding \| V2ChunkKeyEncoding` | `MetadataV3` (loose) |
| `codecs` | `tuple[str \| AnyNamedConfig, ...]` | `tuple[MetadataV3, ...]` |
| `data_type` | `str \| AnyNamedConfig` (+ `parse_dtype_v3`) | `MetadataV3` (keep validator) |
| `fill_value` | narrow local union | `JSONValue` (**widened** — syntax only) |

The local `NamedConfig` is currently a public export; keep `NamedConfig = NamedConfigV3` as a
re-export alias with a deprecation note to avoid breaking importers.

### v2 loose `ArraySpec`

| Field / local type | Today | Replace with |
| --- | --- | --- |
| `order` (`MemoryOrder`) | local `Literal["C","F"]` | `ArrayOrderV2` |
| `dimension_separator` | local `Literal[".","/"]` | `ArrayDimensionSeparatorV2` |
| `compressor` / `filters` (`CodecDict`) | `dict[str, Any]` | `CodecMetadataV2` |
| `dtype` | `DtypeStr \| list[tuple]` | `DataTypeMetadataV2` (keep `parse_dtype_v2`) |
| `fill_value` | narrow local union | `JSONValue` (widened — syntax only) |

All existing `parse_*` `BeforeValidator`s are retained — they coerce user input before the type
check, which zarr-metadata (types-only) does not do.

## Serialization methods (on `_BaseArraySpec`)

### v3 — single document
- `to_json(self) -> ArrayMetadataV3` — wraps `model_dump(mode="json")`, casts to
  `ArrayMetadataV3`. Retains the existing behavior of omitting `dimension_names` when `None`
  (the `model_dump` override moves onto the base).
- No `to_store_json` (v3 stores everything in one `zarr.json`).

### v2 — two documents
- `to_json(self) -> ArrayMetadataV2` — faithful inline form (attributes folded in), wraps
  `model_dump`.
- `to_store_json(self) -> StoreV2` — `{".zarray": ZArrayMetadata, ".zattrs": ZAttrsMetadata}`.
  `StoreV2` is a small local TypedDict (zarr-metadata supplies the member types but not this
  wrapper).

### Groups
- `GroupSpec.to_json -> GroupMetadataV3` / `GroupMetadataV2`.
- v2 `GroupSpec.to_store_json -> {".zgroup": ZGroupMetadata, ".zattrs": ZAttrsMetadata}`.

### Return-type note
Both loose and strict `to_json` return the **loose** `ArrayMetadataV3` / `ArrayMetadataV2`
type: a strict-validated spec is still a valid loose document. Strict only narrows what
validates *in*, not what comes *out*.

## Compatibility

- `ArraySpec` / `GroupSpec` keep their names and loose semantics → existing user code unaffected.
- Internal-type swaps are structurally equivalent to current `model_dump` output → serialized
  dicts unchanged (snapshot-tested).
- Loose `fill_value` **widens** to `JSONValue` (additive; previously-rejected hex-float strings
  now accepted in loose mode).
- New public surface: `StrictArraySpec` (v2 + v3), `to_json`, `to_store_json`.
- `NamedConfig` retained as a deprecated alias of `NamedConfigV3`.

## Tests

1. **Round-trip:** `TypeAdapter(ArrayMetadataV3).validate_python(spec.to_json())` succeeds
   (and the v2 analogues).
2. **Strict coupling** (parametrized over core dtypes): `float64`+`"NaN"`/hex accepted;
   `int64`+`"NaN"` rejected; incomplete/foreign codec config rejected under strict, accepted
   under loose.
3. **Loose back-compat:** existing v2/v3 fixtures still validate; `model_dump` output unchanged
   (snapshot).
4. **v2 store split:** `to_store_json` keys are exactly `{".zarray", ".zattrs"}`
   (group: `{".zgroup", ".zattrs"}`); reassembling equals `to_json`.
5. **Field-drift guard:** loose and strict members share identical non-dtype/non-codec fields
   (mirrors zarr-metadata's `*Partial` equivalence test).
6. **Existing suite** stays green.

## Docs

- Update `docs/usage_zarr_v2.md` / `docs/usage_zarr_v3.md` with a strict-vs-loose section and
  `to_json` / `to_store_json` usage.
- Add a release note.

## Out of scope (possible follow-ups)

- `experimental.v2` / `experimental.v3` adoption.
- Strict coverage for string / datetime64 / timedelta64 / struct dtypes.
- A `StrictGroupSpec` enforcing strict array members within a group.
- Adopting zarr-metadata's strict per-codec/chunk-grid configs as the loose field validators.
