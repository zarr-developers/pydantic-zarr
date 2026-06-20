# zarr-metadata integration — design

**Date:** 2026-06-19
**Status:** Approved (brainstorm), pending implementation plan
**Scope:** Stable `pydantic_zarr.v2` / `pydantic_zarr.v3` only (not `experimental.*`)
**Release posture:** Significant release — **breaking changes are permitted**. Back-compat
shims are not required; breaks are documented in release notes / a migration guide instead.

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
| Group strictness | `StrictGroupSpec` added: members must recursively be `StrictArraySpec`/`StrictGroupSpec`. The group's own fields are unchanged. (Revised 2026-06-19: originally single-class.) |
| Strict generics | **Strict specs are NON-generic** (loose specs stay generic). `StrictArraySpec` members and `StrictGroupSpec` fix `attributes: Mapping[str, object] = {}` and members to the strict union — no `TAttr`/`TItem` params. Loose `ArraySpec[TAttr]` / `GroupSpec[TAttr, TItem]` keep their generics. Rationale: strict specs are validation targets where attribute typing adds friction without value; loose specs remain the ergonomic, typed build-and-manipulate classes. (Decided 2026-06-19.) |

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
- `GroupSpec` keeps its loose form and gains `to_json` (+ v2 `to_store_json`).
- `StrictGroupSpec` is added (v3): same group fields as `GroupSpec`, but its `members` must
  recursively be `AnyStrictArraySpec` / `StrictGroupSpec` (so members validate to the precise
  per-dtype class). The group document itself is otherwise identical; strictness propagates only
  through membership.
- **Strict specs are non-generic.** The strict per-dtype array members and `StrictGroupSpec` do
  not take `TAttr`/`TItem` parameters: `attributes: Mapping[str, object] = {}` (default `{}`,
  matching the existing loose `GroupSpec.members = {}` convention — pydantic copies the default
  per instance, so there is no shared-mutable-default bug), and strict group members are fixed
  to the `StrictArraySpec | StrictGroupSpec` union. The strict array base binds the generic
  base's parameter: `_StrictBase(_BaseArraySpec[Mapping[str, object]])`. Verified to type-check
  clean under `mypy --strict` (binding the param to its bound and adding a default is legal, not
  a narrowing error). Loose `ArraySpec[TAttr]` / `GroupSpec[TAttr, TItem]` keep their generics.
  - *Re-examined 2026-06-19 and confirmed:* making strict generic over `TAttr` was rejected on
    mechanical evidence — mypy rejects an unparametrized `Union` of generic members
    (`Missing type arguments for generic type`), which would force `StrictArraySpec[TAttr]` to be
    a generic union alias parametrized across all 15 members, pushing that complexity onto every
    caller. Users who need typed attributes use the loose `ArraySpec[MyAttrs]`; strict is a
    validation gate where `attributes: Mapping[str, object]` is sufficient.

### Strict↔loose relationship: discriminated union, not inheritance (considered & rejected alternatives)

The design question "every strict array is also a loose array — how do we express that?" was
examined in depth (2026-06-19) and resolved as follows:

- **The strict correlation fact is already expressed.** "If `data_type == "float64"` then
  `fill_value` is a float or one of the special strings" is exactly what the per-dtype
  discriminated union encodes (`_Float64ArraySpec` couples `Float64DataTypeName` with
  `Float64FillValue`). Narrowing on `data_type` gives mypy/IDEs static knowledge of the
  `fill_value` type. This static precision is the chosen, primary value of strict mode.

- **Substitutability holds at the value level, not nominally.** Every strict per-dtype
  `fill_value` type was *verified* to be a subtype of `JSONValue` under `mypy --strict`
  (`Float64FillValue`, `Int64FillValue`, `Complex64FillValue`, … all assign to `JSONValue`;
  `HexFloat64` is a `NewType(str)` ⊆ `str` ⊆ `JSONValue`). So a strict instance's fields already
  flow into any consumer typed on the loose field types. A nominal `StrictArraySpec(ArraySpec)`
  is-a relationship is therefore unnecessary for substitution.

- **Rejected: making strict inherit from loose** (`StrictArraySpec(ArraySpec)` re-typing
  `fill_value` to the per-dtype type). This is a field-narrowing override → `mypy --strict`
  `[assignment]` error. The whole sibling design exists to avoid it.

- **Rejected: nominal is-a with runtime-only enforcement** (inherit `fill_value: JSONValue`
  unnarrowed, enforce the dtype correlation in a `model_validator`). This buys nominal
  substitutability but *loses* the static `fill_value` precision — the opposite of the chosen
  priority.

- **Rejected: `Protocol` (`ArraySpecLike`) + dropping `_BaseArraySpec`.** Prototyped: a
  read-only-`@property` Protocol does type-check for frozen models, but (a) Protocols can't be
  pydantic validators (concrete models stay regardless), and (b) dropping the base forces the 5
  shared fields to be redeclared across all 16 concrete models plus either re-introduced
  `# type: ignore` (mixin) or lost `spec.to_json()` method ergonomics (free functions). The base
  is not the source of inheritance pain (narrowing was, already solved), so it stays.

**Conclusion:** keep the current design — per-dtype discriminated `StrictArraySpec`, strict/loose
as siblings over `_BaseArraySpec`. The consumer functions (`like`, `to_flat`, `from_flat`,
`flatten`, `from_flatten`) remain typed on the loose `AnyArraySpec`/`AnyGroupSpec`; widening a
specific call site to also accept a strict instance is a one-line change to make if and when a
concrete need arises, not pre-solved here.

### Strict families: Core vs Extra (SUPERSEDES the "both"-design naming below; decided 2026-06-20)

Strict mode has **two vocabulary levels**, realized as two parallel class families. They are
identical in the `data_type`↔`fill_value` coupling and in dtype coverage (the deferred extension
dtypes aren't in either yet); they differ only in which **chunk grids** and **codecs** they
accept:

| | **Core** (core Zarr v3 spec only) | **Extra** (core spec + well-known zarr extensions) |
| --- | --- | --- |
| chunk_grid | `regular` | `regular` \| `rectilinear` |
| codecs (object form) | blosc, bytes, crc32c, gzip, sharding_indexed, transpose, zstd | + scale_offset, cast_value |
| codecs (string form) | the 7 core codec **name literals** only | the 9 name literals only |
| dtypes | bool, int*, uint*, float*, complex*, raw | same (extension dtypes deferred) |

Classification is authoritative from each zarr-metadata leaf module's source URL
(`zarr-specs.readthedocs.io` = core; `github.com/zarr-developers/zarr-extensions` = extension).
zstd (a merged spec PR) is treated as core.

**Naming (no `Strict` prefix; loose `ArraySpec` keeps the short default name):**
- Loose: `ArraySpec` / `GroupSpec` (unchanged, generic, default).
- Core family: `CoreArraySpec` (single constructible), `CoreBoolArraySpec` … `CoreFloat64ArraySpec`
  … `CoreRawArraySpec` (15 per-dtype precise), `AnyCoreArraySpec` (discriminated-union validation
  target), `CoreGroupSpec` (members recursively `AnyCoreArraySpec | CoreGroupSpec`).
- Extra family: `ExtraArraySpec`, `Extra<Dtype>ArraySpec` (×15), `AnyExtraArraySpec`,
  `ExtraGroupSpec` — same structure, the Extra vocabulary.
- The interim `Strict*` / `Float64ArraySpec` / `AnyStrictArraySpec` / `StrictGroupSpec` names from
  the "both" design are **replaced** by the Core/Extra families (no compat alias).

**Structure: Approach A — plain non-generic classes (no `TGrid`/`TCodec` generics).** Each family
has its own `_CoreBase` / `_ExtraBase` (carrying that family's `chunk_grid` / `codecs` field
types) and its own 15 concrete per-dtype classes. ~30 per-dtype classes total. Chosen over a
generic `Float64Spec[TGrid, TCodec]` (which halves the class count but reintroduces generics and
produces verbose revealed types like `Float64Spec[RegularChunkGridObject, Union[...]]`),
consistent with the non-generic-strict decision. Per-dtype classes may be produced via a small
internal factory to avoid hand-writing 30 near-identical bodies, but the public result is plain
named classes.

**Codec string strictness:** the bare-string short-hand codec form is permitted (the spec allows
it) but, in strict mode, only for **known** codec names — the union of the family's
`<X>CodecName` literals (e.g. `BloscCodecName = Literal["blosc"]`), NOT arbitrary `str`. This
closes the `| str` hole where strict mode rubber-stamped unknown codec names. An unrecognized
codec name is rejected.

**Single constructible class coupling (both families):** `CoreArraySpec` / `ExtraArraySpec` carry
`data_type: str` + `fill_value: JSONValue` + a `model_validator` that (a) looks up the per-dtype
fill type by `data_type`, (b) handles raw `r<N>` by regex, (c) **rejects unrecognized
data_types**. (Same mechanism as the "both" design's single class, applied per family.)

---

### Strict spec: "both" design — one constructible class AND per-dtype precise classes

> **Naming superseded** by the Core/Extra families above — the `Strict*` names here are replaced
> by `Core*` / `Extra*`. The "both" *mechanism* (single constructible class + per-dtype precise
> classes + discriminated-union validation target) is retained, instantiated once per family.

Strict mode's defining feature is that `data_type` and `fill_value` are **coupled**: e.g. a
`float64` array may have fill value `"NaN"` / `"Infinity"` / `"-Infinity"` / a `HexFloat64`
string / a number, while an `int64` array may not have a string fill value at all. The loose
`fill_value: JSONValue` cannot express this because it does not tie `fill_value` to `data_type`.

A single discriminated union expresses the coupling with full static precision, but a union
**cannot be constructed directly** (`StrictArraySpec(...)` fails — it is a type alias, not a
class); you can only validate *into* it via `TypeAdapter`. To serve both "construct a strict
spec" and "validate with precise static types", strict mode provides **both** (decided
2026-06-20; mechanism verified end-to-end under runtime + `mypy --strict`):

1. **`StrictArraySpec` — a single, directly constructible class.** Weak field annotation
   (`fill_value: JSONValue`, `data_type: str`) plus a `model_validator(mode="after")` that
   enforces the coupling **at runtime**: it looks up the per-dtype zarr-metadata fill-value type
   by `data_type` and validates `fill_value` against it (`TypeAdapter(Float64FillValue)...`).
   `StrictArraySpec(data_type="float64", fill_value="NaN", ...)` works; `int64`+`"NaN"` raises.
   This is the **construction** path. Codecs are validated strictly here too.

2. **Per-dtype precise classes (`Float64ArraySpec`, `Int64ArraySpec`, … — PUBLIC).** One concrete
   class per core dtype, each pairing `data_type: <X>DataTypeName` with `fill_value: <X>FillValue`
   for **static** precision (mypy/IDE know `fill_value` is float-or-special-string for float64).
   Directly constructible. Public (no underscore).

   ```python
   class Float64ArraySpec(_StrictBase):
       data_type: Float64DataTypeName        # Literal["float64"]
       fill_value: Float64FillValue          # float | int | "NaN"|"Infinity"|"-Infinity" | HexFloat64
   # one per core dtype: bool, int8..uint64, float16/32/64, complex64/128, raw
   ```

3. **`AnyStrictArraySpec` — the discriminated union** over the public per-dtype classes. The
   **validation target**: `TypeAdapter(AnyStrictArraySpec).validate_python(doc)` routes by
   `data_type` to the precise per-dtype class.

   ```python
   _LiteralDtypeSpecs = Annotated[
       BoolArraySpec | Int8ArraySpec | ... | Complex128ArraySpec,
       Field(discriminator="data_type"),
   ]
   AnyStrictArraySpec = _LiteralDtypeSpecs | RawArraySpec
   ```

- **Raw dtype caveat (unchanged):** the `raw` (`r<N>`) dtype name is an open-ended `NewType(str)`,
  not a `Literal`, so `RawArraySpec` cannot be a member of a `Field(discriminator="data_type")`
  union. `AnyStrictArraySpec` is therefore a **hybrid**: the discriminated literal-dtype group
  smart-unioned with `RawArraySpec`. Verified to route all members (incl. `r8`) and reject
  mismatched fills.
- Strict codec union is composed locally from zarr-metadata per-codec types
  (`BloscCodecMetadata | GzipCodecMetadata | BytesCodecMetadata | ...` plus `str`); zarr-metadata
  does not ship a pre-built "any known codec" union. Used by both `StrictArraySpec` and the
  per-dtype classes (via the shared `_StrictBase`).
- All strict classes are **non-generic** (`attributes: Mapping[str, object] = {}`), per the
  decision above.

**Verified (2026-06-20):** the union routes `float64`+`"NaN"`→`Float64ArraySpec` and
`r8`→`RawArraySpec`; the single `StrictArraySpec` constructs directly and rejects `int64`+`"NaN"`
at runtime; the public `Float64ArraySpec` constructs directly with its precise type.

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

The local `NamedConfig` is currently a public export. It is **removed** (breaking): consumers
import `NamedConfigV3` from `zarr-metadata` instead. The break is documented in the migration
guide. No deprecated re-export alias is retained.

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

This is a significant release; **breaking changes are permitted** and documented in a
migration guide rather than softened with shims.

- `ArraySpec` / `GroupSpec` keep their names and **loose** semantics → ordinary
  construction/validation of existing user code is unaffected.
- Internal-type swaps are structurally equivalent to current `model_dump` output → serialized
  dicts unchanged (snapshot-tested, as a deliberate guard — not an absolute constraint).
- Loose `fill_value` **widens** to `JSONValue` (additive; previously-rejected hex-float strings
  now accepted in loose mode).
- New public surface: `StrictArraySpec` (v2 + v3), `to_json`, `to_store_json`.
- **Breaking:** the local `NamedConfig` export is **removed** (use `NamedConfigV3` from
  `zarr-metadata`). Other hand-rolled type exports that are replaced wholesale
  (e.g. `MemoryOrder`, `DimensionSeparator`, `CodecDict`, `RegularChunking`,
  `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding`) are likewise removed in favor of their
  zarr-metadata equivalents; any that are easy to alias may be, but no alias is required.

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
- Add a release note **and a migration guide** covering removed exports (`NamedConfig` →
  `NamedConfigV3`, etc.) and the new strict/loose distinction.

## Out of scope (possible follow-ups)

- `experimental.v2` / `experimental.v3` adoption.
- Strict coverage for string / datetime64 / timedelta64 / struct dtypes.
- Adopting zarr-metadata's strict per-codec/chunk-grid configs as the loose field validators.
