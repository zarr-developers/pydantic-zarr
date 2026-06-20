# zarr-metadata Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt `zarr-metadata>=0.3.0` in stable `pydantic_zarr.v2`/`v3` to provide typed serialization (`to_json` / v2 `to_store_json`), replace hand-rolled metadata types, and add an opt-in dtype/codec-aware `StrictArraySpec`.

**Architecture:** `ArraySpec` (loose, default, syntax-only) and `StrictArraySpec` (a discriminated union of per-dtype member classes) are siblings over a behavior-only base; neither narrows the other's fields (so `mypy --strict` passes). Serialization methods live on the shared base and wrap `model_dump`, returning `zarr-metadata` TypedDicts.

**Tech Stack:** Python ≥3.12, pydantic ≥2.10 (+ pydantic.mypy plugin), numpy ≥2, `zarr-metadata>=0.3.0`, pytest, mypy --strict, ruff.

## Global Constraints

- New core runtime dependency: `zarr-metadata>=0.3.0` (add to `[project].dependencies`).
- `mypy --strict` with the pydantic plugin runs in CI + pre-commit — **no field-narrowing in subclasses** (use sibling classes over a behavior-only base).
- Breaking changes are permitted; removed exports go in a migration guide, no compat aliases.
- Keep all existing `parse_*` `BeforeValidator`s — zarr-metadata is types-only and does no coercion.
- `ArraySpec` stays **loose** (default); `StrictArraySpec` is **new/opt-in**.
- Loose `fill_value` = `JSONValue` (syntax only). Strict couples `data_type` ↔ per-dtype `fill_value`.
- Strict dtype coverage (this release): bool, int8/16/32/64, uint8/16/32/64, float16/32/64, complex64/128, raw `r<N>`. (string/datetime64/timedelta64/struct deferred.)
- Strict spec is a **hybrid union**: `Field(discriminator="data_type")` over Literal-named dtypes, smart-unioned with the `_RawArraySpec` member (raw `r<N>` is a `NewType(str)`, not a `Literal`).
- v3: `to_json` only (single `zarr.json`). v2: `to_json` (inline) + `to_store_json` (`.zarray`/`.zattrs`).
- Spec: `docs/superpowers/specs/2026-06-19-zarr-metadata-integration-design.md`.

---

## File Structure

- `pyproject.toml` — add the `zarr-metadata` dependency.
- `src/pydantic_zarr/v3.py` — replace internal types; add `to_json`; add `StrictArraySpec` + per-dtype members.
- `src/pydantic_zarr/v2.py` — replace internal types; add `to_json` + `to_store_json`.
- `src/pydantic_zarr/_strict_v3.py` *(new)* — per-dtype strict member classes + `StrictArraySpec` union (kept separate so `v3.py` stays focused; ~16 small classes would otherwise bloat it).
- `tests/test_pydantic_zarr/test_v3.py` / `test_v2.py` — update removed-symbol imports; add new tests.
- `tests/test_pydantic_zarr/test_strict_v3.py` *(new)* — strict-coupling tests.
- `docs/usage_zarr_v3.md` / `usage_zarr_v2.md`, `docs/release-notes.md`, `docs/migration.md` *(new)* — docs.

---

## Task 1: Add zarr-metadata dependency

**Files:**
- Modify: `pyproject.toml` (the `[project].dependencies` array, ~line 26)

**Interfaces:**
- Produces: `zarr_metadata` importable at runtime for all later tasks.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, change the `dependencies` line:

```toml
dependencies = ["pydantic>=2.10", "numpy>=2.0.0", "packaging>=21.0", "zarr-metadata>=0.3.0"]
```

- [ ] **Step 2: Sync the environment**

Run: `uv pip install -e . --no-deps && python -c "import zarr_metadata; print(zarr_metadata.__version__)"`
Expected: prints `0.3.0` (or newer).

- [ ] **Step 3: Verify the import the later tasks rely on**

Run: `python -c "from zarr_metadata import ArrayMetadataV3, JSONValue, MetadataV3, NamedConfigV3; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add zarr-metadata>=0.3.0 dependency"
```

---

## Task 2: v3 — replace internal loose types

Replace the hand-rolled v3 envelope/config types with zarr-metadata equivalents on the loose `ArraySpec`. This is structurally output-compatible (verified: list-based json dumps validate against `ArrayMetadataV3`).

**Files:**
- Modify: `src/pydantic_zarr/v3.py` (lines 59–128, 184–228, and `AnyNamedConfig` usages in `from_array`/`like`)
- Test: `tests/test_pydantic_zarr/test_v3.py`

**Interfaces:**
- Consumes: `zarr_metadata` (Task 1).
- Produces: loose `ArraySpec` with fields `data_type: DTypeLike`, `chunk_grid: MetadataV3`, `chunk_key_encoding: MetadataV3`, `fill_value: JSONValue`, `codecs: CodecTuple` (tuple of `MetadataV3`). Removes public exports `NamedConfig`, `AnyNamedConfig`, `RegularChunking`, `RegularChunkingConfig`, `DefaultChunkKeyEncoding`, `DefaultChunkKeyEncodingConfig`, `V2ChunkKeyEncoding`, `V2ChunkKeyEncodingConfig`, and the `FillValue` union. `CodecLike` becomes `MetadataV3`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pydantic_zarr/test_v3.py`:

```python
def test_loose_fill_value_accepts_hex_float() -> None:
    """Loose ArraySpec.fill_value is JSONValue: hex-float strings are accepted (was rejected before)."""
    spec = ArraySpec(
        attributes={},
        shape=(4,),
        data_type="float64",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value="0x7ff8000000000000",
        codecs=({"name": "bytes", "configuration": {"endian": "little"}},),
    )
    assert spec.fill_value == "0x7ff8000000000000"


def test_loose_codecs_accept_any_envelope() -> None:
    """Loose codecs accept any {name, configuration} without per-codec validation."""
    spec = ArraySpec(
        attributes={},
        shape=(4,),
        data_type="int32",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0,
        codecs=({"name": "made_up_codec", "configuration": {"whatever": 1}},),
    )
    assert spec.codecs[0]["name"] == "made_up_codec"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pydantic_zarr/test_v3.py::test_loose_fill_value_accepts_hex_float -v`
Expected: FAIL (current `fill_value` union rejects the hex string).

- [ ] **Step 3: Replace the imports and types**

In `src/pydantic_zarr/v3.py`, replace the import of `typing_extensions.TypedDict` usage region. Update the top imports to add:

```python
from zarr_metadata import JSONValue, MetadataV3, NamedConfigV3
```

Delete lines 59–75 (the `BoolFillValue`…`FillValue` block) and lines 81–128 (`NamedConfig`, `AnyNamedConfig`, `CodecLike`, `RegularChunkingConfig`, `RegularChunking`, `DefaultChunkKeyEncodingConfig`, `DefaultChunkKeyEncoding`, `V2ChunkKeyEncodingConfig`, `V2ChunkKeyEncoding`). Replace with:

```python
CodecLike = MetadataV3
```

- [ ] **Step 4: Update `DTypeLike`, `CodecTuple`, and the field annotations**

Replace line 184–185:

```python
DTypeLike = DTypeStr | NamedConfigV3
CodecTuple = Annotated[tuple[MetadataV3, ...], BeforeValidator(ensure_multiple)]
```

Replace the `ArraySpec` field block (lines ~221–228):

```python
    data_type: DTypeLike
    chunk_grid: MetadataV3  # todo: validate this against shape
    chunk_key_encoding: MetadataV3  # todo: validate this against shape
    fill_value: JSONValue  # syntax only; strict mode validates against the data type
    codecs: CodecTuple
    storage_transformers: tuple[NamedConfigV3, ...] = ()
```

- [ ] **Step 5: Replace remaining `AnyNamedConfig` references**

In `from_array`, `like`, and the `_actual` locals, replace every `AnyNamedConfig` with `NamedConfigV3` and every `FillValue` annotation with `JSONValue` (lines ~263–267, 296, 307, 312). Run:

`grep -n "AnyNamedConfig\|FillValue\b" src/pydantic_zarr/v3.py`
Expected after edits: no matches.

- [ ] **Step 6: Update the test imports that reference removed symbols**

In `tests/test_pydantic_zarr/test_v3.py`, remove `DefaultChunkKeyEncoding`, `DefaultChunkKeyEncodingConfig`, `NamedConfig`, `RegularChunking`, `RegularChunkingConfig` from the `pydantic_zarr.v3` import, and replace their uses in existing tests with plain dict literals (e.g. `{"name": "regular", "configuration": {"chunk_shape": (...)}}`).

Run: `grep -n "RegularChunking\|DefaultChunkKeyEncoding\|NamedConfig" tests/test_pydantic_zarr/test_v3.py`
Expected: no matches.

- [ ] **Step 7: Run the new tests + full v3 suite + mypy**

Run: `pytest tests/test_pydantic_zarr/test_v3.py -v && mypy --strict src/pydantic_zarr/v3.py`
Expected: all PASS; mypy clean.

- [ ] **Step 8: Commit**

```bash
git add src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_v3.py
git commit -m "refactor(v3): replace hand-rolled metadata types with zarr-metadata (BREAKING)"
```

---

## Task 3: v3 — add typed `to_json`

**Files:**
- Modify: `src/pydantic_zarr/v3.py` (`ArraySpec.model_dump` region ~242–255; add `to_json`; `GroupSpec` ~508)
- Test: `tests/test_pydantic_zarr/test_v3.py`

**Interfaces:**
- Consumes: loose `ArraySpec`/`GroupSpec` (Task 2).
- Produces: `ArraySpec.to_json(self) -> ArrayMetadataV3`; `GroupSpec.to_json(self) -> GroupMetadataV3`.

- [ ] **Step 1: Write the failing test**

```python
def test_v3_to_json_roundtrips_through_zarr_metadata() -> None:
    from pydantic import TypeAdapter
    from zarr_metadata import ArrayMetadataV3
    import numpy as np
    spec = ArraySpec.from_array(np.zeros((4, 4), dtype="int32"), attributes={"x": 1})
    doc = spec.to_json()
    # to_json output must validate as a spec-defined ArrayMetadataV3
    TypeAdapter(ArrayMetadataV3).validate_python(doc)
    assert doc["zarr_format"] == 3
    assert doc["node_type"] == "array"
    assert "dimension_names" not in doc  # omitted when None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pydantic_zarr/test_v3.py::test_v3_to_json_roundtrips_through_zarr_metadata -v`
Expected: FAIL with `AttributeError: 'ArraySpec' object has no attribute 'to_json'`.

- [ ] **Step 3: Add the import and `to_json` to `ArraySpec`**

Add to imports: `from zarr_metadata import ArrayMetadataV3, GroupMetadataV3`. Add after the existing `model_dump` override in `ArraySpec`:

```python
    def to_json(self) -> ArrayMetadataV3:
        """Serialize to a spec-defined Zarr v3 array metadata document (`zarr.json`)."""
        return cast("ArrayMetadataV3", self.model_dump(mode="json"))
```

- [ ] **Step 4: Add `to_json` to `GroupSpec`**

```python
    def to_json(self) -> GroupMetadataV3:
        """Serialize to a spec-defined Zarr v3 group metadata document (`zarr.json`)."""
        return cast("GroupMetadataV3", self.model_dump(mode="json", exclude={"members"}))
```

- [ ] **Step 5: Run the test + mypy**

Run: `pytest tests/test_pydantic_zarr/test_v3.py::test_v3_to_json_roundtrips_through_zarr_metadata -v && mypy --strict src/pydantic_zarr/v3.py`
Expected: PASS; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_v3.py
git commit -m "feat(v3): add typed to_json returning ArrayMetadataV3/GroupMetadataV3"
```

---

## Task 4: v2 — replace internal loose types

**Files:**
- Modify: `src/pydantic_zarr/v2.py` (lines 52–66, 117, `ArraySpec` fields ~167–177)
- Test: `tests/test_pydantic_zarr/test_v2.py`

**Interfaces:**
- Consumes: `zarr_metadata` (Task 1).
- Produces: loose v2 `ArraySpec` with `order: ArrayOrderV2`, `dimension_separator: ArrayDimensionSeparatorV2`, `compressor: CodecMetadataV2 | None`, `filters: tuple[CodecMetadataV2, ...] | None`, `dtype: DataTypeMetadataV2`, `fill_value: JSONValue`. Removes exports `MemoryOrder`, `DimensionSeparator`, `CodecDict`, the local `FillValue` union.

- [ ] **Step 1: Write the failing test**

```python
def test_v2_loose_fill_value_accepts_hex_float() -> None:
    spec = ArraySpec(
        attributes={}, shape=(4,), chunks=(4,), dtype="<f8",
        fill_value="0x7ff8000000000000", order="C", filters=None, compressor=None,
    )
    assert spec.fill_value == "0x7ff8000000000000"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pydantic_zarr/test_v2.py::test_v2_loose_fill_value_accepts_hex_float -v`
Expected: FAIL (current `FillValue` union rejects it).

- [ ] **Step 3: Replace imports and type aliases**

Add to imports: `from zarr_metadata import ArrayDimensionSeparatorV2, ArrayOrderV2, CodecMetadataV2, DataTypeMetadataV2, JSONValue`.

Delete lines 54–66 (`BoolFillValue`…`MemoryOrder`). Keep `parse_dimension_separator` but retype it to return `ArrayDimensionSeparatorV2`. Delete the `CodecDict` alias (line 117) and keep `dictify_codec`. Add:

```python
CodecDict = Annotated[CodecMetadataV2, BeforeValidator(dictify_codec)]
```

(Reuse the `CodecDict` name so downstream references need no change; it now wraps the zarr-metadata type.)

- [ ] **Step 4: Update `ArraySpec` fields**

Replace the field block (~167–177):

```python
    attributes: TAttr
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr | DataTypeMetadataV2
    fill_value: JSONValue = 0
    order: ArrayOrderV2 = "C"
    filters: tuple[CodecDict, ...] | None = None
    dimension_separator: Annotated[
        ArrayDimensionSeparatorV2, BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None
```

- [ ] **Step 5: Fix the `validate_filters` validator type**

Update its annotation from `list[CodecDict]` to `tuple[CodecDict, ...]` and the empty-check to `if value == ():` returning `None`.

- [ ] **Step 6: Update test imports**

In `tests/test_pydantic_zarr/test_v2.py`, remove any import of `MemoryOrder`, `DimensionSeparator`, `CodecDict` that no longer resolves; replace usages with literals/the new types.

Run: `grep -n "MemoryOrder\|DimensionSeparator" tests/test_pydantic_zarr/test_v2.py`
Expected: no matches (or only local literals).

- [ ] **Step 7: Run tests + mypy**

Run: `pytest tests/test_pydantic_zarr/test_v2.py -v && mypy --strict src/pydantic_zarr/v2.py`
Expected: PASS; mypy clean.

- [ ] **Step 8: Commit**

```bash
git add src/pydantic_zarr/v2.py tests/test_pydantic_zarr/test_v2.py
git commit -m "refactor(v2): replace hand-rolled metadata types with zarr-metadata (BREAKING)"
```

---

## Task 5: v2 — add `to_json` and `to_store_json`

**Files:**
- Modify: `src/pydantic_zarr/v2.py` (`ArraySpec` + `GroupSpec`)
- Test: `tests/test_pydantic_zarr/test_v2.py`

**Interfaces:**
- Consumes: loose v2 `ArraySpec`/`GroupSpec` (Task 4).
- Produces: `ArraySpec.to_json(self) -> ArrayMetadataV2`; `ArraySpec.to_store_json(self) -> Mapping[str, ZArrayMetadata | ZAttrsMetadata]` (keys `.zarray`, `.zattrs`); `GroupSpec.to_json(self) -> GroupMetadataV2`; `GroupSpec.to_store_json(self) -> Mapping[str, ZGroupMetadata | ZAttrsMetadata]` (keys `.zgroup`, `.zattrs`). The store-form return is a `Mapping`, **not** a TypedDict — `.zarray`/`.zattrs` are not valid TypedDict key identifiers. Do not define a `StoreArrayV2`/`StoreGroupV2` class.

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_to_json_roundtrips() -> None:
    from pydantic import TypeAdapter
    from zarr_metadata import ArrayMetadataV2
    import numpy as np
    spec = ArraySpec.from_array(np.zeros((4, 4), dtype="<f8"), attributes={"x": 1})
    doc = spec.to_json()
    TypeAdapter(ArrayMetadataV2).validate_python(doc)
    assert doc["attributes"] == {"x": 1}


def test_v2_to_store_json_splits_zarray_zattrs() -> None:
    import numpy as np
    spec = ArraySpec.from_array(np.zeros((4, 4), dtype="<f8"), attributes={"x": 1})
    store = spec.to_store_json()
    assert set(store) == {".zarray", ".zattrs"}
    assert "attributes" not in store[".zarray"]
    assert store[".zattrs"] == {"x": 1}
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pydantic_zarr/test_v2.py::test_v2_to_store_json_splits_zarray_zattrs -v`
Expected: FAIL with `AttributeError: ... 'to_store_json'`.

- [ ] **Step 3: Add imports**

Add imports: `from zarr_metadata import ArrayMetadataV2, GroupMetadataV2, ZArrayMetadata, ZAttrsMetadata, ZGroupMetadata`. No new TypedDict classes are defined — the store-form methods return a plain `Mapping[str, ...]` (see Step 4), because `.zarray`/`.zattrs` are not valid TypedDict key identifiers. `Mapping` is already imported at the top of `v2.py`.

- [ ] **Step 4: Add the methods to `ArraySpec`**

```python
    def to_json(self) -> ArrayMetadataV2:
        """Serialize to the inline v2 array metadata form (attributes folded in)."""
        return cast("ArrayMetadataV2", self.model_dump(mode="json"))

    def to_store_json(self) -> Mapping[str, ZArrayMetadata | ZAttrsMetadata]:
        """Serialize to the on-disk `.zarray` + `.zattrs` document pair."""
        full = self.model_dump(mode="json")
        attributes = full.pop("attributes", {})
        return {
            ".zarray": cast("ZArrayMetadata", full),
            ".zattrs": cast("ZAttrsMetadata", attributes),
        }
```

- [ ] **Step 5: Add `GroupSpec.to_json` / `to_store_json`**

```python
    def to_json(self) -> GroupMetadataV2:
        return cast("GroupMetadataV2", self.model_dump(mode="json", exclude={"members"}))

    def to_store_json(self) -> Mapping[str, ZGroupMetadata | ZAttrsMetadata]:
        full = self.model_dump(mode="json", exclude={"members"})
        attributes = full.pop("attributes", {})
        return {
            ".zgroup": cast("ZGroupMetadata", full),
            ".zattrs": cast("ZAttrsMetadata", attributes),
        }
```

- [ ] **Step 6: Run tests + mypy**

Run: `pytest tests/test_pydantic_zarr/test_v2.py -k "to_json or to_store_json" -v && mypy --strict src/pydantic_zarr/v2.py`
Expected: PASS; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/pydantic_zarr/v2.py tests/test_pydantic_zarr/test_v2.py
git commit -m "feat(v2): add typed to_json and to_store_json"
```

---

## Task 6: v3 — extract behavior-only `_BaseArraySpec`

Refactor the loose `ArraySpec` so all shared behavior (validators, `from_array`, `from_zarr`, `to_zarr`, `like`, `model_dump`, `to_json`) lives on `_BaseArraySpec[TAttr]`, and `ArraySpec` only declares the loose fields. This prepares for the strict siblings without changing loose behavior.

**Files:**
- Modify: `src/pydantic_zarr/v3.py`
- Test: `tests/test_pydantic_zarr/test_v3.py` (existing suite must stay green)

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: `_BaseArraySpec(NodeSpec, Generic[TAttr])` holding all methods + `attributes`, `shape`, `node_type`, `storage_transformers`, `dimension_names`. `ArraySpec(_BaseArraySpec[TAttr], Generic[TAttr])` adds `data_type`, `chunk_grid`, `chunk_key_encoding`, `fill_value`, `codecs`. Per-dtype strict classes (Task 7) subclass `_BaseArraySpec`.

- [ ] **Step 1: Confirm green baseline**

Run: `pytest tests/test_pydantic_zarr/test_v3.py -q`
Expected: all PASS (this is a pure refactor; behavior must not change).

- [ ] **Step 2: Move shared members to `_BaseArraySpec`**

Rename the current `class ArraySpec(NodeSpec, Generic[TAttr]):` to `class _BaseArraySpec(NodeSpec, Generic[TAttr]):`. Keep on it: `node_type`, `attributes`, `shape`, `storage_transformers`, `dimension_names`, `validate_dimension_names`, `model_dump`, `to_json`, `from_array`, `from_zarr`, `to_zarr`, `like`. **Remove** from it the fields `data_type`, `chunk_grid`, `chunk_key_encoding`, `fill_value`, `codecs` (they move to the subclass).

- [ ] **Step 3: Define the loose `ArraySpec` subclass**

Immediately after `_BaseArraySpec`, add:

```python
class ArraySpec(_BaseArraySpec[TAttr], Generic[TAttr]):
    """Loose Zarr v3 array spec: codecs/dtype validated as syntax only."""

    data_type: DTypeLike
    chunk_grid: MetadataV3
    chunk_key_encoding: MetadataV3
    fill_value: JSONValue
    codecs: CodecTuple
```

- [ ] **Step 4: Verify `from_array`/`like` return typing**

These classmethods use `cls(...)` and return `Self`; they stay on `_BaseArraySpec`. This pattern is verified to type-check under `mypy --strict`: a base classmethod that constructs subclass-only fields via `cls(shape=..., **kw)` returns the concrete subclass (`reveal_type` → `ArraySpec[...]`) with no override/assignment error. No body changes needed beyond keeping the kwargs flowing through.

Run: `mypy --strict src/pydantic_zarr/v3.py`
Expected: clean.

- [ ] **Step 5: Run full v3 suite**

Run: `pytest tests/test_pydantic_zarr/test_v3.py -q`
Expected: all PASS (unchanged behavior).

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_zarr/v3.py
git commit -m "refactor(v3): extract behavior-only _BaseArraySpec for strict/loose siblings"
```

---

## Task 7: v3 — strict per-dtype member classes + `StrictArraySpec` union

**Files:**
- Create: `src/pydantic_zarr/_strict_v3.py`
- Modify: `src/pydantic_zarr/v3.py` (re-export `StrictArraySpec`)
- Test: `tests/test_pydantic_zarr/test_strict_v3.py` (new)

**Interfaces:**
- Consumes: `_BaseArraySpec` (Task 6); zarr-metadata per-dtype `*DataTypeName`/`*FillValue` and per-codec `*CodecMetadata`.
- Produces: `StrictArraySpec` (public type alias) — a hybrid union routing on `data_type`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pydantic_zarr/test_strict_v3.py`:

```python
from __future__ import annotations
import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_zarr.v3 import StrictArraySpec

ADAPTER = TypeAdapter(StrictArraySpec)

def _doc(data_type: str, fill_value: object) -> dict:
    return {
        "zarr_format": 3, "node_type": "array", "data_type": data_type,
        "shape": (4,), "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }

def test_float64_accepts_nan_and_hex() -> None:
    ADAPTER.validate_python(_doc("float64", "NaN"))
    ADAPTER.validate_python(_doc("float64", "0x7ff8000000000000"))

def test_int64_rejects_nan_string() -> None:
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(_doc("int64", "NaN"))

def test_raw_dtype_routes_and_validates() -> None:
    ADAPTER.validate_python(_doc("r8", (1,)))

def test_strict_rejects_unknown_codec() -> None:
    doc = _doc("int32", 0)
    doc["codecs"] = ({"name": "made_up_codec", "configuration": {}},)
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(doc)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pydantic_zarr/test_strict_v3.py -v`
Expected: FAIL with ImportError (`StrictArraySpec` not defined).

- [ ] **Step 3: Create the strict module**

Create `src/pydantic_zarr/_strict_v3.py`:

```python
from __future__ import annotations

from typing import Annotated, Generic, TypeVar, Union

from pydantic import Field

from zarr_metadata import (
    BloscCodecMetadata, BytesCodecMetadata, Crc32cCodecMetadata, GzipCodecMetadata,
    ScaleOffsetCodecMetadata, ShardingIndexedCodecMetadata, TransposeCodecMetadata, ZstdCodecMetadata,
    BoolDataTypeName, BoolFillValue,
    Int8DataTypeName, Int8FillValue, Int16DataTypeName, Int16FillValue,
    Int32DataTypeName, Int32FillValue, Int64DataTypeName, Int64FillValue,
    Uint8DataTypeName, Uint8FillValue, Uint16DataTypeName, Uint16FillValue,
    Uint32DataTypeName, Uint32FillValue, Uint64DataTypeName, Uint64FillValue,
    Float16DataTypeName, Float16FillValue, Float32DataTypeName, Float32FillValue,
    Float64DataTypeName, Float64FillValue,
    Complex64DataTypeName, Complex64FillValue, Complex128DataTypeName, Complex128FillValue,
    RawBytesDataTypeName, RawBytesFillValue,
    RegularChunkGridMetadata, DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata,
)

from pydantic_zarr.v3 import _BaseArraySpec, TAttr

_StrictCodec = (
    BloscCodecMetadata | BytesCodecMetadata | Crc32cCodecMetadata | GzipCodecMetadata
    | ScaleOffsetCodecMetadata | ShardingIndexedCodecMetadata | TransposeCodecMetadata
    | ZstdCodecMetadata | str
)
_StrictChunkKeyEncoding = DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata


class _StrictBase(_BaseArraySpec[TAttr], Generic[TAttr]):
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: _StrictChunkKeyEncoding
    codecs: tuple[_StrictCodec, ...]


class _BoolArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: BoolDataTypeName
    fill_value: BoolFillValue

class _Int8ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Int8DataTypeName
    fill_value: Int8FillValue

# ... repeat for int16/32/64, uint8/16/32/64, float16/32/64, complex64/128
# (one class each, pairing <Dtype>DataTypeName with <Dtype>FillValue)

class _RawArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: RawBytesDataTypeName
    fill_value: RawBytesFillValue


_LiteralDtypeSpecs = Annotated[
    Union[
        _BoolArraySpec, _Int8ArraySpec, _Int16ArraySpec, _Int32ArraySpec, _Int64ArraySpec,
        _Uint8ArraySpec, _Uint16ArraySpec, _Uint32ArraySpec, _Uint64ArraySpec,
        _Float16ArraySpec, _Float32ArraySpec, _Float64ArraySpec,
        _Complex64ArraySpec, _Complex128ArraySpec,
    ],
    Field(discriminator="data_type"),
]

StrictArraySpec = Union[_LiteralDtypeSpecs, _RawArraySpec]
"""Strict Zarr v3 array spec: data_type and fill_value are coupled, codecs validated per type."""
```

Write out all 14 literal-dtype classes explicitly (do not abbreviate in the real file).

- [ ] **Step 4: Re-export from v3**

In `src/pydantic_zarr/v3.py`, at the bottom add:

```python
from pydantic_zarr._strict_v3 import StrictArraySpec as StrictArraySpec  # noqa: E402
```

(Placed at the bottom to avoid the circular import, since `_strict_v3` imports `_BaseArraySpec` from `v3`.)

- [ ] **Step 5: Run strict tests + mypy on both modules**

Run: `pytest tests/test_pydantic_zarr/test_strict_v3.py -v && mypy --strict src/pydantic_zarr/v3.py src/pydantic_zarr/_strict_v3.py`
Expected: PASS; mypy clean (no `[assignment]`/override errors — siblings, not narrowing).

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_zarr/_strict_v3.py src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_strict_v3.py
git commit -m "feat(v3): add StrictArraySpec coupling data_type with per-dtype fill_value"
```

---

## Task 8: v3 — `StrictGroupSpec` (recursive strict members)

Add a `StrictGroupSpec` whose `members` must recursively be `StrictArraySpec` or `StrictGroupSpec`. The group's own fields (`zarr_format`, `node_type`, `attributes`) are unchanged from `GroupSpec`. Strictness propagates only through membership. Verified mechanism: a self-referential `members: Mapping[str, StrictArraySpec | "StrictGroupSpec"] | None` validates nested trees and rejects a deep `int32`+`"NaN"` member.

**Files:**
- Modify: `src/pydantic_zarr/_strict_v3.py` (add `StrictGroupSpec`)
- Modify: `src/pydantic_zarr/v3.py` (re-export `StrictGroupSpec`)
- Test: `tests/test_pydantic_zarr/test_strict_v3.py`

**Interfaces:**
- Consumes: `StrictArraySpec` (Task 7); the loose `GroupSpec` for shared group-field shape.
- Produces: `class StrictGroupSpec(NodeSpec)` with `attributes: Mapping[str, object] = {}` and `members: Annotated[Mapping[str, StrictArraySpec | "StrictGroupSpec"] | None, AfterValidator(ensure_key_no_path)] = {}`; re-exported from `v3.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pydantic_zarr/test_strict_v3.py`:

```python
def test_strict_group_accepts_nested_strict_members() -> None:
    from pydantic import TypeAdapter
    from pydantic_zarr.v3 import StrictGroupSpec
    ta = TypeAdapter(StrictGroupSpec)
    doc = {
        "zarr_format": 3, "node_type": "group", "attributes": {},
        "members": {
            "sub": {
                "zarr_format": 3, "node_type": "group", "attributes": {},
                "members": {
                    "arr": {
                        "zarr_format": 3, "node_type": "array", "data_type": "int32",
                        "shape": (4,),
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
                        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                        "fill_value": 0,
                        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                    },
                },
            },
        },
    }
    ta.validate_python(doc)


def test_strict_group_rejects_nonstrict_member() -> None:
    import pytest
    from pydantic import TypeAdapter, ValidationError
    from pydantic_zarr.v3 import StrictGroupSpec
    ta = TypeAdapter(StrictGroupSpec)
    doc = {
        "zarr_format": 3, "node_type": "group", "attributes": {},
        "members": {
            "arr": {
                "zarr_format": 3, "node_type": "array", "data_type": "int32",
                "shape": (4,),
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": "NaN",  # invalid for int32 -> must propagate to a rejection
                "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
            },
        },
    }
    with pytest.raises(ValidationError):
        ta.validate_python(doc)
```

- [ ] **Step 2: Run to verify failure**

Run: `/home/d-v-b/dev/pydantic-zarr/.venv/bin/python -m pytest tests/test_pydantic_zarr/test_strict_v3.py -k strict_group -v`
Expected: FAIL with ImportError (`StrictGroupSpec` not defined).

- [ ] **Step 3: Add `StrictGroupSpec` to `_strict_v3.py`**

Append to `src/pydantic_zarr/_strict_v3.py`:

```python
from collections.abc import Mapping
from typing import Annotated, Union

from pydantic import AfterValidator

from pydantic_zarr.core import ensure_key_no_path
from pydantic_zarr.v3 import NodeSpec


class StrictGroupSpec(NodeSpec):
    """A Zarr v3 group whose members are recursively strict (StrictArraySpec/StrictGroupSpec)."""

    node_type: Literal["group"] = "group"
    attributes: Mapping[str, object] = {}
    members: Annotated[
        Mapping[str, Union[StrictArraySpec, "StrictGroupSpec"]] | None,
        AfterValidator(ensure_key_no_path),
    ] = {}


StrictGroupSpec.model_rebuild()
```

(`Literal` is already imported in this module from Task 7; add it to the import if not. `NodeSpec` carries `zarr_format: Literal[3] = 3`.)

- [ ] **Step 4: Re-export from v3**

In `src/pydantic_zarr/v3.py`, extend the bottom-of-file strict import:

```python
from pydantic_zarr._strict_v3 import StrictArraySpec as StrictArraySpec  # noqa: E402
from pydantic_zarr._strict_v3 import StrictGroupSpec as StrictGroupSpec  # noqa: E402
```

- [ ] **Step 5: Run strict tests + mypy**

Run: `/home/d-v-b/dev/pydantic-zarr/.venv/bin/python -m pytest tests/test_pydantic_zarr/test_strict_v3.py -v` then `pre-commit run mypy --files src/pydantic_zarr/_strict_v3.py src/pydantic_zarr/v3.py`
Expected: all PASS; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_zarr/_strict_v3.py src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_strict_v3.py
git commit -m "feat(v3): add StrictGroupSpec with recursive strict members"
```

---

## Task 9: Field-drift guard test

**Files:**
- Test: `tests/test_pydantic_zarr/test_strict_v3.py`

**Interfaces:**
- Consumes: `ArraySpec`, `_BaseArraySpec`, strict members (Tasks 6–7).

- [ ] **Step 1: Write the test**

```python
def test_loose_and_strict_share_base_fields() -> None:
    """Loose and strict specs must share identical non-codec/non-dtype fields."""
    from pydantic_zarr.v3 import ArraySpec, _BaseArraySpec
    from pydantic_zarr._strict_v3 import _Float64ArraySpec
    shared = set(_BaseArraySpec.model_fields)
    variant = {"data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs"}
    assert set(ArraySpec.model_fields) - variant == shared
    assert set(_Float64ArraySpec.model_fields) - variant == shared
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_pydantic_zarr/test_strict_v3.py::test_loose_and_strict_share_base_fields -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pydantic_zarr/test_strict_v3.py
git commit -m "test(v3): guard loose/strict shared-field drift"
```

---

## Task 10: Docs + migration guide + release note

**Files:**
- Modify: `docs/usage_zarr_v3.md`, `docs/usage_zarr_v2.md`, `docs/release-notes.md`
- Create: `docs/migration.md`; add it to `mkdocs.yml` nav

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Document strict vs loose + serialization**

In `docs/usage_zarr_v3.md`, add a section showing loose `ArraySpec` vs `StrictArraySpec` validation behavior (float64 `"NaN"` accepted, int64 rejected) and `to_json()`. In `docs/usage_zarr_v2.md`, document `to_json()` and `to_store_json()` (`.zarray`/`.zattrs`).

- [ ] **Step 2: Write the migration guide**

Create `docs/migration.md` listing removed exports and replacements:

```markdown
# Migration guide

This release adopts the `zarr-metadata` package. Removed symbols and replacements:

| Removed (pydantic_zarr.v3) | Use instead |
| --- | --- |
| `NamedConfig`, `AnyNamedConfig` | `zarr_metadata.NamedConfigV3` |
| `RegularChunking`, `RegularChunkingConfig` | plain dicts / `zarr_metadata.RegularChunkGridMetadata` |
| `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding` (+Config) | `zarr_metadata.DefaultChunkKeyEncodingMetadata` / `V2ChunkKeyEncodingMetadata` |
| `FillValue` union | `zarr_metadata.JSONValue` (loose) or `StrictArraySpec` (per-dtype) |

| Removed (pydantic_zarr.v2) | Use instead |
| --- | --- |
| `MemoryOrder` | `zarr_metadata.ArrayOrderV2` |
| `DimensionSeparator` | `zarr_metadata.ArrayDimensionSeparatorV2` |
| `CodecDict` (as a public type) | `zarr_metadata.CodecMetadataV2` |

New: `StrictArraySpec`, `StrictGroupSpec`, `ArraySpec.to_json()`, v2 `ArraySpec.to_store_json()`.
```

- [ ] **Step 3: Add a release note**

Append a dated entry to `docs/release-notes.md` summarizing the breaking change and new features.

- [ ] **Step 4: Build docs strict**

Run: `mkdocs build --clean --strict`
Expected: builds with no warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/ mkdocs.yml
git commit -m "docs: strict/loose usage, serialization, and migration guide"
```

---

## Task 11: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 2: Run mypy strict over the package**

Run: `mypy --strict src/pydantic_zarr tests`
Expected: clean.

- [ ] **Step 3: Run pre-commit on all files**

Run: `pre-commit run --all-files`
Expected: all hooks pass.

- [ ] **Step 4: Commit any lint fixups**

```bash
git add -A
git commit -m "chore: lint and type-check fixups for zarr-metadata integration"
```

---

## Task 12: Rework strict to the "both" design (constructible class + public per-dtype + union)

**Added 2026-06-20.** The current strict implementation (Tasks 7–8) made `StrictArraySpec` the discriminated *union* (not constructible) over *private* per-dtype classes. The maintainer chose a "both" design: a single **constructible** `StrictArraySpec` class (runtime coupling) AND **public** per-dtype precise classes, with the union renamed `AnyStrictArraySpec`. This task transforms the existing module.

**Files:**
- Modify: `src/pydantic_zarr/_strict_v3.py`
- Modify: `src/pydantic_zarr/v3.py` (re-exports)
- Modify: `tests/test_pydantic_zarr/test_strict_v3.py`

**Interfaces:**
- Produces: `StrictArraySpec` (single constructible class), public per-dtype `BoolArraySpec`/`Int8ArraySpec`/…/`Float64ArraySpec`/…/`RawArraySpec`, `AnyStrictArraySpec` (discriminated-union alias), updated `StrictGroupSpec`.

- [ ] **Step 1: Make the 15 per-dtype classes public**

In `_strict_v3.py`, rename each `_<Dtype>ArraySpec` → `<Dtype>ArraySpec` (drop leading underscore) for all 15: `BoolArraySpec`, `Int8ArraySpec`, `Int16ArraySpec`, `Int32ArraySpec`, `Int64ArraySpec`, `Uint8ArraySpec`, `Uint16ArraySpec`, `Uint32ArraySpec`, `Uint64ArraySpec`, `Float16ArraySpec`, `Float32ArraySpec`, `Float64ArraySpec`, `Complex64ArraySpec`, `Complex128ArraySpec`, `RawArraySpec`. Keep `_StrictBase` and `_StrictCodec` private. Update the union member list accordingly.

- [ ] **Step 2: Rename the union to `AnyStrictArraySpec`**

Rename the existing `StrictArraySpec = Annotated[... discriminated ...] | _RawArraySpec` alias to `AnyStrictArraySpec` (now referencing the public class names). Add a module docstring note: "`AnyStrictArraySpec` is the discriminated-union validation target; validate into it with `TypeAdapter`."

- [ ] **Step 3: Add the single constructible `StrictArraySpec` class**

Add a `data_type → FillValue` lookup table and the class:

```python
import re
from typing import Self
from pydantic import model_validator
from zarr_metadata import (
    BoolFillValue, Int8FillValue, Int16FillValue, Int32FillValue, Int64FillValue,
    Uint8FillValue, Uint16FillValue, Uint32FillValue, Uint64FillValue,
    Float16FillValue, Float32FillValue, Float64FillValue,
    Complex64FillValue, Complex128FillValue, RawBytesFillValue,
)

_RAW_DTYPE_RE = re.compile(r"^r\d+$")
_FILL_BY_DTYPE = {
    "bool": BoolFillValue,
    "int8": Int8FillValue, "int16": Int16FillValue, "int32": Int32FillValue, "int64": Int64FillValue,
    "uint8": Uint8FillValue, "uint16": Uint16FillValue, "uint32": Uint32FillValue, "uint64": Uint64FillValue,
    "float16": Float16FillValue, "float32": Float32FillValue, "float64": Float64FillValue,
    "complex64": Complex64FillValue, "complex128": Complex128FillValue,
}


class StrictArraySpec(_StrictBase):
    """A directly-constructible strict v3 array spec.

    `fill_value` is annotated loosely (`JSONValue`) but validated at runtime
    against the per-`data_type` fill-value type. An unrecognized `data_type`
    is rejected. For static per-dtype `fill_value` typing, use the public
    per-dtype classes (e.g. `Float64ArraySpec`) or validate into
    `AnyStrictArraySpec`.
    """

    data_type: str
    fill_value: JSONValue

    @model_validator(mode="after")
    def _validate_fill_matches_dtype(self) -> Self:
        ft = _FILL_BY_DTYPE.get(self.data_type)
        if ft is not None:
            TypeAdapter(ft).validate_python(self.fill_value)
        elif _RAW_DTYPE_RE.match(self.data_type):
            TypeAdapter(RawBytesFillValue).validate_python(self.fill_value)
        else:
            raise ValueError(f"Unrecognized strict data_type: {self.data_type!r}")
        return self
```

(`_StrictBase` already supplies `chunk_grid`/`chunk_key_encoding`/`codecs`/`attributes`/`shape`/etc. `TypeAdapter` is imported from pydantic.)

- [ ] **Step 4: Update `StrictGroupSpec.members`**

Change the members union from `Union[StrictArraySpec, "StrictGroupSpec"]` to `Union[AnyStrictArraySpec, "StrictGroupSpec"]`, then `StrictGroupSpec.model_rebuild()`. (Members validate to the precise per-dtype class.)

- [ ] **Step 5: Update re-exports in `v3.py`**

At the bottom of `v3.py`, re-export: `StrictArraySpec`, `AnyStrictArraySpec`, `StrictGroupSpec`, and all 15 public per-dtype classes (`from pydantic_zarr._strict_v3 import (... ) ` with `# noqa: E402`). Keep the names alphabetized or grouped.

- [ ] **Step 6: Update + extend tests**

In `test_strict_v3.py`:
- Replace `TypeAdapter(StrictArraySpec)` validation-target usages with `TypeAdapter(AnyStrictArraySpec)`.
- Add: direct construction of `StrictArraySpec(data_type="float64", fill_value="NaN", shape=(4,), chunk_grid=..., chunk_key_encoding=..., codecs=...)` succeeds; `StrictArraySpec(data_type="int64", fill_value="NaN", ...)` raises `ValidationError`; an unrecognized `data_type="float128"` raises; `data_type="r8"` with a bytes-tuple fill succeeds and with `"NaN"` raises.
- Add: a public per-dtype class constructs directly: `Float64ArraySpec(data_type="float64", fill_value="Infinity", shape=(4,), ...)`.
- Update the drift-guard test (Task 9) to reference a public per-dtype class (`Float64ArraySpec`) instead of `_Float64ArraySpec`.

- [ ] **Step 7: Verify**

Run: `/home/d-v-b/dev/pydantic-zarr/.venv/bin/python -m pytest tests/test_pydantic_zarr/test_strict_v3.py tests/test_pydantic_zarr/test_v3.py -q` (all pass) and `pre-commit run mypy ruff --files src/pydantic_zarr/_strict_v3.py src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_strict_v3.py` (clean, no new ignores).

- [ ] **Step 8: Commit**

```bash
git add src/pydantic_zarr/_strict_v3.py src/pydantic_zarr/v3.py tests/test_pydantic_zarr/test_strict_v3.py
git commit -m "feat(v3): 'both' strict design — constructible StrictArraySpec + public per-dtype + AnyStrictArraySpec"
```

> **Note:** Task 10 (docs) must be redone AFTER this task, since it documented the old union-only `StrictArraySpec`. The docs need all three paths: constructible `StrictArraySpec`, public per-dtype classes, and `AnyStrictArraySpec` as the validation target.

---

## Self-Review notes

- **Spec coverage:** dependency (T1), v3 type replacement (T2), v3 to_json (T3), v2 type replacement (T4), v2 to_json/to_store_json (T5), base extraction (T6), strict union incl. raw hybrid (T7), StrictGroupSpec recursive members (T8), drift guard (T9), docs+migration (T10), full gate (T11). All spec sections mapped.
- **Pre-verified mechanics (empirically, during planning):** (1) loose json dumps validate against `ArrayMetadataV3`; (2) sibling-over-base classes type-check clean under `mypy --strict` + pydantic plugin (no field narrowing); (3) the hybrid discriminated+raw union routes all dtypes incl. `r8` and rejects mismatched fills; (4) a base classmethod constructing subclass-only fields returns the concrete subclass under mypy strict with no override error.
- **Type-consistency check:** `to_json`/`to_store_json` signatures, `_BaseArraySpec`, `StrictArraySpec`, `_StrictCodec`, and the per-dtype class names are used consistently across T3/T5/T6/T7/T8. v2 store-return is `Mapping[str, ...]` (TypedDict can't express `.zarray` keys) — noted in T5 Step 3.
