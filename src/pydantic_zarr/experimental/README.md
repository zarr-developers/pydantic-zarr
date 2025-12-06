# Experimental Module

This module contains refactored versions of the core pydantic-zarr modules with breaking API changes. It is provided for early testing and feedback on proposed changes.

## What's Different

The experimental module removes generic type parameters from `ArraySpec` and `GroupSpec`, simplifying the type system while maintaining full functionality.

### Key Changes

#### 1. No Generic Type Parameters

**Before (main module - with generics):**
```python
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from collections.abc import Mapping

# Generic type parameters allowed complex type constraints
# SpecialGroup = GroupSpec[Mapping[str, "ArraySpec | GroupSpec"]]  # Not supported in current version
```

**After (experimental module - without generics):**
```python
from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

# No generics - simpler and more straightforward
group = GroupSpec(attributes={}, members={})
print(group)
#> zarr_format=2 attributes={} members={}
```

#### 2. New `BaseGroupSpec` Class

The experimental module introduces `BaseGroupSpec` - a model of a Zarr group without members. This enables two important patterns:

- **Flattened hierarchies**: In `to_flat()` output, groups appear as `BaseGroupSpec` (without recursive members)
- **Partial loading**: Load a group's metadata without traversing its children

**Example:**
```python
from pydantic_zarr.experimental.v2 import ArraySpec, BaseGroupSpec, GroupSpec

# BaseGroupSpec: just metadata
base_group = BaseGroupSpec(attributes={"foo": "bar"})

# Create an array spec
array_spec = ArraySpec(shape=(10,), dtype='uint8', chunks=(10,), attributes={})

# GroupSpec: metadata + hierarchy
group = GroupSpec(
    attributes={"foo": "bar"},
    members={"array": array_spec}
)

# Flattened representation uses BaseGroupSpec
flat = group.to_flat()
# Returns: {"": BaseGroupSpec(...), "/array": ArraySpec(...)}
```

#### 3. Union Types Instead of Generics

Member values are now concrete union types:

**Before:**
```
members: Mapping[str, T]  # T was generic
```

**After:**
```
members: dict[str, ArraySpec | GroupSpec | BaseGroupSpec]
```

This provides:
- ✅ Better IDE autocomplete
- ✅ Clearer error messages
- ✅ No runtime type checking complexity
- ✅ More explicit code

#### 4. Refactored `to_zarr()` Method

Both `BaseGroupSpec` and `GroupSpec` have `to_zarr()` methods:

- `BaseGroupSpec.to_zarr()`: Creates a group and sets attributes (no recursion)
- `GroupSpec.to_zarr()`: Calls `super().to_zarr()` then recursively writes members

This eliminates code duplication while maintaining the inheritance hierarchy.

## API Stability

**⚠️ WARNING:** This is an experimental module. The API may change in future releases. Do not rely on it in production code without understanding the risks.

## Migration Guide

To try the experimental module:

```python
# Current (stable)
from pydantic_zarr.v2 import ArraySpec, GroupSpec

# Experimental (breaking changes)
from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec, BaseGroupSpec
```

### What Works the Same

- `ArraySpec.from_array()`
- `ArraySpec.from_zarr()` / `ArraySpec.to_zarr()`
- `GroupSpec.from_zarr()` / `GroupSpec.to_zarr()`
- `GroupSpec.to_flat()` / `GroupSpec.from_flat()`
- `.like()` comparisons
- All codec/compressor configurations
- All Zarr v2 and v3 array properties

### What Changed

- ❌ Generic type parameters no longer supported
- ✅ `BaseGroupSpec` class added
- ✅ Member types are now explicit unions
- ✅ Cleaner separation of concerns (base group vs hierarchical group)

## Rationale

The generic type parameters were:
- Not validated at runtime
- Complex to understand and use
- Provided false confidence in type safety
- Made error messages harder to read

Removing them in favor of explicit union types:
- Improves readability
- Maintains full functionality
- Reduces cognitive overhead
- Enables better error messages

The addition of `BaseGroupSpec`:
- Clarifies intent when working with flat hierarchies
- Enables efficient partial loading
- Prevents accidental null checks
- Improves code maintainability

## Testing

The experimental module passes all the same tests as the main module, with the addition of new tests for `BaseGroupSpec` functionality.

## Feedback

If you use this module and have feedback on the API changes, please open an issue on GitHub with your thoughts.
