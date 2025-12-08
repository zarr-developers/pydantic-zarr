# Release Notes

<!-- towncrier release notes start -->

## pydantic-zarr 0.9.0 (2025-12-08)

### Features

- Make the zarr dependency optional to allow usage without installing zarr-python. ([#112](https://github.com/zarr-developers/pydantic-zarr/issues/112))
- Add experimental module with improved implementations of `ArraySpec` and `GroupSpec` classes. ([#120](https://github.com/zarr-developers/pydantic-zarr/issues/120))

### Bugfixes

- Remove default empty dictionary for attributes in ArraySpec and GroupSpec. ([#115](https://github.com/zarr-developers/pydantic-zarr/issues/115))
- Fix a broken bare install by making the dependency on `packaging` explicit. ([#125](https://github.com/zarr-developers/pydantic-zarr/issues/125))

### Improved Documentation

- Update documentation URLs to point to pydantic-zarr.readthedocs.io. ([#123](https://github.com/zarr-developers/pydantic-zarr/issues/123))
- Add `towncrier` for managing the changelog. ([#128](https://github.com/zarr-developers/pydantic-zarr/issues/128))

### Misc

- [#121](https://github.com/zarr-developers/pydantic-zarr/issues/121)

## pydantic-zarr 0.8.4 (2025-09-09)

### Bugfixes

- Fix from_zarr for arrays with no dimension names. ([#108](https://github.com/zarr-developers/pydantic-zarr/issues/108))

### Misc

- Bump actions/setup-python from 5 to 6 in the actions group. ([#109](https://github.com/zarr-developers/pydantic-zarr/issues/109))

## pydantic-zarr 0.8.3 (2025-08-28)

### Features

- Correctly propagate dimension names from zarr arrays. ([#103](https://github.com/zarr-developers/pydantic-zarr/issues/103))
- Improve runtime type checking in from_flat(). ([#101](https://github.com/zarr-developers/pydantic-zarr/issues/101))

### Bugfixes

- Make typing of v2 from_flat() invariant. ([#100](https://github.com/zarr-developers/pydantic-zarr/issues/100))

### Improved Documentation

- Remove out of date disclaimer. ([#99](https://github.com/zarr-developers/pydantic-zarr/issues/99))

### Misc

- Bump actions/checkout from 4 to 5 in the actions group. ([#97](https://github.com/zarr-developers/pydantic-zarr/issues/97))

## pydantic-zarr 0.8.2 (2025-08-14)

### Features

- Disallow empty codecs and use a sane default in auto_codecs, allow codecs to be specified by strings. ([#95](https://github.com/zarr-developers/pydantic-zarr/issues/95))

### Bugfixes

- Fix GroupSpec.from_zarr typing. ([#91](https://github.com/zarr-developers/pydantic-zarr/issues/91))

## pydantic-zarr 0.8.1 (2025-08-05)

### Features

- Add a py.typed file for better type checking support. ([#87](https://github.com/zarr-developers/pydantic-zarr/issues/87))

### Misc

- Update cd workflow to use hatch. ([#85](https://github.com/zarr-developers/pydantic-zarr/issues/85))
- Remove GH actions doc action. ([#84](https://github.com/zarr-developers/pydantic-zarr/issues/84))

## pydantic-zarr 0.8.0 (2025-07-30)

### Features

- Use the JSON form of the fill value. ([#77](https://github.com/zarr-developers/pydantic-zarr/issues/77))
- Add types for order and dimension separator. ([#81](https://github.com/zarr-developers/pydantic-zarr/issues/81))
- Allow zarr Arrays in from_array(). ([#80](https://github.com/zarr-developers/pydantic-zarr/issues/80))
- Allow BaseModel in TBaseAttr. ([#78](https://github.com/zarr-developers/pydantic-zarr/issues/78))

### Bugfixes

- Fix readthedocs config. ([#83](https://github.com/zarr-developers/pydantic-zarr/issues/83))

## pydantic-zarr 0.7.0 (2024-03-20)

### Bugfixes

- Default chunks fix. ([#28](https://github.com/zarr-developers/pydantic-zarr/issues/28))

## pydantic-zarr 0.6.0 (2024-03-03)

### Features

- Add to_flat, from_flat, like, and better handling for existing arrays and groups. ([#25](https://github.com/zarr-developers/pydantic-zarr/issues/25))

### Improved Documentation

- Fix repo name in docs. ([#26](https://github.com/zarr-developers/pydantic-zarr/issues/26))

## pydantic-zarr 0.5.2 (2023-11-08)

### Features

- Add Zarr V3 support. ([#17](https://github.com/zarr-developers/pydantic-zarr/issues/17))

### Bugfixes

- Various bugfixes. ([#18](https://github.com/zarr-developers/pydantic-zarr/issues/18))

## pydantic-zarr 0.5.1 (2023-10-06)

### Bugfixes

- Fix GroupSpec.from_zarr. ([#16](https://github.com/zarr-developers/pydantic-zarr/issues/16))

## pydantic-zarr 0.5.0 (2023-08-22)

### Features

- Rename items to members. ([#12](https://github.com/zarr-developers/pydantic-zarr/issues/12))

### Improved Documentation

- Fix doc link. ([#11](https://github.com/zarr-developers/pydantic-zarr/issues/11))
