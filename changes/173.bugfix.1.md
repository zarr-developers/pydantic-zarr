The test suite now tolerates a known zarr-python/numpy interaction: under numpy 2.5,
`DateTime64`/`TimeDelta64.default_scalar()` returns a generic-unit `NaT` scalar, which numpy 2.5
deprecates. The dtype-example test fixtures exercise these types, so the (zarr-python-originating)
`DeprecationWarning` is allow-listed in the test `filterwarnings` config rather than failing
collection under the `error` filter.
