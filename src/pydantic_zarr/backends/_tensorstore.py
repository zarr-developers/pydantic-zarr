import tensorstore as ts

dataset = ts.open({'driver': 'zarr','kvstore': {'driver': 'memory'},},
    dtype=ts.uint32,
    shape=[1000, 20000],
    create=True
).result()

breakpoint()