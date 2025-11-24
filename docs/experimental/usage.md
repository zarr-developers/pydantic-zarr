# Usage

This page demonstrates how to use the experimental `ArraySpec` and `GroupSpec` models for Zarr V2 and V3.

## Creating an `ArraySpec`

The `ArraySpec` model represents Zarr array metadata.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec

    # Create a simple array specification
    array = ArraySpec(
        shape=(1000, 1000),
        dtype='uint8',
        chunks=(100, 100),
        attributes={'description': 'my array', 'units': 'meters'}
    )

    # Get the model as a JSON string
    spec_json = array.model_dump_json(indent=2)
    print(spec_json)
    """
    {
      "zarr_format": 2,
      "attributes": {
        "description": "my array",
        "units": "meters"
      },
      "shape": [
        1000,
        1000
      ],
      "chunks": [
        100,
        100
      ],
      "dtype": "|u1",
      "fill_value": 0,
      "order": "C",
      "filters": null,
      "dimension_separator": "/",
      "compressor": null
    }
    """
    ```

=== "Zarr V3"

    ```python
    from pydantic_zarr.experimental.v3 import ArraySpec

    # Create a simple array specification
    array = ArraySpec(
        shape=(1000, 1000),
        data_type='uint8',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (100, 100)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'description': 'my array', 'units': 'meters'}
    )

    # Get the model as JSON string
    spec_json = array.model_dump_json(indent=2)
    print(spec_json)
    """
    {
      "zarr_format": 3,
      "node_type": "array",
      "attributes": {
        "description": "my array",
        "units": "meters"
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
            100,
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
        }
      ],
      "storage_transformers": [],
      "dimension_names": null
    }
    """
    ```

## Creating a Group Specification

The `GroupSpec` model represents a Zarr group that can contain arrays and other groups as members.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Create ArraySpec for group members
    data_array = ArraySpec(
        shape=(1000, 1000),
        dtype='float32',
        chunks=(100, 100),
        attributes={'description': 'image data'}
    )

    metadata_array = ArraySpec(
        shape=(1000,),
        dtype='uint32',
        chunks=(100,),
        attributes={'description': 'pixel metadata'}
    )

    # Create a group containing these arrays
    group = GroupSpec(
        attributes={
            'name': 'experiment_001',
            'date': '2024-11-23',
            'version': 1
        },
        members={
            'image': data_array,
            'metadata': metadata_array
        }
    )
    ```

=== "Zarr V3"

    ```python
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Create ArraySpec for group members
    data_array = ArraySpec(
        shape=(1000, 1000),
        data_type='float32',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (100, 100)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'description': 'image data'}
    )

    metadata_array = ArraySpec(
        shape=(1000,),
        data_type='uint32',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (100,)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'description': 'pixel metadata'}
    )

    # Create a GroupSpec containing these arrays
    group = GroupSpec(
        attributes={
            'name': 'experiment_001',
            'date': '2024-11-23',
            'version': 1
        },
        members={
            'image': data_array,
            'metadata': metadata_array
        }
    )
    ```

## Nested Groups

You can create hierarchical structures by nesting groups within groups.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Create a multi-level hierarchy
    raw_data = ArraySpec(
        shape=(512, 512),
        dtype='uint8',
        chunks=(64, 64),
        attributes={}
    )

    processed_data = ArraySpec(
        shape=(512, 512),
        dtype='float32',
        chunks=(64, 64),
        attributes={}
    )

    # Create sub-groups
    raw_group = GroupSpec(
        attributes={'processing_level': 'raw'},
        members={'data': raw_data}
    )

    processed_group = GroupSpec(
        attributes={'processing_level': 'processed'},
        members={'data': processed_data}
    )

    # Create root group containing sub-groups
    root = GroupSpec(
        attributes={'project': 'imaging_study'},
        members={
            'raw': raw_group,
            'processed': processed_group
        }
    )
    ```

=== "Zarr V3"

    ```python
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Create a multi-level hierarchy
    raw_data = ArraySpec(
        shape=(512, 512),
        data_type='uint8',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (64, 64)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={}
    )

    processed_data = ArraySpec(
        shape=(512, 512),
        data_type='float32',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (64, 64)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={}
    )

    # Create sub-groups
    raw_group = GroupSpec(
        attributes={'processing_level': 'raw'},
        members={'data': raw_data}
    )

    processed_group = GroupSpec(
        attributes={'processing_level': 'processed'},
        members={'data': processed_data}
    )

    # Create root group containing sub-groups
    root = GroupSpec(
        attributes={'project': 'imaging_study'},
        members={
            'raw': raw_group,
            'processed': processed_group
        }
    )
    ```

## Working with Flattened Hierarchies

The `to_flat()` method converts a hierarchical group structure into a flat dictionary representation. In the dict form, instances of `GroupSpec` are converted to instances of `BaseGroupSpec`, which models a Zarr group without any members. We use a different type because in the flat representation, the hierarchy structure is fully encoded by the keys of the dict.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Create a group hierarchy
    array = ArraySpec(
        shape=(100,),
        dtype='float32',
        chunks=(10,),
        attributes={}
    )

    subgroup = GroupSpec(
        attributes={'level': 1},
        members={'data': array}
    )

    root = GroupSpec(
        attributes={'level': 0},
        members={'sub': subgroup}
    )

    # Convert to flat representation
    flat = root.to_flat()
    print(flat)
    """
    {
        '': BaseGroupSpec(zarr_format=2, attributes={'level': 0}),
        '/sub': BaseGroupSpec(zarr_format=2, attributes={'level': 1}),
        '/sub/data': ArraySpec(
            zarr_format=2,
            attributes={},
            shape=(100,),
            chunks=(10,),
            dtype='<f4',
            fill_value=0,
            order='C',
            filters=None,
            dimension_separator='/',
            compressor=None,
        ),
    }
    """
    ```

=== "Zarr V3"

    ```python
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Create a group hierarchy
    array = ArraySpec(
        shape=(100,),
        data_type='float32',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (10,)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={}
    )

    subgroup = GroupSpec(
        attributes={'level': 1},
        members={'data': array}
    )

    root = GroupSpec(
        attributes={'level': 0},
        members={'sub': subgroup}
    )

    # Convert to flat representation
    flat = root.to_flat()
    print(flat)
    """
    {
        '': BaseGroupSpec(zarr_format=3, attributes={'level': 0}),
        '/sub': BaseGroupSpec(zarr_format=3, attributes={'level': 1}),
        '/sub/data': ArraySpec(
            zarr_format=3,
            node_type='array',
            attributes={},
            shape=(100,),
            data_type='float32',
            chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10,)}},
            chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
            fill_value=0,
            codecs=({'name': 'bytes'},),
            storage_transformers=(),
            dimension_names=None,
        ),
    }
    """
    ```

## Comparing Arrays and Groups

Use the `like()` method to compare `ArraySpec` or `GroupSpec` instances to check if they are structurally equivalent.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Create two similar arrays
    array1 = ArraySpec(
        shape=(100, 100),
        dtype='uint8',
        chunks=(10, 10),
        attributes={'name': 'array1'}
    )

    array2 = ArraySpec(
        shape=(100, 100),
        dtype='uint8',
        chunks=(10, 10),
        attributes={'name': 'array2'}
    )

    # False because of differing attributes
    print(array1.like(array2))
    #> False

    # True because we are ignoring attributes
    print(array1.like(array2, exclude={'attributes'}))
    #> True

    # Create two groups
    group1 = GroupSpec(
        attributes={'version': 1},
        members={'data': array1}
    )

    group2 = GroupSpec(
        attributes={'version': 2},
        members={'data': array1}
    )

    # False because of differing attributes
    print(group1.like(group2))
    #> False

    # True because we are ignoring attributes
    print(group1.like(group2, exclude={'attributes'}))
    #> True
    ```

=== "Zarr V3"

    ```python
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Create two similar arrays
    array1 = ArraySpec(
        shape=(100, 100),
        data_type='uint8',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (10, 10)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'name': 'array1'}
    )

    array2 = ArraySpec(
        shape=(100, 100),
        data_type='uint8',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (10, 10)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'name': 'array2'}
    )

    # False because of differing attributes
    print(array1.like(array2))
    #> False

    # True because we are ignoring attributes
    print(array1.like(array2, exclude={'attributes'}))
    #> True

    # Create two groups
    group1 = GroupSpec(
        attributes={'version': 1},
        members={'data': array1}
    )

    group2 = GroupSpec(
        attributes={'version': 2},
        members={'data': array1}
    )

    # False because of differing attributes
    print(group1.like(group2))
    #> False

    # True because we are ignoring attributes
    print(group1.like(group2, exclude={'attributes'}))
    #> True
    ```

## Type-safe Group Members with TypedDict

Define strict schemas for group members using `TypedDict` to enable runtime validation.

=== "Zarr V2"

    ```python
    from typing_extensions import TypedDict
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Define the expected structure of group members
    class TimeseriesMembers(TypedDict):
        timestamps: ArraySpec
        values: ArraySpec

    # Create ArraySpec
    timestamps = ArraySpec(
        shape=(10000,),
        dtype='float64',
        chunks=(1000,),
        attributes={'units': 'seconds since epoch'}
    )

    values = ArraySpec(
        shape=(10000,),
        dtype='float32',
        chunks=(1000,),
        attributes={'units': 'meters'}
    )

    # Define a custom GroupSpec with typed members
    class TimeseriesGroup(GroupSpec):
        members: TimeseriesMembers

    # This succeeds - all required members present
    ts_group = TimeseriesGroup(
        attributes={'sensor': 'accelerometer'},
        members={'timestamps': timestamps, 'values': values}
    )

    # This fails because the required member 'values' is missing
    try:
        ts_group = TimeseriesGroup(
            attributes={'sensor': 'accelerometer'},
            members={'timestamps': timestamps}
        )
    except ValueError as e:
        print(e)
        """
        1 validation error for TimeseriesGroup
        members.values
          Field required [type=missing, input_value={'timestamps': ArraySpec(...r='/', compressor=None)}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/missing
        """
    ```

=== "Zarr V3"

    ```python
    from typing_extensions import TypedDict
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Define the expected structure of group members
    class TimeseriesMembers(TypedDict):
        timestamps: ArraySpec
        values: ArraySpec

    # Create ArraySpec
    timestamps = ArraySpec(
        shape=(10000,),
        data_type='float64',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (1000,)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'units': 'seconds since epoch'}
    )

    values = ArraySpec(
        shape=(10000,),
        data_type='float32',
        chunk_grid={
            'name': 'regular',
            'configuration': {'chunk_shape': (1000,)}
        },
        chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        codecs=[{'name': 'bytes'}],
        fill_value=0,
        attributes={'units': 'meters'}
    )

    # Define a custom GroupSpec with typed members
    class TimeseriesGroup(GroupSpec):
        members: TimeseriesMembers

    # This succeeds - all required members present
    ts_group = TimeseriesGroup(
        attributes={'sensor': 'accelerometer'},
        members={'timestamps': timestamps, 'values': values}
    )

    # This fails because the required member 'values' is missing
    try:
        ts_group = TimeseriesGroup(
            attributes={'sensor': 'accelerometer'},
            members={'timestamps': timestamps}
        )
    except ValueError as e:
        print(e)
        """
        1 validation error for TimeseriesGroup
        members.values
          Field required [type=missing, input_value={'timestamps': ArraySpec(..., dimension_names=None)}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/missing
        """
    ```
