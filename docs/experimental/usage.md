# Usage

This page demonstrates how to use the experimental `ArraySpec` and `GroupSpec` models. The code examples show both Zarr V2 and V3 implementations side-by-side.

## Creating an Array Specification

The `ArraySpec` model represents a Zarr array with its metadata and configuration.

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

    # Get the model as a dictionary
    spec_dict = array.model_dump()
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
    ```

## Creating a Group Specification

The `GroupSpec` model represents a Zarr group that can contain arrays and other groups as members.

=== "Zarr V2"

    ```python
    from pydantic_zarr.experimental.v2 import ArraySpec, GroupSpec

    # Create array specifications for group members
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

    # Create array specifications for group members
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

The `to_flat()` method converts a hierarchical group structure into a flat dictionary representation using `BaseGroupSpec` for groups without members.

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

    # Iterate over the flat representation
    for path, spec in flat.items():
        # This shows how BaseGroupSpec replaces GroupSpec in flattened form
        pass
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

    # Iterate over the flat representation
    for path, spec in flat.items():
        # This shows how BaseGroupSpec replaces GroupSpec in flattened form
        pass
    ```

## Comparing Specifications

Use the `like()` method to compare two specifications and check if they are structurally equivalent.

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

    # Compare ignoring attributes
    if array1.like(array2, exclude={'attributes'}):
        print("Arrays have the same structure")
        #> Arrays have the same structure

    # Create two groups
    group1 = GroupSpec(
        attributes={'version': 1},
        members={'data': array1}
    )

    group2 = GroupSpec(
        attributes={'version': 2},
        members={'data': array2}
    )

    # Compare groups
    if group1.like(group2, exclude={'attributes'}):
        print("Groups have the same structure")
        #> Groups have the same structure
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

    # Compare ignoring attributes
    if array1.like(array2, exclude={'attributes'}):
        print("Arrays have the same structure")
        #> Arrays have the same structure

    # Create two groups
    group1 = GroupSpec(
        attributes={'version': 1},
        members={'data': array1}
    )

    group2 = GroupSpec(
        attributes={'version': 2},
        members={'data': array2}
    )

    # Compare groups
    if group1.like(group2, exclude={'attributes'}):
        print("Groups have the same structure")
        #> Groups have the same structure
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

    # Create array specifications
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
    print("Timeseries group created successfully")
    #> Timeseries group created successfully

    # Validation enforces all required members present
    # Attempting to create without 'values' would raise ValidationError
    # Attempting to add extra members not in TypedDict would raise ValidationError
    ```

=== "Zarr V3"

    ```python
    from typing_extensions import TypedDict
    from pydantic_zarr.experimental.v3 import ArraySpec, GroupSpec

    # Define the expected structure of group members
    class TimeseriesMembers(TypedDict):
        timestamps: ArraySpec
        values: ArraySpec

    # Create array specifications
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
    print("Timeseries group created successfully")
    #> Timeseries group created successfully

    # Validation enforces all required members present
    # Attempting to create without 'values' would raise ValidationError
    # Attempting to add extra members not in TypedDict would raise ValidationError
    ```
