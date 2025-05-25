from __future__ import annotations

import os
from collections.abc import Mapping
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
)

import numcodecs
import numpy as np
import numpy.typing as npt
import zarr
from numcodecs.abc import Codec
from pydantic import AfterValidator, model_validator
from pydantic.functional_validators import BeforeValidator
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.storage import BaseStore, contains_array, contains_group, init_group
from zarr.util import guess_chunks

from pydantic_zarr.core import (
    IncEx,
    StrictBase,
    ensure_key_no_path,
    model_like,
)

TAttr = TypeVar("TAttr", bound=Mapping[str, Any])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])


def stringify_dtype(value: npt.DTypeLike) -> str:
    """
    Convert a `numpy.dtype` object into a `str`.

    Parameters
    ---------
    value: `npt.DTypeLike`
        Some object that can be coerced to a numpy dtype

    Returns
    -------

    A numpy dtype string representation of `value`.
    """
    return np.dtype(value).str


DtypeStr = Annotated[str, BeforeValidator(stringify_dtype)]


def dictify_codec(value: dict[str, Any] | Codec) -> dict[str, Any]:
    """
    Ensure that a `numcodecs.abc.Codec` is converted to a `dict`. If the input is not an
    instance of `numcodecs.abc.Codec`, then it is assumed to be a `dict` with string keys
    and it is returned unaltered.

    Parameters
    ---------

    value : dict[str, Any] | numcodecs.abc.Codec
        The value to be dictified if it is not already a dict.

    Returns
    -------
    dict[str, Any]
        If the input was a `Codec`, then the result of calling `get_config()` on that
        object is returned. This should be a dict with string keys. All other values pass
        through unaltered.
    """
    if isinstance(value, Codec):
        return value.get_config()
    return value


def parse_dimension_separator(data: Any) -> Literal["/", "."]:
    """
    Parse the dimension_separator metadata as per the Zarr version 2 specification.
    If the input is `None`, this returns ".".
    If the input is either "." or "/", this returns it.
    Otherwise, raises a ValueError.

    Parameters
    ----------
    data : Any
        The input data to parse.

    Returns
    -------
    Literal["/", "."]
    """
    if data is None:
        return "."
    if data in ("/", "."):
        return data
    raise ValueError(f'Invalid data, expected one of ("/", ".", None), got {data}')


CodecDict = Annotated[dict[str, Any], BeforeValidator(dictify_codec)]


class NodeSpec(StrictBase):
    """
    The base class for V2 `ArraySpec` and `GroupSpec`.

    Attributes
    ----------

    zarr_format: Literal[2]
        The Zarr version represented by this node. Must be 2.
    """

    zarr_format: Literal[2] = 2


class ArraySpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 2 Array. The specification for the data structure being modeled by
    this class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#arrays).

    Attributes
    ----------
    attributes: TAttr, default = {}
        User-defined metadata associated with this array. Should be JSON-serializable.
    shape: tuple[int, ...]
        The shape of this array.
    dtype: str
        The data type of this array.
    chunks: Tuple[int, ...]
        The chunk size for this array.
    order: "C" | "F", default = "C"
        The memory order of this array. Must be either "C", which designates "C order",
        AKA lexicographic ordering or "F", which designates "F order", AKA colexicographic ordering.
        The default is "C".
    fill_value: FillValue, default = 0
        The fill value for this array. The default is 0.
    compressor: CodecDict | None
        A JSON-serializable representation of a compression codec, or None. The default is None.
    filters: List[CodecDict] | None, default = None
        A list of JSON-serializable representations of compression codec, or None.
        The default is None.
    dimension_separator: "." | "/", default = "/"
        The character used for partitioning the different dimensions of a chunk key.
        Must be either "/" or ".", or absent, in which case it is interpreted as ".".
        The default is "/".
    """

    attributes: TAttr = cast(TAttr, {})
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr
    fill_value: int | float | None = 0
    order: Literal["C", "F"] = "C"
    filters: list[CodecDict] | None = None
    dimension_separator: Annotated[
        Literal["/", "."], BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None

    @model_validator(mode="after")
    def check_ndim(self):
        """
        Check that the `shape` and `chunks` and attributes have the same length.
        """
        if (lshape := len(self.shape)) != (lchunks := len(self.chunks)):
            msg = (
                f"Length of shape must match length of chunks. Got {lshape} elements "
                f"for shape and {lchunks} elements for chunks."
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[Any],
        chunks: Literal["auto"] | tuple[int, ...] = "auto",
        attributes: Literal["auto"] | TAttr = "auto",
        fill_value: Literal["auto"] | float | None = "auto",
        order: Literal["auto", "C", "F"] = "auto",
        filters: Literal["auto"] | list[CodecDict] | None = "auto",
        dimension_separator: Literal["auto", "/", "."] = "auto",
        compressor: Literal["auto"] | CodecDict | None = "auto",
    ) -> Self:
        """
        Create an `ArraySpec` from an array-like object. This is a convenience method for when Zarr array will be modelled from an existing array.
        This method takes nearly the same arguments as the `ArraySpec` constructor, minus `shape` and `dtype`, which will be inferred from the `array` argument.
        Additionally, this method accepts the string "auto" as a parameter for all other `ArraySpec` attributes, in which case these attributes will be
        inferred from the `array` argument, with a fallback value equal to the default `ArraySpec` parameters.

        Parameters
        ----------
        array : an array-like object
            Must have `shape` and `dtype` attributes.
            The `shape` and `dtype` of this object will be used to construct an `ArraySpec`.
        chunks : "auto" | tuple[int, ...], default = "auto"
            The chunks for this `ArraySpec`. If `chunks` is "auto" (the default), then this method first checks if `array` has a `chunksize` attribute, using it if present.
            This supports copying chunk sizes from dask arrays. If `array` does not have `chunksize`, then a routine from `zarr-python` is used to guess the chunk size,
            given the `shape` and `dtype` of `array`. If `chunks` is not auto, then it should be a tuple of ints.
        attributes : "auto" | TAttr, default = "auto"
            User-defined metadata associated with this array. Should be JSON-serializable. The default is "auto", which means that `array.attributes` will be used,
            with a fallback value of the empty dict `{}`.
        fill_value : "auto" | int | float | None, default = "auto"
            The fill value for this array. Either "auto" or FillValue. The default is "auto", which means that `array.fill_value` will be used if that attribute exists, with a fallback value of 0.
        order : "auto" | "C" | "F", default = "auto"
            The memory order of the `ArraySpec`. One of "auto", "C", or "F". The default is "auto", which means that, if present, `array.order`
            will be used, falling back to "C" if `array` does not have an `order` attribute.
        filters : "auto" | List[CodecDict] | None, default = "auto"
            The filters for this `ArraySpec`. One of "auto", a list of JSON-serializable representations of compression codec, or `None`. The default is "auto", which means that the `array.filters` attribute will be
            used, with a fallback value of `None`.
        dimension_separator : "auto" | "." | "/", default = "auto"
            Sets the character used for partitioning the different dimensions of a chunk key.
            Must be one of "auto", "/" or ".". The default is "auto", which means that `array.dimension_separator` is used, with a fallback value of "/".
        compressor : "auto" | CodecDict | None, default = "auto"
            The compressor for this `ArraySpec`. One of "auto", a JSON-serializable representation of a compression codec, or `None`. The default is "auto", which means that `array.compressor` attribute will be used, with a fallback value of `None`.

        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with `shape` and `dtype` attributes derived from `array`.

        Examples
        --------
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> import numpy as np
        >>> x = ArraySpec.from_array(np.arange(10))
        >>> x
        ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)


        """
        shape_actual = array.shape
        dtype_actual = array.dtype

        if chunks == "auto":
            chunks_actual = auto_chunks(array)
        else:
            chunks_actual = chunks

        if attributes == "auto":
            attributes_actual = auto_attributes(array)
        else:
            attributes_actual = attributes

        if fill_value == "auto":
            fill_value_actual = auto_fill_value(array)
        else:
            fill_value_actual = fill_value

        if compressor == "auto":
            compressor_actual = auto_compresser(array)
        else:
            compressor_actual = compressor

        if filters == "auto":
            filters_actual = auto_filters(array)
        else:
            filters_actual = filters

        if order == "auto":
            order_actual = auto_order(array)
        else:
            order_actual = order

        if dimension_separator == "auto":
            dimension_separator_actual = auto_dimension_separator(array)
        else:
            dimension_separator_actual = dimension_separator

        return cls(
            shape=shape_actual,
            dtype=stringify_dtype(dtype_actual),
            chunks=chunks_actual,
            attributes=attributes_actual,
            fill_value=fill_value_actual,
            order=order_actual,
            compressor=compressor_actual,
            filters=filters_actual,
            dimension_separator=dimension_separator_actual,
        )

    @classmethod
    def from_zarr(cls, array: zarr.Array) -> Self:
        """
        Create an `ArraySpec` from an instance of `zarr.Array`.

        Parameters
        ----------
        array : zarr.Array

        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with properties derived from `array`.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> ArraySpec.from_zarr(x)
        ArraySpec(zarr_format=2, attributes={}, shape=(10, 10), chunks=(10, 10), dtype='<f8', fill_value=0.0, order='C', filters=None, dimension_separator='.', compressor={'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0})

        """
        return cls(
            shape=array.shape,
            chunks=array.chunks,
            dtype=str(array.dtype),
            # explicitly cast to numpy type and back to python
            # so that int 0 isn't serialized as 0.0
            fill_value=array.dtype.type(array.fill_value).tolist(),
            order=array.order,
            filters=array.filters,
            dimension_separator=array._dimension_separator,
            compressor=array.compressor,
            attributes=array.attrs.asdict(),
        )

    def to_zarr(
        self,
        store: BaseStore,
        path: str,
        *,
        overwrite: bool = False,
        **kwargs,
    ) -> zarr.Array:
        """
        Serialize an `ArraySpec` to a Zarr array at a specific path in a Zarr store. This operation
        will create metadata documents in the store, but will not write any chunks.

        Parameters
        ----------
        store : instance of zarr.BaseStore
            The storage backend that will manifest the array.
        path : str
            The location of the array inside the store.
        overwrite : bool, default = False
            Whether to overwrite existing objects in storage to create the Zarr array.
        **kwargs : Any
            Additional keyword arguments are passed to `zarr.create`.

        Returns
        -------
        zarr.Array
            A Zarr array that is structurally identical to `self`.
        """
        spec_dict = self.model_dump(exclude={"zarr_format": True})
        attrs = spec_dict.pop("attributes")
        if self.compressor is not None:
            spec_dict["compressor"] = numcodecs.get_codec(spec_dict["compressor"])
        if self.filters is not None:
            spec_dict["filters"] = [numcodecs.get_codec(f) for f in spec_dict["filters"]]
        if contains_array(store, path):
            extant_array = zarr.open_array(store, path=path, mode="r")

            if not self.like(extant_array):
                if not overwrite:
                    msg = (
                        f"An array already exists at path {path}. "
                        "That array is structurally dissimilar to the array you are trying to "
                        "store. Call to_zarr with overwrite=True to overwrite that array."
                    )
                    raise ContainsArrayError(msg)
            else:
                if not overwrite:
                    # extant_array is read-only, so we make a new array handle that
                    # takes **kwargs
                    return zarr.open_array(
                        store=extant_array.store, path=extant_array.path, **kwargs
                    )
        result = zarr.create(store=store, path=path, overwrite=overwrite, **spec_dict, **kwargs)
        result.attrs.put(attrs)
        return result

    def like(
        self,
        other: ArraySpec | zarr.Array,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ):
        """
        Compare am `ArraySpec` to another `ArraySpec` or a `zarr.Array`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to `dict` via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other : ArraySpec | zarr.Array
            The array (model or actual) to compare with. If other is a `zarr.Array`, it will be
            converted to `ArraySpec` first.
        include : IncEx, default = None
            A specification of fields to include in the comparison. The default value is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude : IncEx, default = None
            A specification of fields to exclude from the comparison. The default value is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise, given
            the set of fields specified by the `include` and `exclude` keyword arguments.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> x.attrs.put({'foo': 10})
        >>> x_model = ArraySpec.from_zarr(x)
        >>> print(x_model.like(x_model)) # it is like itself.
        True
        >>> print(x_model.like(x))
        True
        >>> y = zarr.create((10,10))
        >>> y.attrs.put({'foo': 11}) # x and y are the same, other than their attrs
        >>> print(x_model.like(y))
        False
        >>> print(x_model.like(y, exclude={'attributes'}))
        True
        """

        other_parsed: ArraySpec
        if isinstance(other, zarr.Array):
            other_parsed = ArraySpec.from_zarr(other)
        else:
            other_parsed = other

        result = model_like(self, other_parsed, include=include, exclude=exclude)
        return result


class GroupSpec(NodeSpec, Generic[TAttr, TItem]):
    """
    A model of a Zarr Version 2 Group.
    The specification for the data structure being modeled by this
    class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#groups).

    Attributes
    ----------

    attributes: TAttr, default = {}
        The user-defined attributes of this group. Should be JSON-serializable.
    members: dict[str, TItem] | None, default = {}
        The members of this group. `members` may be `None`, which models the condition
        where the members are unknown, e.g., because they have not been discovered yet.
        If `members` is not s`None`, then it must be a dict with string keys and values that
        are either `ArraySpec` or `GroupSpec`.
    """

    attributes: TAttr = cast(TAttr, {})
    members: Annotated[dict[str, TItem] | None, AfterValidator(ensure_key_no_path)] = {}

    @classmethod
    def from_zarr(cls, group: zarr.Group, *, depth: int = -1) -> Self:
        """
        Create a GroupSpec from an instance of `zarr.Group`. Subgroups and arrays contained in the
        Zarr group will be converted to instances of `GroupSpec` and `ArraySpec`, respectively,
        and these spec instances will be stored in the `members` attribute of the parent
        `GroupSpec`.

        This is a recursive function, and the depth of recursion can be controlled by the `depth`
        keyword argument. The default value for `depth` is -1, which directs this function to
        traverse the entirety of a `zarr.Group`. This may be slow for large hierarchies, in which
        case setting `depth` to a positive integer can limit how deep into the hierarchy the
        recursion goes.

        Parameters
        ----------
        group : zarr.Group
            The Zarr group to model.
        depth : int
            An integer which may be no lower than -1. Determines how far into the tree to parse.

        Returns
        -------
        An instance of GroupSpec that represents the structure of the Zarr hierarchy.
        """

        result: GroupSpec[TAttr, TItem]
        attributes = group.attrs.asdict()
        members = {}

        if depth < -1:
            msg = (
                f"Invalid value for depth. Got {depth}, expected an integer "
                "greater than or equal to -1."
            )
            raise ValueError(msg)
        if depth == 0:
            return cls(attributes=attributes, members=None)
        new_depth = max(depth - 1, -1)
        for name, item in group.items():
            if isinstance(item, zarr.Array):
                # convert to dict before the final typed GroupSpec construction
                item_out = ArraySpec.from_zarr(item).model_dump()
            elif isinstance(item, zarr.Group):
                # convert to dict before the final typed GroupSpec construction
                item_out = GroupSpec.from_zarr(item, depth=new_depth).model_dump()
            else:
                msg = (
                    f"Unparsable object encountered: {type(item)}. Expected zarr.Array"
                    " or zarr.Group."
                )

                raise ValueError(msg)
            members[name] = item_out

        result = cls(attributes=attributes, members=members)
        return result

    def to_zarr(self, store: BaseStore, path: str, *, overwrite: bool = False, **kwargs):
        """
        Serialize this `GroupSpec` to a Zarr group at a specific path in a `zarr.BaseStore`.
        This operation will create metadata documents in the store.
        Parameters
        ----------
        store : zarr.BaseStore
            The storage backend that will manifest the group and its contents.
        path : str
            The location of the group inside the store.
        overwrite: bool, default = False
            Whether to overwrite existing objects in storage to create the Zarr group.
        **kwargs : Any
            Additional keyword arguments that will be passed to `zarr.create` for creating
            sub-arrays.
        Returns
        -------
        zarr.Group
            A zarr group that is structurally identical to `self`.

        """
        spec_dict = self.model_dump(exclude={"members": True})
        attrs = spec_dict.pop("attributes")
        if contains_group(store, path):
            extant_group = zarr.group(store, path=path)
            if not self.like(extant_group):
                if not overwrite:
                    msg = (
                        f"A group already exists at path {path}. "
                        "That group is structurally dissimilar to the group you are trying to store."
                        "Call to_zarr with overwrite=True to overwrite that group."
                    )
                    raise ContainsGroupError(msg)
            else:
                if not overwrite:
                    # if the extant group is structurally identical to self, and overwrite is false,
                    # then just return the extant group
                    return extant_group

        elif contains_array(store, path) and not overwrite:
            msg = (
                f"An array already exists at path {path}. "
                "Call to_zarr with overwrite=True to overwrite the array."
            )
            raise ContainsArrayError(msg)
        else:
            init_group(store=store, overwrite=overwrite, path=path)

        result = zarr.group(store=store, path=path, overwrite=overwrite)
        result.attrs.put(attrs)
        # consider raising an exception if a partial GroupSpec is provided
        if self.members is not None:
            for name, member in self.members.items():
                subpath = os.path.join(path, name)
                member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

        return result

    def like(
        self,
        other: GroupSpec | zarr.Group,
        include: IncEx = None,
        exclude: IncEx = None,
    ):
        """
        Compare a `GroupSpec` to another `GroupSpec` or a `zarr.Group`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to dict via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other : GroupSpec | zarr.Group
            The group (model or actual) to compare with. If other is a `zarr.Group`, it will be
            converted to a `GroupSpec`.
        include : IncEx, default = None
            A specification of fields to include in the comparison. The default is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude : IncEx, default = None
            A specification of fields to exclude from the comparison. The default is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import GroupSpec
        >>> import numpy as np
        >>> z1 = zarr.group(path='z1')
        >>> z1a = z1.array(name='foo', data=np.arange(10))
        >>> z1_model = GroupSpec.from_zarr(z1)
        >>> print(z1_model.like(z1_model)) # it is like itself
        True
        >>> print(z1_model.like(z1))
        True
        >>> z2 = zarr.group(path='z2')
        >>> z2a = z2.array(name='foo', data=np.arange(10))
        >>> print(z1_model.like(z2))
        True
        >>> z2.attrs.put({'foo' : 100}) # now they have different attributes
        >>> print(z1_model.like(z2))
        False
        >>> print(z1_model.like(z2, exclude={'attributes'}))
        True
        """

        other_parsed: GroupSpec
        if isinstance(other, zarr.Group):
            other_parsed = GroupSpec.from_zarr(other)
        else:
            other_parsed = other

        result = model_like(self, other_parsed, include=include, exclude=exclude)
        return result

    def to_flat(self, root_path: str = ""):
        """
        Flatten this `GroupSpec`.
        This method returns a `dict` with string keys and values that are `GroupSpec` or
        `ArraySpec`.

        Then the resulting `dict` will contain a copy of the input with a null `members` attribute
        under the key `root_path`, as well as copies of the result of calling `node.to_flat` on each
        element of `node.members`, each under a key created by joining `root_path` with a '/`
        character to the name of each member, and so on recursively for each sub-member.

        Parameters
        ---------
        root_path : `str`, default = ''.
            The root path. The keys in `self.members` will be
            made relative to `root_path` when used as keys in the result dictionary.

        Returns
        -------
        Dict[str, ArraySpec | GroupSpec]
            A flattened representation of the hierarchy.

        Examples
        --------

        >>> from pydantic_zarr.v2 import to_flat, GroupSpec
        >>> g1 = GroupSpec(members=None, attributes={'foo': 'bar'})
        >>> to_flat(g1)
        {'': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
        >>> to_flat(g1 root_path='baz')
        {'baz': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
        >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
        {'/g1': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None), '': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
        """
        return to_flat(self, root_path=root_path)

    @classmethod
    def from_flat(cls, data: dict[str, ArraySpec | GroupSpec]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation. The flattened hierarchy is a
        `dict` with the following constraints: keys must be valid paths; values must
        be `ArraySpec` or `GroupSpec` instances.

        Parameters
        ----------
        data : Dict[str, ArraySpec | GroupSpec]
            A flattened representation of a Zarr hierarchy.

        Returns
        -------
        GroupSpec
            A `GroupSpec` representation of the hierarchy.

        Examples
        --------
        >>> from pydantic_zarr.v2 import GroupSpec, ArraySpec
        >>> import numpy as np
        >>> flat = {'': GroupSpec(attributes={'foo': 10}, members=None)}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={})
        >>> flat = {
            '': GroupSpec(attributes={'foo': 10}, members=None),
            '/a': ArraySpec.from_array(np.arange(10))}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={'a': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
        """
        from_flated = from_flat_group(data)
        return cls(**from_flated.model_dump())


@overload
def from_zarr(element: zarr.Group) -> GroupSpec: ...


@overload
def from_zarr(element: zarr.Array) -> ArraySpec: ...


def from_zarr(element: zarr.Array | zarr.Group, depth: int = -1) -> ArraySpec | GroupSpec:
    """
    Recursively parse a `zarr.Group` or `zarr.Array` into an `ArraySpec` or `GroupSpec`.

    Parameters
    ----------
    element : zarr.Array | zarr.Group
        The `zarr.Array` or `zarr.Group` to model.
    depth : int, default = -1
        An integer which may be no lower than -1. If `element` is a `zarr.Group`, the `depth`
        parameter determines how deeply the hierarchy defined by `element` will be parsed.
        This argument has no effect if `element` is a `zarr.Array`.

    Returns
    -------
    ArraySpec | GroupSpec
        An instance of `GroupSpec` or `ArraySpec` that models the structure of the input Zarr group
        or array.
    """

    if isinstance(element, zarr.Array):
        return ArraySpec.from_zarr(element)
    else:
        return GroupSpec.from_zarr(element, depth=depth)


@overload
def to_zarr(
    spec: ArraySpec,
    store: BaseStore,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: BaseStore,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs,
) -> zarr.Group: ...


def to_zarr(
    spec: ArraySpec | GroupSpec,
    store: BaseStore,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs,
) -> zarr.Array | zarr.Group:
    """
    Serialize a `GroupSpec` or `ArraySpec` to a Zarr group or array at a specific path in
    a Zarr store.

    Parameters
    ----------
    spec : ArraySpec | GroupSpec
        The `GroupSpec` or `ArraySpec` that will be serialized to storage.
    store : zarr.BaseStore
        The storage backend that will manifest the Zarr group or array modeled by `spec`.
    path : str
        The location of the Zarr group or array inside the store.
    overwrite : bool, default = False
        Whether to overwrite existing objects in storage to create the Zarr group or array.
    **kwargs
        Additional keyword arguments will be

    Returns
    -------
    zarr.Array | zarr.Group
        A `zarr.Group` or `zarr.Array` that is structurally equivalent to `spec`.
        This operation will create metadata documents in the store.

    """
    result = spec.to_zarr(store, path, overwrite=overwrite, **kwargs)
    return result


def to_flat(node: ArraySpec | GroupSpec, root_path: str = "") -> dict[str, ArraySpec | GroupSpec]:
    """
    Flatten a `GroupSpec` or `ArraySpec`.
    Converts a `GroupSpec` or `ArraySpec` and a string, into a `dict` with string keys and
    values that are `GroupSpec` or `ArraySpec`.

    If the input is an `ArraySpec`, then this function just returns the dict `{root_path: node}`.

    If the input is a `GroupSpec`, then the resulting `dict` will contain a copy of the input
    with a null `members` attribute under the key `root_path`, as well as copies of the result of
    calling `flatten_node` on each element of `node.members`, each under a key created by joining
    `root_path` with a '/` character to the name of each member, and so on recursively for each
    sub-member.

    Parameters
    ---------
    node: `GroupSpec` | `ArraySpec`
        The node to flatten.
    root_path: `str`, default = ''.
        The root path. If `node` is a `GroupSpec`, then the keys in `node.members` will be
        made relative to `root_path` when used as keys in the result dictionary.

    Returns
    -------
    Dict[str, ArraySpec | GroupSpec]
        A flattened representation of the hierarchy.

    Examples
    --------

    >>> from pydantic_zarr.v2 import flatten, GroupSpec
    >>> g1 = GroupSpec(members=None, attributes={'foo': 'bar'})
    >>> to_flat(g1)
    {'': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
    >>> to_flat(g1 root_path='baz')
    {'baz': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
    >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
    {'/g1': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None), '': GroupSpec(zarr_format=2, attributes={'foo': 'bar'}, members=None)}
    """
    result = {}
    model_copy: ArraySpec | GroupSpec
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        model_copy = node.model_copy(deep=True, update={"members": None})
        if node.members is not None:
            for name, value in node.members.items():
                result.update(to_flat(value, "/".join([root_path, name])))

    result[root_path] = model_copy
    # sort by increasing key length
    result_sorted_keys = dict(sorted(result.items(), key=lambda v: len(v[0])))
    return result_sorted_keys


def from_flat(data: dict[str, ArraySpec | GroupSpec]) -> ArraySpec | GroupSpec:
    """
    Wraps `from_flat_group`, handling the special case where a Zarr array is defined at the root of
    a hierarchy and thus is not contained by a Zarr group.

    Parameters
    ----------

    data : Dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy. This is a `dict` with keys that are strings,
        and values that are either `GroupSpec` or `ArraySpec` instances.

    Returns
    -------
    ArraySpec | GroupSpec
        The `ArraySpec` or `GroupSpec` representation of the input data.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # special case of a Zarr array at the root of the hierarchy
    ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # note that an implicit Group is created
    GroupSpec(zarr_format=2, attributes={}, members={'foo': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """

    # minimal check that the keys are valid
    invalid_keys = []
    for key in data:
        if key.endswith("/"):
            invalid_keys.append(key)
    if len(invalid_keys) > 0:
        msg = f'Invalid keys {invalid_keys} found in data. Keys may not end with the "/"" character'
        raise ValueError(msg)

    if tuple(data.keys()) == ("",) and isinstance(tuple(data.values())[0], ArraySpec):
        return tuple(data.values())[0]
    else:
        return from_flat_group(data)


def from_flat_group(data: dict[str, ArraySpec | GroupSpec]) -> GroupSpec:
    """
    Generate a `GroupSpec` from a flat representation of a hierarchy, i.e. a `dict` with
    string keys (paths) and `ArraySpec` / `GroupSpec` values (nodes).

    Parameters
    ----------
    data : Dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy rooted at a Zarr group.

    Returns
    -------
    GroupSpec
        A `GroupSpec` that represents the hierarchy described by `data`.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat_group, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat_group(tree) # note that an implicit Group is created
    GroupSpec(zarr_format=2, attributes={}, members={'foo': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """
    root_name = ""
    sep = "/"
    # arrays that will be members of the returned GroupSpec
    member_arrays: dict[str, ArraySpec] = {}
    # groups, and their members, that will be members of the returned GroupSpec.
    # this dict is populated by recursively applying `from_flat_group` function.
    member_groups: dict[str, GroupSpec] = {}
    # this dict collects the arrayspecs and groupspecs that belong to one of the members of the
    # groupspecs we are constructing. They will later be aggregated in a recursive step that
    # populates member_groups
    submember_by_parent_name: dict[str, dict[str, ArraySpec | GroupSpec]] = {}
    # copy the input to ensure that mutations are contained inside this function
    data_copy = data.copy()
    # Get the root node
    try:
        # The root node is a GroupSpec with the key ""
        root_node = data_copy.pop(root_name)
        if isinstance(root_node, ArraySpec):
            raise ValueError("Got an ArraySpec as the root node. This is invalid.")
    except KeyError:
        # If a root node was not found, create a default one
        root_node = GroupSpec(attributes={}, members=None)

    # partition the tree (sans root node) into 2 categories: (arrays, groups + their members).
    for key, value in data_copy.items():
        key_parts = key.split(sep)
        if key_parts[0] != root_name:
            raise ValueError(f'Invalid path: {key} does not start with "{root_name}{sep}".')

        subparent_name = key_parts[1]
        if len(key_parts) == 2:
            # this is an array or group that belongs to the group we are ultimately returning
            if isinstance(value, ArraySpec):
                member_arrays[subparent_name] = value
            else:
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                submember_by_parent_name[subparent_name][root_name] = value
        else:
            # these are groups or arrays that belong to one of the member groups
            # not great that we repeat this conditional dict initialization
            if subparent_name not in submember_by_parent_name:
                submember_by_parent_name[subparent_name] = {}
            submember_by_parent_name[subparent_name][sep.join([root_name, *key_parts[2:]])] = value

    # recurse
    for subparent_name, submemb in submember_by_parent_name.items():
        member_groups[subparent_name] = from_flat_group(submemb)

    return GroupSpec(members={**member_groups, **member_arrays}, attributes=root_node.attributes)


def auto_chunks(data: Any) -> tuple[int, ...]:
    """
    Guess chunks from:
      input with a `chunksize` attribute, or
      input with a `chunks` attribute, or,
      input with `shape` and `dtype` attributes
    """
    if hasattr(data, "chunksize"):
        return data.chunksize
    if hasattr(data, "chunks"):
        return data.chunks
    return guess_chunks(data.shape, np.dtype(data.dtype).itemsize)


def auto_attributes(data: Any) -> Mapping[str, Any]:
    """
    Guess attributes from:
        input with an `attrs` attribute, or
        input with an `attributes` attribute,
        or anything (returning {})
    """
    if hasattr(data, "attrs"):
        return data.attrs
    if hasattr(data, "attributes"):
        return data.attributes
    return {}


def auto_fill_value(data: Any) -> Any:
    """
    Guess fill value from an input with a `fill_value` attribute, returning 0 otherwise.
    """
    if hasattr(data, "fill_value"):
        return data.fill_value
    return 0


def auto_compresser(data: Any) -> Codec | None:
    """
    Guess compressor from an input with a `compressor` attribute, returning `None` otherwise.
    """
    if hasattr(data, "compressor"):
        return data.compressor
    return None


def auto_filters(data: Any) -> list[Codec] | None:
    """
    Guess filters from an input with a `filters` attribute, returning `None` otherwise.
    """
    if hasattr(data, "filters"):
        return data.filters
    return None


def auto_order(data: Any) -> Literal["C", "F"]:
    """
    Guess array order from an input with an `order` attribute, returning "C" otherwise.
    """
    if hasattr(data, "order"):
        return data.order
    return "C"


def auto_dimension_separator(data: Any) -> Literal["/", "."]:
    """
    Guess dimension separator from an input with a `dimension_separator` attribute, returning "/" otherwise.
    """
    if hasattr(data, "dimension_separator"):
        return data.dimension_separator
    return "/"
