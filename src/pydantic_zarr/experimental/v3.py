from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from importlib.metadata import version
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    NotRequired,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
from packaging.version import Version
from pydantic import BeforeValidator, Field, field_validator
from typing_extensions import TypedDict

from pydantic_zarr.experimental.core import (
    BaseAttributes,
    IncEx,
    StrictBase,
    ensure_key_no_path,
    ensure_multiple,
    maybe_node,
    model_like,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import zarr  # noqa: TC004
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams

BaseMember: TypeAlias = Mapping[str, "ArraySpec | GroupSpec"]

NodeType = Literal["group", "array"]

BoolFillValue = bool
IntFillValue = int
# todo: introduce a type that represents hexadecimal representations of floats
FloatFillValue = Literal["Infinity", "-Infinity", "NaN"] | float
ComplexFillValue = tuple[FloatFillValue, FloatFillValue]
RawFillValue = tuple[int, ...]

FillValue = BoolFillValue | IntFillValue | FloatFillValue | ComplexFillValue | RawFillValue | str

TName = TypeVar("TName", bound=str)
TConfig = TypeVar("TConfig", bound=Mapping[str, object])


class NamedConfig(TypedDict, Generic[TName, TConfig]):
    """
    A Zarr V3 metadata object.

    This class is parametrized by two type parameters: `TName` and `TConfig`.

    Attributes
    ----------
    name: TName
        The name of the metadata object.
    configuration: NotRequired[TConfig]
        The configuration of the metadata object.
    """

    name: TName
    configuration: NotRequired[TConfig]


class AnyNamedConfig(NamedConfig[str, Mapping[str, object]]):
    """
    This class models any Zarr metadata object that takes the form of a
    {"name": ..., "configuration": ...} dict, where the "configuration" key is not required.
    """


CodecLike = str | AnyNamedConfig
"""A type modelling the permissible declarations for codecs"""


class RegularChunkingConfig(TypedDict):
    chunk_shape: tuple[int, ...]


RegularChunking = NamedConfig[Literal["regular"], RegularChunkingConfig]


class DefaultChunkKeyEncodingConfig(TypedDict):
    separator: Literal[".", "/"]


DefaultChunkKeyEncoding = NamedConfig[Literal["default"], DefaultChunkKeyEncodingConfig]


class NodeSpec(StrictBase):
    """
    The base class for V3 ArraySpec and GroupSpec.

    Attributes
    ----------

    zarr_format: Literal[3]
        The Zarr version represented by this node. Must be 3.
    """

    zarr_format: Literal[3] = 3


def parse_dtype_v3(dtype: npt.DTypeLike | Mapping[str, object]) -> Mapping[str, object] | str:
    """
    Todo: refactor this when the zarr python dtypes work is released
    """
    if isinstance(dtype, str | Mapping):
        return dtype
    else:
        match np.dtype(dtype):
            case np.dtypes.Int8DType():
                return "int8"
            case np.dtypes.Int16DType():
                return "int16"
            case np.dtypes.Int32DType():
                return "int32"
            case np.dtypes.Int64DType():
                return "int64"
            case np.dtypes.UInt8DType():
                return "uint8"
            case np.dtypes.UInt16DType():
                return "uint16"
            case np.dtypes.UInt32DType():
                return "uint32"
            case np.dtypes.UInt64DType():
                return "uint64"
            case np.dtypes.Float16DType():
                return "float16"
            case np.dtypes.Float32DType():
                return "float32"
            case np.dtypes.Float16DType():
                return "float64"
            case np.dtypes.Float32DType():
                return "complex64"
            case np.dtypes.Complex128DType():
                return "complex128"
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")


DTypeStr = Annotated[str, BeforeValidator(parse_dtype_v3)]
DTypeLike = DTypeStr | AnyNamedConfig
CodecTuple = Annotated[tuple[CodecLike, ...], BeforeValidator(ensure_multiple)]


class ArraySpec(NodeSpec):
    """
    A model of a Zarr Version 3 Array.

    Attributes
    ----------

    node_type: Literal['array']
        The node type. Must be the string 'array'.
    attributes: BaseAttributes
        JSON-serializable metadata associated with this array.
    shape: Sequence[int]
        The shape of this array.
    data_type: str
        The data type of this array.
    chunk_grid: NamedConfig
        A `NamedConfig` object defining the chunk shape of this array.
    chunk_key_encoding: NamedConfig
        A `NamedConfig` object defining the chunk_key_encoding for the array.
    fill_value: FillValue
        The fill value for this array.
    codecs: Sequence[NamedConfig]
        The sequence of codices for this array.
    storage_transformers: Optional[Sequence[NamedConfig]]
        An optional sequence of `NamedConfig` objects that define the storage
        transformers for this array.
    dimension_names: Optional[Sequence[str]]
        An optional sequence of strings that gives names to each axis of the array.
    """

    node_type: Literal["array"] = "array"
    attributes: BaseAttributes = Field(default_factory=dict)  # type: ignore[arg-type]
    shape: tuple[int, ...]
    data_type: DTypeLike
    chunk_grid: RegularChunking  # todo: validate this against shape
    chunk_key_encoding: DefaultChunkKeyEncoding  # todo: validate this against shape
    fill_value: FillValue  # todo: validate this against the data type
    codecs: CodecTuple
    storage_transformers: tuple[AnyNamedConfig, ...] = ()
    dimension_names: tuple[str | None, ...] | None = None  # todo: validate this against shape

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Override this method because the Zarr V3 spec requires that the dimension_names
        field be omitted from metadata entirely if it's empty.
        """
        d = super().model_dump(**kwargs)

        if d["dimension_names"] is None:
            d.pop("dimension_names")
        return d

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[Any] | zarr.Array,
        *,
        attributes: Literal["auto"] | BaseAttributes = "auto",
        chunk_grid: Literal["auto"] | AnyNamedConfig = "auto",
        chunk_key_encoding: Literal["auto"] | AnyNamedConfig = "auto",
        fill_value: Literal["auto"] | FillValue = "auto",
        codecs: Literal["auto"] | Sequence[CodecLike] = "auto",
        storage_transformers: Literal["auto"] | Sequence[AnyNamedConfig] = "auto",
        dimension_names: Literal["auto"] | Sequence[str | None] = "auto",
    ) -> Self:
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array :
            object that conforms to the numpy array API.
            The shape and dtype of this object will be used to construct an ArraySpec.
            If the `chunks` keyword argument is not given, the shape of the array will
            be used for the chunks.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """
        if attributes == "auto":
            attributes_actual = auto_attributes(array)
        else:
            attributes_actual = attributes  # type: ignore[assignment]

        if chunk_grid == "auto":
            chunk_grid_actual = auto_chunk_grid(array)
        else:
            chunk_grid_actual = chunk_grid

        chunk_key_actual: AnyNamedConfig
        if chunk_key_encoding == "auto":
            chunk_key_actual = {"name": "default", "configuration": {"separator": "/"}}
        else:
            chunk_key_actual = chunk_key_encoding

        if fill_value == "auto":
            fill_value_actual = auto_fill_value(array)
        else:
            fill_value_actual = fill_value

        codecs_actual: tuple[CodecLike, ...]
        if codecs == "auto":
            codecs_actual = auto_codecs(array)
        else:
            codecs_actual = tuple(codecs)
        storage_transformers_actual: Sequence[AnyNamedConfig]
        if storage_transformers == "auto":
            storage_transformers_actual = auto_storage_transformers(array)
        else:
            storage_transformers_actual = storage_transformers

        dimension_names_actual: Sequence[str | None] | None
        if dimension_names == "auto":
            dimension_names_actual = auto_dimension_names(array)
        else:
            dimension_names_actual = dimension_names

        return cls(
            shape=array.shape,
            data_type=str(array.dtype),
            chunk_grid=chunk_grid_actual,
            attributes=attributes_actual,
            chunk_key_encoding=chunk_key_actual,
            fill_value=fill_value_actual,
            codecs=tuple(codecs_actual),
            storage_transformers=tuple(storage_transformers_actual),
            dimension_names=dimension_names_actual,
        )

    @classmethod
    def from_zarr(cls, array: zarr.Array) -> Self:
        """
        Create an ArraySpec from a `zarr.Array`.

        Parameters
        ----------
        array : zarr.Array

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided zarr
        array.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> ArraySpec.from_zarr(x)
        ArraySpec(zarr_format=2, attributes={}, shape=(10, 10), chunks=(10, 10), dtype='<f8', fill_value=0.0, order='C', filters=None, dimension_separator='.', compressor={'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0})

        """
        try:
            from zarr.core.metadata import ArrayV3Metadata
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        meta_json: Mapping[str, object]

        if not isinstance(array.metadata, ArrayV3Metadata):
            raise ValueError("Only zarr v3 arrays are supported")  # noqa: TRY004
        if Version(version("zarr")) < Version("3.1.0"):
            # this class was removed from zarr python 3.1.0
            from zarr.core.metadata.v3 import V3JsonEncoder  # type: ignore[attr-defined]

            meta_json = json.loads(json.dumps(array.metadata.to_dict(), cls=V3JsonEncoder))
        else:
            meta_json = array.metadata.to_dict()
        return cls(
            attributes=meta_json["attributes"],
            shape=array.shape,
            data_type=meta_json["data_type"],
            chunk_grid=meta_json["chunk_grid"],
            chunk_key_encoding=meta_json["chunk_key_encoding"],
            fill_value=meta_json["fill_value"],
            codecs=meta_json["codecs"],
            storage_transformers=meta_json["storage_transformers"],
            dimension_names=meta_json.get("dimension_names", None),
        )

    def to_zarr(
        self,
        store: Store,
        path: str,
        *,
        overwrite: bool = False,
        config: ArrayConfigParams | None = None,
    ) -> zarr.Array:
        """
        Serialize an ArraySpec to a zarr array at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.abc.store.Store
            The storage backend that will manifest the array.
        path : str
            The location of the array inside the store.
        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.
        config : ArrayConfigParams | None, default = None
            An instance of `ArrayConfigParams` that defines the runtime configuration for the array.

        Returns
        -------
        A zarr array that is structurally identical to the ArraySpec.
        This operation will create metadata documents in the store.
        """
        try:
            import zarr
            from zarr.core.array import Array, AsyncArray
            from zarr.core.metadata.v3 import ArrayV3Metadata
            from zarr.core.sync import sync
            from zarr.errors import ContainsArrayError, ContainsGroupError
            from zarr.storage._common import make_store_path
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        store_path = sync(make_store_path(store, path=path))
        extant_node = maybe_node(store, path, zarr_format=3)
        if isinstance(extant_node, zarr.Array):
            if not self.like(extant_node) and not overwrite:
                raise ContainsArrayError(store, path)
            else:
                # If there's an existing array that is identical to the model, and overwrite is False,
                # we can just return that existing array.
                if not overwrite:
                    return extant_node
        if isinstance(extant_node, zarr.Group) and not overwrite:
            raise ContainsGroupError(store, path)

        meta: ArrayV3Metadata = ArrayV3Metadata.from_dict(self.model_dump())
        async_array = AsyncArray(metadata=meta, store_path=store_path, config=config)
        sync(async_array._save_metadata(meta))
        return Array(_async_array=async_array)

    def like(
        self,
        other: ArraySpec | zarr.Array,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
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
        >>> from pydantic_zarr.v3 import ArraySpec
        >>> x = zarr.create((10,10), zarr_format=3)
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
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Array):
            other_parsed = ArraySpec.from_zarr(other)
        else:
            other_parsed = other  # type: ignore[assignment]

        return model_like(self, other_parsed, include=include, exclude=exclude)


class BaseGroupSpec(StrictBase):
    """
    A base GroupSpec class that only has core Zarr V3 group attributes
    """

    zarr_format: Literal[3] = 3
    attributes: BaseAttributes


class GroupSpec(BaseGroupSpec):
    """
    A model of a Zarr Version 3 Group.

    Attributes
    ----------

    node_type: Literal['group']
        The type of this node. Must be the string "group".
    attributes: BaseAttributes
        The user-defined attributes of this group.
    members: dict[str, ArraySpec | GroupSpec | BaseGroupSpec]
        The members of this group. This is a dict with string keys and values that
        must be ArraySpec, GroupSpec, or BaseGroupSpec instances.
    """

    node_type: Literal["group"] = "group"
    attributes: BaseAttributes
    members: BaseMember

    @field_validator("members", mode="after")
    @classmethod
    def validate_members(cls, v: BaseMember) -> BaseMember:
        return ensure_key_no_path(v)

    @classmethod
    def from_flat(cls, data: Mapping[str, ArraySpec | BaseGroupSpec]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation.

        The flattened hierarchy is a
        `dict` with the following constraints: keys must be valid paths; values must
        be `ArraySpec` or `GroupSpec` instances.

        Parameters
        ----------
        data : dict[str, ArraySpec | GroupSpec]
            A flattened representation of a Zarr hierarchy.

        Returns
        -------
        GroupSpec
            A `GroupSpec` representation of the hierarchy.

        Examples
        --------
        ```py
        from pydantic_zarr.experimental.v3 import GroupSpec, ArraySpec, BaseGroupSpec
        import numpy as np
        flat = {'': BaseGroupSpec(attributes={'foo': 10})}
        GroupSpec.from_flat(flat)
        # GroupSpec(zarr_format=3, node_type='group', attributes={'foo': 10}, members={})
        flat = {
            '': BaseGroupSpec(attributes={'foo': 10}),
            '/a': ArraySpec.from_array(np.arange(10))}
        GroupSpec.from_flat(flat)
        # GroupSpec(
        #    zarr_format=3,
        #    node_type='group',
        #    attributes={'foo': 10},
        #    members={
        #        'a': ArraySpec(
        #                zarr_format=3,
        #                node_type='array',
        #                attributes={},
        #                shape=(10,),
        #                data_type='int64',
        #                chunk_grid={'name': 'regular', 'configuration': {'chunk_shape': (10,)}},
        #                chunk_key_encoding={'name': 'default', 'configuration': {'separator': '/'}},
        #                fill_value=0,
        #                codecs=(),
        #                storage_transformers=(),
        #                dimension_names=None)})
        ```
        """
        from_flated = from_flat_group(data)
        return cls(**from_flated.model_dump())

    def to_flat(self, root_path: str = "") -> dict[str, ArraySpec | BaseGroupSpec]:
        """
        Flatten this `GroupSpec`.
        This method returns a `dict` with string keys and values that are `BaseGroupSpec` or
        `ArraySpec`.

        Then the resulting `dict` will contain a copy of the input with a null `members` attribute
        under the key `root_path`, as well as copies of the result of calling `node.to_flat` on each
        element of `node.members`, each under a key created by joining `root_path` with a '/`
        character to the name of each member, and so on recursively for each sub-member.

        Parameters
        ----------
        root_path : `str`, default = ''
            The root path. The keys in `self.members` will be
            made relative to `root_path` when used as keys in the result dictionary.

        Returns
        -------
        dict[str, ArraySpec | GroupSpec]
            A flattened representation of the hierarchy.

        Examples
        --------

        >>> from pydantic_zarr.v3 import to_flat, GroupSpec, BaseGroupSpec
        >>> g1 = GroupSpec(members={}, attributes={'foo': 'bar'})
        >>> to_flat(g1)
        {'': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
        >>> to_flat(g1, root_path='baz')
        {'baz': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
        >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
        {'/g1': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'}), '': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
        """
        return to_flat(self, root_path=root_path)

    @classmethod
    def from_zarr(cls, group: zarr.Group, *, depth: int = -1) -> Self:
        """
        Create a GroupSpec from a zarr group. Subgroups and arrays contained in the zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .members attribute of the parent
        GroupSpec. This occurs recursively, so the entire zarr hierarchy below a given
        group can be represented as a GroupSpec.

        Parameters
        ----------
        group : zarr.Group
            The Zarr group to model.
        depth : int, default = -1
            An integer which may be no lower than -1. Determines how far into the tree to parse.
            The default value of -1 indicates that the entire hierarchy should be parsed.

        Returns
        -------
        An instance of GroupSpec that represents the structure of the zarr hierarchy.
        """
        try:
            import zarr
        except ImportError as e:
            raise ImportError("zarr must be installed to use from_zarr") from e

        result: GroupSpec
        attributes = group.attrs.asdict()
        members: dict[str, ArraySpec | GroupSpec] = {}

        if depth < -1:
            msg = (
                f"Invalid value for depth. Got {depth}, expected an integer "
                "greater than or equal to -1."
            )
            raise ValueError(msg)
        if depth == 0:
            return cls(attributes=attributes, members={})
        new_depth = max(depth - 1, -1)
        for name, item in group.members():
            if isinstance(item, zarr.Array):
                members[name] = ArraySpec.from_zarr(item)
            elif isinstance(item, zarr.Group):
                members[name] = GroupSpec.from_zarr(item, depth=new_depth)
            else:
                msg = (  # type: ignore[unreachable]
                    f"Unparsable object encountered: {type(item)}. Expected zarr.Array"
                    " or zarr.Group."
                )

                raise ValueError(msg)  # noqa: TRY004

        result = cls(attributes=attributes, members=members)
        return result

    def to_zarr(
        self, store: Store, path: str, *, overwrite: bool = False, **kwargs: Any
    ) -> zarr.Group:
        """
        Serialize a GroupSpec to a zarr group at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.abc.store.Store
            The storage backend that will manifest the group and its contents.
        path : str
            The location of the group inside the store.
        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.

        Returns
        -------
        A zarr group that is structurally identical to the GroupSpec.
        This operation will create metadata documents in the store.
        """
        try:
            import zarr
            from zarr.errors import ContainsArrayError, ContainsGroupError
        except ImportError as e:
            raise ImportError("zarr must be installed to use to_zarr") from e

        spec_dict = self.model_dump(exclude={"members": True})
        attrs = spec_dict.pop("attributes")
        extant_node = maybe_node(store, path, zarr_format=3)
        if isinstance(extant_node, zarr.Group):
            if not self.like(extant_node):
                if not overwrite:
                    """
                    msg = (
                        f"A group already exists at path {path}. "
                        "That group is structurally dissimilar to the group you are trying to store. "
                        "Call `to_zarr` with `overwrite=True` to overwrite that group."
                    )
                    """
                    # TODO: use the above message when we fix the ContainsGroupError in zarr python
                    # To accept a proper message
                    raise ContainsGroupError(store, path)
            else:
                if not overwrite:
                    # if the extant group is structurally identical to self, and overwrite is false,
                    # then just return the extant group
                    return extant_node

        elif isinstance(extant_node, zarr.Array) and not overwrite:
            """
            msg = (
                f"An array already exists at path {path}. "
                "Call to_zarr with overwrite=True to overwrite the array."
            )
            """
            # TODO: use the above message when we fix the ContainsArrayError in zarr python
            raise ContainsArrayError(store, path)
        else:
            zarr.create_group(store=store, overwrite=overwrite, path=path, zarr_format=3)

        result = zarr.group(store=store, path=path, overwrite=overwrite, zarr_format=3)
        result.attrs.put(attrs)
        # consider raising an exception if a partial GroupSpec is provided
        if self.members is not None:
            for name, member in self.members.items():
                subpath = f"{path.rstrip('/')}/{name.lstrip('/')}"
                member.to_zarr(store, subpath, overwrite=overwrite, **kwargs)

        return result

    def like(
        self,
        other: GroupSpec | zarr.Group,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare a `GroupSpec` to another `GroupSpec` or a `zarr.Group`.

        This is parameterized over the fields to exclude or include in the comparison.
        Models are first converted to dict via the `model_dump` method of `pydantic.BaseModel`,
        then compared with the `==` operator.

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
        >>> from pydantic_zarr.v3 import GroupSpec
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
        if (zarr := sys.modules.get("zarr")) and isinstance(other, zarr.Group):
            other_parsed = GroupSpec.from_zarr(other)
        else:
            other_parsed = other  # type: ignore[assignment]

        return model_like(self, other_parsed, include=include, exclude=exclude)


@overload
def from_zarr(element: zarr.Array, *, depth: int = ...) -> ArraySpec: ...


@overload
def from_zarr(element: zarr.Group, *, depth: int = ...) -> GroupSpec: ...


def from_zarr(element: zarr.Array | zarr.Group, *, depth: int = -1) -> ArraySpec | GroupSpec:
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.

    Parameters
    ----------
    element : a zarr Array or zarr Group

    depth : int, default = -1
        An integer which may be no lower than -1. Determines how far into the tree to parse.
        The default value of -1 indicates that the entire hierarchy should be parsed.

    Returns
    -------
    An instance of GroupSpec or ArraySpec that represents the
    structure of the zarr group or array.
    """

    if isinstance(element, zarr.Array):
        return ArraySpec.from_zarr(element)
    else:
        return GroupSpec.from_zarr(element, depth=depth)


@overload
def to_zarr(
    spec: ArraySpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Array: ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Group: ...


def to_zarr(
    spec: ArraySpec | GroupSpec,
    store: Store,
    path: str,
    overwrite: bool = False,
) -> zarr.Array | zarr.Group:
    """
    Serialize a GroupSpec or ArraySpec to a zarr group or array at a specific path in
    a zarr store.

    Parameters
    ----------
    spec : GroupSpec or ArraySpec
        The GroupSpec or ArraySpec that will be serialized to storage.
    store : instance of zarr.abc.store.Store
        The storage backend that will manifest the group or array.
    path : str
        The location of the group or array inside the store.
    overwrite : bool
        Whether to overwrite an existing array or group at the path. If overwrite is
        False and an array or group already exists at the path, an exception will be
        raised. Defaults to False.

    Returns
    -------
    A zarr Group or Array that is structurally equivalent to the spec object.
    This operation will create metadata documents in the store.

    """
    return spec.to_zarr(store, path, overwrite=overwrite)


def from_flat(
    data: Mapping[str, ArraySpec | GroupSpec],
) -> ArraySpec | GroupSpec:
    """
    Wraps `from_flat_group`, handling the special case where a Zarr array is defined at the root of
    a hierarchy and thus is not contained by a Zarr group.

    Parameters
    ----------

    data : dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy. This is a `dict` with keys that are strings,
        and values that are either `GroupSpec` or `ArraySpec` instances.

    Returns
    -------
    ArraySpec | GroupSpec
        The `ArraySpec` or `GroupSpec` representation of the input data.

    Examples
    --------
    >>> from pydantic_zarr.v3 import from_flat, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # special case of a Zarr array at the root of the hierarchy
    ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # note that an implicit Group is created
    GroupSpec(zarr_format=2, attributes={}, members={'foo': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """

    # minimal check that the keys are valid
    invalid_keys = [key for key in data if key.endswith("/")]
    if len(invalid_keys) > 0:
        msg = f'Invalid keys {invalid_keys} found in data. Keys may not end with the "/"" character'
        raise ValueError(msg)

    if tuple(data.keys()) == ("",) and isinstance(next(iter(data.values())), ArraySpec):
        return next(iter(data.values()))
    else:
        return from_flat_group(data)


def from_flat_group(
    data: Mapping[str, ArraySpec | BaseGroupSpec],
) -> GroupSpec:
    """
    Generate a `GroupSpec` from a flat representation of a hierarchy, i.e. a `dict` with
    string keys (paths) and `ArraySpec` / `GroupSpec` values (nodes).

    Parameters
    ----------
    data : dict[str, ArraySpec | BaseGroupSpec]
        A flat representation of a Zarr hierarchy rooted at a Zarr group.

    Returns
    -------
    GroupSpec
        A `GroupSpec` that represents the hierarchy described by `data`.

    Examples
    --------
    >>> from pydantic_zarr.v3 import from_flat_group, GroupSpec, ArraySpec
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
    submember_by_parent_name: dict[str, dict[str, ArraySpec | BaseGroupSpec]] = {}
    # copy the input to ensure that mutations are contained inside this function
    data_copy = dict(data).copy()
    # Get the root node
    try:
        # The root node is a GroupSpec with the key ""
        root_node = data_copy.pop(root_name)
        if isinstance(root_node, ArraySpec):
            raise ValueError("Got an ArraySpec as the root node. This is invalid.")  # noqa: TRY004
    except KeyError:
        # If a root node was not found, create a default one
        root_node = BaseGroupSpec(attributes={})

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
            elif isinstance(value, BaseGroupSpec):
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                # Convert BaseGroupSpec to GroupSpec with empty members if needed
                if not isinstance(value, GroupSpec):
                    value = GroupSpec(attributes=value.attributes, members={})
                submember_by_parent_name[subparent_name][root_name] = value
            else:
                raise ValueError(
                    f"Value at '{key}' is not a v3 ArraySpec or BaseGroupSpec (got {type(value)=})"
                )
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


def auto_attributes(data: object) -> Mapping[str, object]:
    if hasattr(data, "attributes") and isinstance(data.attributes, Mapping):
        return data.attributes
    return {}


def auto_chunk_grid(data: object) -> AnyNamedConfig:
    if hasattr(data, "chunk_shape"):
        return {"name": "regular", "configuration": {"chunk_shape": tuple(data.chunk_shape)}}
    elif hasattr(data, "shape"):
        return {"name": "regular", "configuration": {"chunk_shape": tuple(data.shape)}}
    raise ValueError("Cannot get chunk grid from object without .shape or .chunk_shape attribute")


def auto_chunk_key_encoding(data: object) -> AnyNamedConfig:
    if hasattr(data, "metadata") and hasattr(data.metadata, "chunk_key_encoding"):
        return data.metadata.to_dict()["chunk_key_encoding"]
    return {"name": "default", "configuration": {"separator": "/"}}


def auto_fill_value(data: object) -> FillValue:
    if hasattr(data, "fill_value"):
        return data.fill_value
    elif hasattr(data, "dtype"):
        kind = np.dtype(data.dtype).kind
        if kind == "?":
            return False
        elif kind in ["i", "u"]:
            return 0
        elif kind in ["f"]:
            return "NaN"
        elif kind in ["c"]:
            return ("NaN", "NaN")
        else:
            raise ValueError(f"Cannot determine default fill value for data type {kind}")
    raise ValueError("Cannot determine default data type for object without shape attribute.")


def auto_codecs(data: object) -> tuple[CodecLike, ...]:
    """
    Automatically create a tuple of codecs from an arbitrary python object.
    """
    if hasattr(data, "codecs"):
        # todo: type check
        return tuple(data.codecs)
    return ({"name": "bytes"},)


def auto_storage_transformers(data: object) -> tuple[AnyNamedConfig, ...]:
    if hasattr(data, "storage_transformers"):
        return tuple(data.storage_transformers)
    return ()


def auto_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if hasattr(data, "metadata") and hasattr(data.metadata, "dimension_names"):
        if data.metadata.dimension_names is None:
            return None
        else:
            return tuple(data.metadata.dimension_names)
    return None


def to_flat(
    node: ArraySpec | GroupSpec, root_path: str = ""
) -> dict[str, ArraySpec | BaseGroupSpec]:
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
    ----------
    node : `GroupSpec` | `ArraySpec`
        The node to flatten.
    root_path : `str`, default = ''
        The root path. If `node` is a `GroupSpec`, then the keys in `node.members` will be
        made relative to `root_path` when used as keys in the result dictionary.

    Returns
    -------
    dict[str, ArraySpec | GroupSpec]
        A flattened representation of the hierarchy.

    Examples
    --------

    >>> from pydantic_zarr.v3 import to_flat, GroupSpec, BaseGroupSpec
    >>> g1 = GroupSpec(members={}, attributes={'foo': 'bar'})
    >>> to_flat(g1)
    {'': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
    >>> to_flat(g1, root_path='baz')
    {'baz': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
    >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
    {'/g1': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'}), '': BaseGroupSpec(zarr_format=3, attributes={'foo': 'bar'})}
    """
    result = {}
    model_copy: ArraySpec | BaseGroupSpec
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        # Create a BaseGroupSpec for the flattened representation (no members)
        model_copy = BaseGroupSpec(zarr_format=node.zarr_format, attributes=node.attributes)
        for name, value in node.members.items():
            result.update(to_flat(value, f"{root_path}/{name}"))

    result[root_path] = model_copy
    # sort by increasing key length
    return dict(sorted(result.items(), key=lambda v: len(v[0])))
