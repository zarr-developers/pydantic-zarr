def _zarr_python_version() -> None | str:
    """
    Returns the installed version of zarr-python, if available. Otherwise returns `None`.
    """
    try:
        import zarr

        return str(zarr.__version__)
    except ImportError:
        return None


def _require_zarr_python() -> None:
    """
    Check if zarr-python is installed. If it is, return None. If it is not, raise an ImportError.
    """
    if _zarr_python_version() is None:
        msg = (
            'The python package "zarr" is not installed. '
            "Install zarr-python with `pip install zarr`, or ",
            "re-install this package with `pip install pydantic-zarr[zarr]`, ",
        )
        raise ImportError(msg)
