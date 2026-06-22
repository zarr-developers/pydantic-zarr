from __future__ import annotations

from typing import Any

from pydantic_zarr._strict_fill import is_valid_fill
from pydantic_zarr.strict.v3.codec import codec_spec_for


def _check_cast_value(meta: Any, current_dtype: str, errs: list[str]) -> None:
    cfg = meta.get("configuration", {})
    target = cfg.get("data_type")
    sm = cfg.get("scalar_map") or {}
    for direction in ("encode", "decode"):
        for entry in sm.get(direction, ()):  # entry = (input, output)
            if len(entry) == 2:
                inp, outp = entry
                if not is_valid_fill(current_dtype, inp):
                    errs.append(
                        f"cast_value {direction} input scalar {inp!r} invalid for dtype {current_dtype!r}"
                    )
                if isinstance(target, str) and not is_valid_fill(target, outp):
                    errs.append(
                        f"cast_value {direction} output scalar {outp!r} invalid for target dtype {target!r}"
                    )


def validate_pipeline(array_data_type: str, codecs: tuple[Any, ...]) -> list[str]:
    errs: list[str] = []
    current = array_data_type
    n_array_bytes = 0
    for c in codecs or ():
        spec = codec_spec_for(c)
        if spec is None:
            continue
        if spec.has_dtype_dependent_config and isinstance(c, dict):
            _check_cast_value(c, current, errs)
        if spec.kind in ("array_array", "array_bytes") and isinstance(c, dict):
            current = spec.dtype_out(c, current)
        if spec.kind == "array_bytes":
            n_array_bytes += 1
    if n_array_bytes != 1:
        errs.append(
            f"codec pipeline must have exactly one array->bytes codec, found {n_array_bytes}"
        )
    return errs
