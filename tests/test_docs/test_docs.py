from __future__ import annotations

from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

SOURCES_ROOT: Path = Path(__file__).parent.parent.parent / "src/pydantic_zarr"


@pytest.mark.parametrize("example", find_examples(str(SOURCES_ROOT)), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample) -> None:
    eval_example.run_print_check(example)


@pytest.mark.parametrize("example", find_examples("docs"), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample) -> None:
    pytest.importorskip("zarr")

    eval_example.run_print_check(example)
