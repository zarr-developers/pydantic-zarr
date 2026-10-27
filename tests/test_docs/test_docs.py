from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

REPO_ROOT: Path = Path(__file__).parent.parent.parent
SOURCES_ROOT: Path = REPO_ROOT / "src/pydantic_zarr"

# fenced blocks that hold a pymdownx-snippets include directive are rendered
# from a script in examples/, which test_example_scripts executes whole; the
# directive itself is not python
DOC_EXAMPLES = tuple(example for example in find_examples("docs") if "--8<--" not in example.source)
EXAMPLE_SCRIPTS = sorted((REPO_ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("example", tuple(find_examples(str(SOURCES_ROOT))), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample) -> None:
    # honoring update_examples makes `pytest --update-examples` rewrite the
    # expected outputs in place instead of only reporting the diff
    if eval_example.update_examples:
        eval_example.run_print_update(example)
    else:
        eval_example.run_print_check(example)


@pytest.mark.parametrize("example", DOC_EXAMPLES, ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample) -> None:
    pytest.importorskip("zarr")

    if eval_example.update_examples:
        eval_example.run_print_update(example)
    else:
        eval_example.run_print_check(example)


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_scripts(script: Path) -> None:
    """Every example script is self-contained and runs cleanly, top to bottom."""
    pytest.importorskip("zarr")

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stdout}\n{result.stderr}"
