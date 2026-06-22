from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

SOURCES_ROOT: Path = Path(__file__).parent.parent.parent / "src/pydantic_zarr"


class GroupModuleGlobals:
    """Share a module namespace across consecutive doc examples in the same ``group``.

    Examples in a Markdown file may opt into shared state with a fence setting like
    ``` ```python {group="strict-v3"} ```. Blocks with the same group name then run in
    one namespace, so setup (imports, fixtures) defined in an earlier block is visible to
    later ones. This mirrors how pydantic tests its own documentation. Examples run in
    document order, so a single "current group" slot is sufficient.
    """

    def __init__(self) -> None:
        self.name: str | None = None
        self.module_dict: dict[str, Any] | None = None

    def get(self, name: str | None) -> dict[str, Any] | None:
        if name is not None and name == self.name:
            return self.module_dict
        return None

    def set(self, name: str | None, module_dict: dict[str, Any]) -> None:
        self.name = name
        self.module_dict = module_dict if name is not None else None


_group_globals = GroupModuleGlobals()


@pytest.mark.parametrize("example", tuple(find_examples(str(SOURCES_ROOT))), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample) -> None:
    eval_example.run_print_check(example)


def _published_doc_examples() -> tuple[CodeExample, ...]:
    """Doc examples from the published docs, excluding the gitignored ``docs/superpowers``
    working area (design specs / implementation plans whose code blocks are illustrative
    pseudocode, not runnable examples)."""
    superpowers = (Path("docs") / "superpowers").resolve()
    return tuple(
        ex for ex in find_examples("docs") if superpowers not in Path(ex.path).resolve().parents
    )


@pytest.mark.parametrize("example", _published_doc_examples(), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample) -> None:
    pytest.importorskip("zarr")

    group_name = example.prefix_settings().get("group")
    module_globals = _group_globals.get(group_name)
    updated = eval_example.run_print_check(example, module_globals=module_globals)
    _group_globals.set(group_name, updated)
