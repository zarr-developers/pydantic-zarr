# Project actions for pydantic-zarr. Run `just --list` to see all recipes.

# Default python / zarr versions for single-environment recipes.
# Override like `just py=3.13 zarr=3.0.10 test`.
py := "3.12"
zarr := "3.1.0"

# Every environment is installed from uv.lock by this one uv, pinned so that
# CI and a development machine resolve, install and check the lockfile the
# same way (a newer uv may write or read the lock differently).
# `--locked` fails if uv.lock is stale; `--exact` removes what the recipe's
# groups do not list, so a test without zarr really runs without it.
uv := "uvx uv@0.13.5"
run := uv + " run --locked --exact --no-default-groups --python " + py

# The python/zarr matrix CI tests: `just test-all` runs every cell.
pythons := "3.12 3.13 3.14"
zarrs := "3.0.10 3.1.0 3.3.0 3.4.1"

# List available recipes
default:
    @just --list

# Install the git hooks (with prek)
setup:
    prek install

# Run all linters and formatters (the hooks in .pre-commit-config.yaml, via prek)
lint:
    prek run --all-files --show-diff-on-failure

# Run the type checkers (pyrefly, ty and pyright; see the `types` group)
typecheck:
    {{ run }} --group types pyrefly check
    {{ run }} --group types ty check src/pydantic_zarr/core.py src/pydantic_zarr/_metadata.py tests/typing_conformance.py
    {{ run }} --group types pyright tests/typing_conformance.py

# The zarr-python a test cell runs on: the locked one, replaced by `zarr~=<zarr>`
_zarr := "--with zarr~=" + zarr

# Run the test suite against one python/zarr combo, passing extra args to pytest
test *args:
    {{ run }} --group test {{ _zarr }} pytest tests/test_pydantic_zarr/ {{ args }}

# Run the test suite with coverage against one python/zarr combo
test-cov:
    {{ run }} --group test {{ _zarr }} python -c "{{ _list_env }}"
    {{ run }} --group test {{ _zarr }} pytest --cov-config=pyproject.toml --cov=pkg --cov-report html --cov=src tests/test_pydantic_zarr

# Run the test suite without zarr installed
test-base *args:
    {{ run }} --group test-base pytest tests/test_pydantic_zarr/ {{ args }}

# Run the test suite without zarr installed, with coverage
test-base-cov:
    {{ run }} --group test-base python -c "{{ _list_env }}"
    {{ run }} --group test-base pytest --cov-config=pyproject.toml --cov=pkg --cov-report html --cov=src tests/test_pydantic_zarr

# Run the test suite across the full python/zarr matrix
test-all:
    for p in {{ pythons }}; do for z in {{ zarrs }}; do just py=$p zarr=$z test || exit 1; done; done

# Run the test suite without zarr installed across every python version
test-base-all:
    for p in {{ pythons }}; do just py=$p test-base || exit 1; done

# Re-resolves uv.lock in place (restore it with `git checkout uv.lock`), and
# tests with the newest zarr-python rather than a matrix version.
# Type check and test on the newest dependencies pyproject.toml allows
test-latest:
    {{ uv }} lock --upgrade
    just py={{ py }} typecheck
    {{ run }} --group test pytest tests/test_pydantic_zarr/

# Print the installed distributions, for the CI logs
_list_env := "import importlib.metadata as m; print(*sorted(f'{d.name}=={d.version}' for d in m.distributions()), sep=chr(10))"

# Check that pydantic-zarr imports with only its required dependencies installed
test-min-reqs:
    {{ uv }} run --isolated --no-project --python {{ py }} --with . --reinstall-package pydantic-zarr \
        python -c "import pydantic_zarr.v2; import pydantic_zarr.v3"

# Run the documentation tests (doctests)
test-docs:
    {{ run }} --group docs pytest tests/test_docs

# Build the documentation
docs-build:
    {{ run }} --group docs mkdocs build --clean --strict

# Serve the documentation locally, rebuilding on changes
docs-serve:
    {{ run }} --group docs mkdocs serve --watch src

# Check that changelog entries are named correctly
changelog-check:
    {{ uv }} run --no-project --python {{ py }} python ci/check_changelog_entries.py

# Check that a pull request title follows Conventional Commits (defaults to $PR_TITLE)
pr-title-check title=env("PR_TITLE", ""):
    {{ uv }} run --no-project --python {{ py }} python ci/check_pr_title.py {{ quote(title) }}

# Fold the changelog fragments into docs/release-notes.md as release <version>
# (the release workflow runs this on the release pull request; pass --draft to preview)
release-notes version *args:
    uvx towncrier@26.9.0 build --version {{ version }} {{ args }}

# uv is pinned so a new uv release's lock-format or resolver changes cannot
# fail (or wrongly pass) unrelated pull requests.
# Update uv.lock after a change to pyproject.toml
lock:
    {{ uv }} lock

# Check that uv.lock matches pyproject.toml
lock-check:
    {{ uv }} lock --check

# Run every CI check locally (the test matrix without coverage)
ci: lint lock-check changelog-check typecheck test-docs test-min-reqs test-all test-base-all

# Build the sdist and wheel distributions
build:
    {{ uv }} build

# Remove build, test, and docs artifacts
clean:
    rm -rf dist site htmlcov .coverage*
